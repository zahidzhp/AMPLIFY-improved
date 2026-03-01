import math
import os

import hydra
import pyinstrument
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

import wandb

from amplify.models.ctclai import CTCLAIHeads
from amplify.models.encoders.vision_encoders import VisionEncoder
from amplify.models.inverse_dynamics import InverseDynamics
from amplify.models.motion_tokenizer import load_motion_tokenizer
from amplify.utils.cfg_utils import get_device
from amplify.utils.logger import Logger
from amplify.utils.train import (
    get_checkpoint_dir,
    get_dataloaders,
    get_datasets,
    get_root_dir,
    latest_checkpoint_from_dir,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
)


def _make_risk_labels(start_t: torch.Tensor, rollout_len: torch.Tensor, horizon: int) -> torch.Tensor:
    """Produce shaped per-step risk labels from offline demo metadata.

    We interpret 'success time' as the end of the demo episode. Then:
      risk(t+k) = 1 if (t+k) is before the terminal step, else 0.

    This gives non-trivial supervision even when demos are all successful.
    """
    # start_t, rollout_len are (B,) int tensors.
    device = start_t.device
    # terminal index (inclusive)
    t_star = rollout_len - 1
    k = torch.arange(1, horizon + 1, device=device).view(1, horizon)  # (1, T)
    t_future = start_t.view(-1, 1) + k  # (B, T)
    # 1 until terminal, then 0
    y = (t_future < t_star.view(-1, 1)).float()
    return y


def train_epoch(
    train_global_iter,
    models,
    train_loader,
    optimizer,
    scaler,
    device,
    logger,
    cfg,
    motion_tokenizer_cfg,
    id_cfg,
):
    models["ctclai"].train()
    grad_accum = math.ceil(cfg.batch_size / cfg.gpu_max_bs)
    optimizer.zero_grad(set_to_none=True)

    running_total = 0.0
    running_tok = 0.0
    running_risk = 0.0

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training (CTCLAI)")):
        actions = batch["actions"].to(device)  # (B, T, A)

        with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.float16, enabled=cfg.amp and device.type != "mps"):
            # --- frozen feature extraction + token targets ---
            with torch.no_grad():
                B = actions.shape[0]

                # Image features
                if id_cfg.cond_on_img:
                    imgs = batch["images"].to(device)  # (B, V, H, W, C)
                    imgs_flat = rearrange(imgs, "b v h w c -> (b v) h w c")
                    img_tokens = models["img_encoder"](imgs_flat)  # (B*V, S, D)
                    img_tokens = rearrange(
                        img_tokens,
                        "(b v) s d -> b s (v d)",
                        b=B,
                        v=len(motion_tokenizer_cfg.cond_cameraviews),
                    )
                    img_tokens_h = models["inverse_dynamics"].img_proj(img_tokens)  # (B, S, H)
                    img_feat = img_tokens_h.mean(dim=1)  # (B, H)
                else:
                    img_feat = torch.zeros((B, motion_tokenizer_cfg.hidden_dim), device=device)

                # Proprio features
                if id_cfg.cond_on_proprio:
                    prop = batch["proprioception"].to(device).unsqueeze(1)  # (B, 1, P)
                    prop_h = models["inverse_dynamics"].proprio_proj(prop)  # (B, 1, H)
                    proprio_feat = prop_h.squeeze(1)  # (B, H)
                else:
                    proprio_feat = torch.zeros((B, motion_tokenizer_cfg.hidden_dim), device=device)

                # Token targets from ground-truth future tracks
                traj = batch["traj"].to(device)  # (B, V, Tt, N, 2)
                traj_vel = traj[:, :, 1:] - traj[:, :, :-1]  # (B, V, Tt-1, N, 2)
                if motion_tokenizer_cfg.cond_on_img:
                    # NOTE: motion tokenizer expects images shaped (B, V, H, W, C).
                    z = models["motion_tokenizer"].encode(traj_vel, imgs)
                else:
                    z = models["motion_tokenizer"].encode(traj_vel)
                _, token_targets = models["motion_tokenizer"].quantize(z)
                token_targets = token_targets.long()  # (B, L)

                # Risk labels
                start_t = batch["start_t"].to(device)
                rollout_len = batch["rollout_len"].to(device)
                risk_targets = _make_risk_labels(start_t, rollout_len, horizon=motion_tokenizer_cfg.true_horizon)

            # --- CTCLAI heads forward (trainable) ---
            token_logits, risk_logits = models["ctclai"](img_feat, proprio_feat, actions)

            # Token CE
            tok_loss = F.cross_entropy(
                token_logits.view(-1, token_logits.size(-1)),
                token_targets.view(-1),
            )

            # Risk BCE (with temporal discount weights)
            w = (cfg.risk_discount ** torch.arange(motion_tokenizer_cfg.true_horizon, device=device)).view(1, -1)
            risk_bce = F.binary_cross_entropy_with_logits(risk_logits, risk_targets, reduction="none")
            risk_loss = (risk_bce * w).mean()

            loss = (cfg.tok_loss_weight * tok_loss) + (cfg.risk_loss_weight * risk_loss)

        scaler.scale(loss).backward()
        if (batch_idx + 1) % grad_accum == 0:
            if cfg.clip_grad and cfg.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(models["ctclai"].parameters(), cfg.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_total += float(loss.detach())
        running_tok += float(tok_loss.detach())
        running_risk += float(risk_loss.detach())

        logger.update(
            {
                "train_total_loss": float(loss.detach()),
                "train_tok_loss": float(tok_loss.detach()),
                "train_risk_loss": float(risk_loss.detach()),
            },
            train_global_iter,
            phase="train",
        )
        train_global_iter += cfg.gpu_max_bs

    n = max(1, len(train_loader))
    return (
        running_total / n,
        running_tok / n,
        running_risk / n,
        train_global_iter,
    )


@torch.no_grad()
def val_epoch(
    val_global_iter,
    models,
    val_loader,
    device,
    logger,
    cfg,
    motion_tokenizer_cfg,
    id_cfg,
):
    models["ctclai"].eval()

    running_total = 0.0
    running_tok = 0.0
    running_risk = 0.0

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation (CTCLAI)")):
        actions = batch["actions"].to(device)

        with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.float16, enabled=cfg.amp and device.type != "mps"):
            B = actions.shape[0]

            if id_cfg.cond_on_img:
                imgs = batch["images"].to(device)
                imgs_flat = rearrange(imgs, "b v h w c -> (b v) h w c")
                img_tokens = models["img_encoder"](imgs_flat)
                img_tokens = rearrange(
                    img_tokens,
                    "(b v) s d -> b s (v d)",
                    b=B,
                    v=len(motion_tokenizer_cfg.cond_cameraviews),
                )
                img_tokens_h = models["inverse_dynamics"].img_proj(img_tokens)
                img_feat = img_tokens_h.mean(dim=1)
            else:
                img_feat = torch.zeros((B, motion_tokenizer_cfg.hidden_dim), device=device)

            if id_cfg.cond_on_proprio:
                prop = batch["proprioception"].to(device).unsqueeze(1)
                proprio_feat = models["inverse_dynamics"].proprio_proj(prop).squeeze(1)
            else:
                proprio_feat = torch.zeros((B, motion_tokenizer_cfg.hidden_dim), device=device)

            traj = batch["traj"].to(device)
            traj_vel = traj[:, :, 1:] - traj[:, :, :-1]
            if motion_tokenizer_cfg.cond_on_img:
                z = models["motion_tokenizer"].encode(traj_vel, imgs)
            else:
                z = models["motion_tokenizer"].encode(traj_vel)
            _, token_targets = models["motion_tokenizer"].quantize(z)
            token_targets = token_targets.long()

            start_t = batch["start_t"].to(device)
            rollout_len = batch["rollout_len"].to(device)
            risk_targets = _make_risk_labels(start_t, rollout_len, horizon=motion_tokenizer_cfg.true_horizon)

            token_logits, risk_logits = models["ctclai"](img_feat, proprio_feat, actions)

            tok_loss = F.cross_entropy(
                token_logits.view(-1, token_logits.size(-1)),
                token_targets.view(-1),
            )
            w = (cfg.risk_discount ** torch.arange(motion_tokenizer_cfg.true_horizon, device=device)).view(1, -1)
            risk_bce = F.binary_cross_entropy_with_logits(risk_logits, risk_targets, reduction="none")
            risk_loss = (risk_bce * w).mean()
            loss = (cfg.tok_loss_weight * tok_loss) + (cfg.risk_loss_weight * risk_loss)

        running_total += float(loss.detach())
        running_tok += float(tok_loss.detach())
        running_risk += float(risk_loss.detach())

        logger.update(
            {
                "val_total_loss": float(loss.detach()),
                "val_tok_loss": float(tok_loss.detach()),
                "val_risk_loss": float(risk_loss.detach()),
            },
            val_global_iter,
            phase="val",
        )
        val_global_iter += cfg.gpu_max_bs

    n = max(1, len(val_loader))
    return (
        running_total / n,
        running_tok / n,
        running_risk / n,
        val_global_iter,
    )


@hydra.main(config_path="cfg", config_name="train_ctclai", version_base="1.2")
def main(cfg):
    seed_everything(cfg.seed)
    run_name = f"{cfg.run_name}_seed_{cfg.seed}"

    if cfg.root_dir is None:
        cfg.root_dir = get_root_dir()

    if cfg.profile:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        profiler = pyinstrument.Profiler()

    device = get_device()
    print("Using device:", device)

    assert cfg.motion_tokenizer_checkpoint is not None, "motion_tokenizer_checkpoint must be set"
    assert cfg.inverse_dynamics_checkpoint is not None, "inverse_dynamics_checkpoint must be set"

    # --- Load frozen staged components ---
    models = {}
    models["motion_tokenizer"], motion_tokenizer_cfg = load_motion_tokenizer(cfg.motion_tokenizer_checkpoint, frozen=True)
    models["motion_tokenizer"] = models["motion_tokenizer"].to(device).eval()

    # Load inverse dynamics checkpoint to reuse projections / config.
    id_ckpt = torch.load(cfg.inverse_dynamics_checkpoint, map_location=str(device), weights_only=False)
    id_cfg = OmegaConf.create(id_ckpt["config"])

    # Vision encoder (frozen) for image tokens.
    models["img_encoder"] = VisionEncoder(**id_cfg.vision_encoder).eval().to(device)

    # Recreate inverse dynamics module to reuse its projections (frozen).
    # NOTE: we only need img_proj / proprio_proj weights for feature alignment.
    models["inverse_dynamics"] = InverseDynamics(motion_tokenizer_cfg, id_cfg).to(device).eval()
    models["inverse_dynamics"].load_state_dict(id_ckpt["model"], strict=False)
    for p in models["inverse_dynamics"].parameters():
        p.requires_grad_(False)
    for p in models["img_encoder"].parameters():
        p.requires_grad_(False)
    for p in models["motion_tokenizer"].parameters():
        p.requires_grad_(False)

    # --- CTCLAI heads (trainable) ---
    token_seq_len = motion_tokenizer_cfg.track_pred_horizon - 1
    if getattr(motion_tokenizer_cfg, "per_view", False):
        token_seq_len *= len(motion_tokenizer_cfg.cond_cameraviews)

    models["ctclai"] = CTCLAIHeads(
        hidden_dim=motion_tokenizer_cfg.hidden_dim,
        action_dim=id_cfg.action_dim,
        action_horizon=motion_tokenizer_cfg.true_horizon,
        token_seq_len=token_seq_len,
        codebook_size=motion_tokenizer_cfg.codebook_size,
        action_mlp_hidden=cfg.action_mlp_hidden,
        action_mlp_layers=cfg.action_mlp_layers,
        dropout=cfg.dropout,
    ).to(device)

    if cfg.compile:
        try:
            models["ctclai"] = torch.compile(models["ctclai"])
        except Exception:
            pass

    # --- Data ---
    keys_to_load = list(set(motion_tokenizer_cfg.keys_to_load + ["actions", "proprioception"]))
    train_datasets, val_datasets = get_datasets(
        root_dir=cfg.root_dir,
        train_datasets=cfg.train_datasets,
        val_datasets=cfg.val_datasets,
        keys_to_load=keys_to_load,
        motion_tokenizer_cfg=motion_tokenizer_cfg,
        task_names=cfg.task_names,
    )
    train_dl_dict, val_dl_dict = get_dataloaders(
        train_datasets,
        val_datasets,
        gpu_max_bs=cfg.gpu_max_bs,
        num_workers=cfg.num_workers,
        quick=cfg.quick,
    )
    train_loader = train_dl_dict["action"]
    val_loader = val_dl_dict["action"] if val_dl_dict is not None else None

    optimizer = torch.optim.AdamW(
        models["ctclai"].parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.adam_betas,
    )

    warmup_epochs = max(1, cfg.num_epochs // 10)
    if cfg.lr_schedule == "cosine":
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=cfg.lr / 100)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    elif cfg.lr_schedule is None:
        scheduler = None
    else:
        raise ValueError(f"Invalid lr_schedule: {cfg.lr_schedule}")

    scaler = torch.amp.GradScaler(enabled=cfg.amp and device.type != "mps")

    # --- Checkpointing ---
    resume = (cfg.checkpoint is not None or cfg.resume)
    checkpoint_dir = get_checkpoint_dir(stage="ctclai", run_name=run_name, resume=resume)
    print(f"Checkpoint dir: {checkpoint_dir}")
    if cfg.resume and cfg.checkpoint is None:
        cfg.checkpoint = latest_checkpoint_from_dir(checkpoint_dir)

    if cfg.checkpoint is not None:
        models["ctclai"], optimizer, scheduler, scaler, checkpoint_cfg, checkpoint_info = load_checkpoint(
            cfg.checkpoint, models["ctclai"], optimizer, scheduler, scaler
        )
        start_epoch = checkpoint_info["epoch"] + 1
        train_global_iter = checkpoint_info["train_global_iter"]
        val_global_iter = checkpoint_info["val_global_iter"]
        wandb_run_id = checkpoint_info["wandb_run_id"]
    else:
        start_epoch = 1
        train_global_iter = 0
        val_global_iter = 0
        wandb_run_id = None

    # --- Logging ---
    logger = Logger(train_log_interval=cfg.log_interval, val_log_interval=cfg.log_interval)
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger.wandb_init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        config=wandb_cfg,
        name=run_name,
        group=cfg.wandb_group,
        resume="allow" if wandb_run_id is not None else None,
        id=wandb_run_id,
    )

    if cfg.profile:
        profiler.start()

    # --- Train loop ---
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        print(f"Epoch {epoch}/{cfg.num_epochs}")
        train_total, train_tok, train_risk, train_global_iter = train_epoch(
            train_global_iter,
            models,
            train_loader,
            optimizer,
            scaler,
            device,
            logger,
            cfg,
            motion_tokenizer_cfg,
            id_cfg,
        )

        if scheduler is not None:
            scheduler.step()

        val_total = None
        if val_loader is not None:
            val_total, val_tok, val_risk, val_global_iter = val_epoch(
                val_global_iter,
                models,
                val_loader,
                device,
                logger,
                cfg,
                motion_tokenizer_cfg,
                id_cfg,
            )
            print(f"val_total={val_total:.6f} val_tok={val_tok:.6f} val_risk={val_risk:.6f}")

        print(f"train_total={train_total:.6f} train_tok={train_tok:.6f} train_risk={train_risk:.6f}")

        # Save
        if epoch % cfg.save_interval == 0 or epoch == cfg.num_epochs:
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch:04d}.pt")
            save_checkpoint(
                checkpoint_path=ckpt_path,
                epoch=epoch,
                cfg=cfg,
                model=models["ctclai"],
                optimizer=optimizer,
                scaler=scaler,
                train_loss=train_total,
                val_loss=val_total,
                train_global_iter=train_global_iter,
                val_global_iter=val_global_iter,
                scheduler=scheduler,
            )
            # Also update latest
            save_checkpoint(
                checkpoint_path=os.path.join(checkpoint_dir, "latest.pt"),
                epoch=epoch,
                cfg=cfg,
                model=models["ctclai"],
                optimizer=optimizer,
                scaler=scaler,
                train_loss=train_total,
                val_loss=val_total,
                train_global_iter=train_global_iter,
                val_global_iter=val_global_iter,
                scheduler=scheduler,
            )

    if cfg.profile:
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))


if __name__ == "__main__":
    main()
