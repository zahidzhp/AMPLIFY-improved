import os
import math
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf

from amplify.models.encoders.vision_encoders import VisionEncoder
from amplify.models.encoders.t5 import T5
from amplify.models.forward_dynamics import ForwardDynamics
from amplify.models.inverse_dynamics import InverseDynamics
from amplify.models.motion_tokenizer import MotionTokenizer
from amplify.models.ctclai import CTCLAIHeads, CTCLAIInferenceConfig
from amplify.utils.cfg_utils import get_device
from amplify.utils.data_utils import velocities_to_points
from amplify.utils.train import get_root_dir



class AMPLIFY(nn.Module):
    """
    Unified policy that bundles encoders, forward dynamics, inverse dynamics, and
    the full MotionTokenizer VAE for optional trajectory decoding/visualization.

    Public API:
      - AMPLIFY.load(path, device='auto', compile=False)
      - AMPLIFY.bundle(mt_ckpt, fd_ckpt, id_ckpt, save_to=None)
      - act(images, proprio, text=None, text_emb=None) -> (b, action_horizon, action_dim)
      - predict_codes(images, text=None, text_emb=None) -> (b, pred_seq_len), (b, pred_seq_len, hidden_dim)
      - predict_traj(images, init_queries, text=None, text_emb=None) -> (b, v, t+1, n, d)
    """

    def __init__(
        self,
        motion_tokenizer_cfg: OmegaConf,
        fd_cfg: OmegaConf,
        id_cfg: OmegaConf,
        vision_encoder_cfg: Dict[str, Any],
        text_encoder_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        # Save configs for checkpointing
        self.motion_tokenizer_cfg = motion_tokenizer_cfg
        self.fd_cfg = fd_cfg
        self.id_cfg = id_cfg
        self.vision_encoder_cfg = dict(vision_encoder_cfg)
        self.text_encoder_cfg = dict(text_encoder_cfg or {})

        # Vision encoder
        self.img_encoder = VisionEncoder(**self.vision_encoder_cfg).eval()

        # Text encoder (instantiate only if FD config does not use preprocessed embeddings)
        self.text_encoder = None
        self.fd_preprocessed_embs = True
        self.fd_preprocessed_embs = bool(fd_cfg.forward_dynamics.text_encoder.use_preprocessed_embs)

        if not self.fd_preprocessed_embs:
            self.text_encoder = T5(**self.text_encoder_cfg).eval()

        # Forward Dynamics
        num_views = len(motion_tokenizer_cfg.cond_cameraviews)
        text_seq_len = 1 if self.text_encoder is None else self.text_encoder.seq_len
        cond_seq_len = self.img_encoder.seq_len * num_views + text_seq_len
        pred_seq_len = motion_tokenizer_cfg.track_pred_horizon - 1  # velocities

        self.motion_tokenizer = MotionTokenizer(motion_tokenizer_cfg, load_encoder=True, load_decoder=True).eval()

        self.forward_dynamics = ForwardDynamics(
            trunk_cfg=fd_cfg.forward_dynamics.transformer,
            hidden_dim=motion_tokenizer_cfg.hidden_dim,
            img_dim=self.img_encoder.embed_dim,
            text_dim=(self.text_encoder.embed_dim if self.text_encoder is not None else 512),
            cond_seq_len=cond_seq_len,
            pred_seq_len=pred_seq_len,
            codebook_size=motion_tokenizer_cfg.codebook_size,
            quantize=self.motion_tokenizer.quantize
        )

        # Inverse Dynamics
        self.inverse_dynamics = InverseDynamics(motion_tokenizer_cfg, id_cfg)

        # Optional CTCLAI reranking heads (disabled by default)
        self.ctclai: Optional[CTCLAIHeads] = None
        self.ctclai_cfg: CTCLAIInferenceConfig = CTCLAIInferenceConfig(enabled=False)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def _encode_images(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes images and returns tokens for FD and ID respectively.

        Returns:
          img_tokens_fd: (b, v*t, d)
          img_tokens_id: (b, t, v*d)
        """
        b, v = images.shape[0], images.shape[1]
        img = rearrange(images, 'b v h w c -> (b v) h w c')
        img_tokens = self.img_encoder(img)  # (b*v, t, d)
        img_tokens_fd = rearrange(img_tokens, '(b v) t d -> b (v t) d', b=b, v=v)
        img_tokens_id = rearrange(img_tokens, '(b v) t d -> b t (v d)', b=b, v=v)
        return img_tokens_fd, img_tokens_id

    def _encode_text(self, b: int, text: Optional[List[str]], text_emb: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Match training-time logic with graceful fallbacks.
        - If FD was trained WITHOUT preprocessed embeddings, prefer T5 on raw text; if text is absent but
          a preprocessed embedding is provided, use it as a fallback for robustness in tests.
        - If FD was trained WITH preprocessed embeddings, use the provided `text_emb`.
        """
        # No preprocessed embeddings during training -> must compute from raw text via T5
        if not self.fd_preprocessed_embs:
            if self.text_encoder is None:
                raise RuntimeError("FD expects raw text embeddings but text encoder is not instantiated.")
            assert text is not None, "text strings must be provided when FD is configured for raw text embeddings"
            return self.text_encoder(text).unsqueeze(1).to(self.device)

        # Preprocessed embeddings expected -> text_emb must be provided
        assert text_emb is not None, "text_emb must be provided when FD is configured for preprocessed embeddings"
        te = text_emb.to(self.device)
        if te.dim() == 2:
            te = te.unsqueeze(1)
        return te

    def _predict_codes(self, img_tokens_fd: torch.Tensor, text_tokens: torch.Tensor, ar_sampling: str) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = {"image": img_tokens_fd}
        goal = {"text_emb": text_tokens}
        pred_indices = self.forward_dynamics.predict(
            self.forward_dynamics.get_cond_tokens(obs, goal), ar_sampling=ar_sampling
        )
        pred_codes = self.motion_tokenizer.quantize.indices_to_codes(pred_indices)
        return pred_indices, pred_codes

    def _build_id_inputs(self, img_tokens_id: torch.Tensor, text_tokens: torch.Tensor, proprio: torch.Tensor, codes: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = img_tokens_id.shape[0]
        if proprio is None:
            proprio = torch.zeros(b, self.id_cfg.proprio_dim, device=self.device)
        return {
            'img_tokens': img_tokens_id,
            'text_tokens': text_tokens,
            'proprioception': proprio.to(self.device).unsqueeze(1),
            'codes': codes.to(self.device),
        }

    def enable_ctclai(
        self,
        ctclai_checkpoint: str,
        *,
        n_samples: int = 8,
        lambda_tok: float = 1.0,
        lambda_risk: float = 0.5,
        lambda_prior: float = 0.1,
        entropy_weighted_tok: bool = True,
        risk_discount: float = 0.99,
    ) -> None:
        """Enable CTCLAI reranking at inference.

        Args:
            ctclai_checkpoint: path to a CTCLAI stage-4 checkpoint produced by train_ctclai.py.
        """
        ckpt = torch.load(ctclai_checkpoint, map_location=str(self.device), weights_only=False)

        token_seq_len = self.motion_tokenizer_cfg.track_pred_horizon - 1
        if getattr(self.motion_tokenizer_cfg, 'per_view', False):
            token_seq_len *= len(self.motion_tokenizer_cfg.cond_cameraviews)

        self.ctclai = CTCLAIHeads(
            hidden_dim=self.motion_tokenizer_cfg.hidden_dim,
            action_dim=self.id_cfg.action_dim,
            action_horizon=self.motion_tokenizer_cfg.true_horizon,
            token_seq_len=token_seq_len,
            codebook_size=self.motion_tokenizer_cfg.codebook_size,
        ).to(self.device).eval()

        load_res = self.ctclai.load_state_dict(ckpt['model'], strict=False)
        if load_res.missing_keys or load_res.unexpected_keys:
            print('[AMPLIFY.enable_ctclai] missing keys:', load_res.missing_keys)
            print('[AMPLIFY.enable_ctclai] unexpected keys:', load_res.unexpected_keys)

        self.ctclai_cfg = CTCLAIInferenceConfig(
            enabled=True,
            n_samples=int(n_samples),
            lambda_tok=float(lambda_tok),
            lambda_risk=float(lambda_risk),
            lambda_prior=float(lambda_prior),
            entropy_weighted_tok=bool(entropy_weighted_tok),
            risk_discount=float(risk_discount),
        )

    def disable_ctclai(self) -> None:
        self.ctclai = None
        self.ctclai_cfg = CTCLAIInferenceConfig(enabled=False)

    @torch.no_grad()
    def _act_ctclai(
        self,
        input_dict: Dict[str, torch.Tensor],
        *,
        plan_indices: torch.Tensor,
    ) -> torch.Tensor:
        """CTCLAI reranking for a single policy step.

        Args:
            input_dict: inverse-dynamics inputs (same as baseline)
            plan_indices: (B, L) predicted motion token ids from forward dynamics
        """
        if self.ctclai is None or not self.ctclai_cfg.enabled:
            return self.inverse_dynamics.act(input_dict)

        if getattr(self.id_cfg.head, 'type', None) != 'gaussian':
            raise NotImplementedError(
                f"CTCLAI sampling is implemented for Gaussian inverse head only; got head.type={getattr(self.id_cfg.head, 'type', None)}"
            )

        # 1) Sample candidate action chunks from inverse dynamics distribution.
        dist = self.inverse_dynamics(input_dict)  # TruncatedNormal
        N = int(self.ctclai_cfg.n_samples)
        candidates = dist.sample(sample_shape=torch.Size([N]))  # (N, B, T, A)

        B = candidates.shape[1]
        T = candidates.shape[2]
        A = candidates.shape[3]
        L = plan_indices.shape[1]

        # 2) Build per-state features (encode once, reuse for all candidates).
        # Image pooled feature
        if getattr(self.id_cfg, 'cond_on_img', False) and hasattr(self.inverse_dynamics, 'img_proj') and self.inverse_dynamics.img_proj is not None:
            img_tokens_h = self.inverse_dynamics.img_proj(input_dict['img_tokens'])  # (B, S, H)
            img_feat = img_tokens_h.mean(dim=1)  # (B, H)
        else:
            img_feat = torch.zeros((B, self.motion_tokenizer_cfg.hidden_dim), device=self.device)

        # Proprio pooled feature
        if getattr(self.id_cfg, 'cond_on_proprio', False) and hasattr(self.inverse_dynamics, 'proprio_proj') and self.inverse_dynamics.proprio_proj is not None:
            proprio_h = self.inverse_dynamics.proprio_proj(input_dict['proprioception'])  # (B, 1, H)
            proprio_feat = proprio_h.squeeze(1)  # (B, H)
        else:
            proprio_feat = torch.zeros((B, self.motion_tokenizer_cfg.hidden_dim), device=self.device)

        # 3) Predict tokens / risk for each candidate and score.
        cand_flat = candidates.reshape(N * B, T, A)
        img_rep = img_feat.unsqueeze(0).expand(N, -1, -1).reshape(N * B, -1)
        prop_rep = proprio_feat.unsqueeze(0).expand(N, -1, -1).reshape(N * B, -1)

        token_logits, risk_logits = self.ctclai(img_rep, prop_rep, cand_flat)

        # Token consistency cost: CE between predicted token distribution and the plan token IDs.
        plan_rep = plan_indices.long().unsqueeze(0).expand(N, -1, -1).reshape(N * B, L)
        logp = F.log_softmax(token_logits, dim=-1)  # (N*B, L, V)
        nll = -logp.gather(-1, plan_rep.unsqueeze(-1)).squeeze(-1)  # (N*B, L)

        if self.ctclai_cfg.entropy_weighted_tok:
            p = logp.exp()
            entropy = -(p * logp).sum(dim=-1)  # (N*B, L)
            norm_entropy = entropy / max(1e-6, math.log(self.motion_tokenizer_cfg.codebook_size))
            w = (1.0 - norm_entropy).detach().clamp(min=0.0)
            c_tok = (w * nll).sum(dim=-1) / (w.sum(dim=-1) + 1e-8)
        else:
            c_tok = nll.mean(dim=-1)

        # Risk cost: discounted mean predicted risk over the action horizon.
        risk_prob = torch.sigmoid(risk_logits)  # (N*B, T)
        if risk_prob.shape[1] != self.motion_tokenizer_cfg.true_horizon:
            # Be conservative: only use the overlapping part.
            risk_prob = risk_prob[:, : self.motion_tokenizer_cfg.true_horizon]
        t_risk = risk_prob.shape[1]
        discount = float(self.ctclai_cfg.risk_discount)
        w_risk = (discount ** torch.arange(t_risk, device=self.device)).view(1, -1)
        c_risk = (risk_prob * w_risk).mean(dim=-1)

        # Prior cost: discourage outliers under the base inverse policy.
        if hasattr(dist, 'normalized_log_prob'):
            lp = dist.normalized_log_prob(candidates)  # (N, B, T, A)
        else:
            lp = dist.log_prob(candidates)
        c_prior = -lp.sum(dim=(-1, -2)).reshape(N * B)  # sum over (T, A)

        total = (
            self.ctclai_cfg.lambda_tok * c_tok
            + self.ctclai_cfg.lambda_risk * c_risk
            + self.ctclai_cfg.lambda_prior * c_prior
        ).view(N, B)

        best_idx = total.argmin(dim=0)  # (B,)
        batch_idx = torch.arange(B, device=self.device)
        best_actions = candidates[best_idx, batch_idx]  # (B, T, A)
        return best_actions

    @torch.no_grad()
    def act(
        self,
        images: torch.Tensor,  # (b, v, h, w, c) float in [0, 1]
        proprio: torch.Tensor = None,  # (b, d)
        text: Optional[List[str]] = None, # list of strings of length b
        text_emb: Optional[torch.Tensor] = None, # (b, hidden_dim) or (b, 1, hidden_dim)
        ar_sampling: str = 'argmax',
    ) -> torch.Tensor:
        """Returns full-horizon actions: (b, action_horizon, action_dim)."""

        b = images.shape[0]
        img_tokens_fd, img_tokens_id = self._encode_images(images)
        text_tokens = self._encode_text(b, text=text, text_emb=text_emb)
        pred_indices, pred_codes = self._predict_codes(img_tokens_fd, text_tokens, ar_sampling)
        input_dict = self._build_id_inputs(img_tokens_id, text_tokens, proprio, pred_codes)

        # Baseline decode, or CTCLAI reranking if enabled.
        if self.ctclai is None or not self.ctclai_cfg.enabled:
            actions = self.inverse_dynamics.act(input_dict)
        else:
            actions = self._act_ctclai(input_dict, plan_indices=pred_indices)

        return actions

    @torch.no_grad()
    def predict_codes(
        self,
        images: torch.Tensor,
        text: Optional[List[str]] = None,
        text_emb: Optional[torch.Tensor] = None,
        ar_sampling: str = 'argmax',
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        b = images.shape[0]
        img_tokens_fd, _ = self._encode_images(images)
        text_tokens = self._encode_text(b, text=text, text_emb=text_emb)

        return self._predict_codes(img_tokens_fd, text_tokens, ar_sampling)

    @torch.no_grad()
    def predict_traj(
        self,
        images: torch.Tensor,
        init_queries: torch.Tensor,
        text: Optional[List[str]] = None,
        text_emb: Optional[torch.Tensor] = None,
        ar_sampling: str = 'argmax',
    ) -> torch.Tensor:
        """Returns predicted trajectories (points) with shape (b, v, t+1, n_tracks, pt_dim)."""
        _, pred_codes = self.predict_codes(images, text=text, text_emb=text_emb, ar_sampling=ar_sampling)
        vel, _ = self.motion_tokenizer.decode(pred_codes)  # velocities
        pred_traj = velocities_to_points(vel, time_dim=2, init_points=init_queries[:, :, [0]])

        return pred_traj

    def _config_snapshot(self) -> Dict[str, Any]:
        return {
            'motion_tokenizer_cfg': OmegaConf.to_container(self.motion_tokenizer_cfg, resolve=True, throw_on_missing=True),
            'forward_dynamics_cfg': OmegaConf.to_container(self.fd_cfg, resolve=True, throw_on_missing=True),
            'inverse_dynamics_cfg': OmegaConf.to_container(self.id_cfg, resolve=True, throw_on_missing=True),
            'vision_encoder_cfg': self.vision_encoder_cfg,
            'text_encoder_cfg': self.text_encoder_cfg,

        }

    def save(self, path: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        os.makedirs(os.path.join("checkpoints", "AMPLIFY"), exist_ok=True)
        save_path = path or os.path.join("checkpoints", "AMPLIFY", "latest.pt")
        ckpt = {
            'config': self._config_snapshot(),
            'model': self.state_dict(),
            'metadata': metadata or {},
        }
        torch.save(ckpt, save_path)
        return save_path

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None, compile: bool = False) -> "AMPLIFY":
        device = get_device() if device is None else device
        ckpt = torch.load(path, map_location=str(device), weights_only=False)
        cfg = ckpt['config']

        motion_tokenizer_cfg = OmegaConf.create(cfg['motion_tokenizer_cfg'])
        fd_cfg = OmegaConf.create(cfg['forward_dynamics_cfg'])
        id_cfg = OmegaConf.create(cfg['inverse_dynamics_cfg'])
        vision_encoder_cfg = cfg['vision_encoder_cfg']
        text_encoder_cfg = cfg['text_encoder_cfg']


        model = cls(
            motion_tokenizer_cfg=motion_tokenizer_cfg,
            fd_cfg=fd_cfg,
            id_cfg=id_cfg,
            vision_encoder_cfg=vision_encoder_cfg,
            text_encoder_cfg=text_encoder_cfg,

        )
        load_res = model.load_state_dict(ckpt['model'], strict=False)
        if load_res.missing_keys or load_res.unexpected_keys:
            print("[AMPLIFY.load] missing keys:", load_res.missing_keys)
            print("[AMPLIFY.load] unexpected keys:", load_res.unexpected_keys)
        model.to(device).eval()

        if compile:
            for m in [model.img_encoder, model.text_encoder, model.forward_dynamics, model.inverse_dynamics, model.motion_tokenizer]:
                if m is not None:
                    try:
                        torch.compile(m)
                    except Exception:
                        pass
        return model

    @classmethod
    def bundle(
        cls,
        motion_tokenizer_ckpt: str,
        forward_dynamics_ckpt: str,
        inverse_dynamics_ckpt: str,
        save_to: Optional[str] = None,
    ) -> Tuple["AMPLIFY", str]:
        """
        Assemble an AMPLIFY model from existing checkpoints. Returns (model, save_path).
        """
        root = get_root_dir()
        device = get_device()

        # Load configs
        mt_path = motion_tokenizer_ckpt if os.path.isabs(motion_tokenizer_ckpt) else os.path.join(root, motion_tokenizer_ckpt)
        fd_path = forward_dynamics_ckpt
        id_path = inverse_dynamics_ckpt

        mt_ckpt = torch.load(mt_path, map_location=str(device), weights_only=False)
        motion_tokenizer_cfg = OmegaConf.create(mt_ckpt['config'])

        fd_ckpt = torch.load(fd_path, map_location=str(device), weights_only=False)
        fd_cfg = OmegaConf.create(fd_ckpt['config'])

        id_ckpt = torch.load(id_path, map_location=str(device), weights_only=False)
        id_cfg = OmegaConf.create(id_ckpt['config'])

        # Build model skeleton
        vision_encoder_cfg = OmegaConf.to_container(fd_cfg.forward_dynamics.vision_encoder, resolve=True)
        text_encoder_cfg = OmegaConf.to_container(fd_cfg.forward_dynamics.text_encoder, resolve=True)
        model = cls(
            motion_tokenizer_cfg=motion_tokenizer_cfg,
            fd_cfg=fd_cfg,
            id_cfg=id_cfg,
            vision_encoder_cfg=vision_encoder_cfg,
            text_encoder_cfg=text_encoder_cfg,

        ).to(device)

        # Load submodule weights
        fd_res = model.forward_dynamics.load_state_dict(fd_ckpt['model'], strict=False)
        if fd_res.missing_keys or fd_res.unexpected_keys:
            print("[AMPLIFY.bundle] FD missing keys:", fd_res.missing_keys)
            print("[AMPLIFY.bundle] FD unexpected keys:", fd_res.unexpected_keys)
        id_res = model.inverse_dynamics.load_state_dict(id_ckpt['model'], strict=False)
        if id_res.missing_keys or id_res.unexpected_keys:
            print("[AMPLIFY.bundle] ID missing keys:", id_res.missing_keys)
            print("[AMPLIFY.bundle] ID unexpected keys:", id_res.unexpected_keys)

        mt_res = model.motion_tokenizer.load_state_dict(mt_ckpt['model'], strict=False)
        if mt_res.missing_keys or mt_res.unexpected_keys:
            print("[AMPLIFY.bundle] MT missing keys:", mt_res.missing_keys)
            print("[AMPLIFY.bundle] MT unexpected keys:", mt_res.unexpected_keys)

        # Finalize
        model.eval()

        metadata = {
            'source_checkpoints': {
                'motion_tokenizer': motion_tokenizer_ckpt,
                'forward_dynamics': forward_dynamics_ckpt,
                'inverse_dynamics': inverse_dynamics_ckpt,
            }
        }
        save_path = model.save(path=save_to or os.path.join("checkpoints", "AMPLIFY", "latest.pt"), metadata=metadata)
        return model, save_path
