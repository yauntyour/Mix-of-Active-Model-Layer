import torch
import torch.nn as nn
import torch.nn.functional as F
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple


# ===========================================================================
# Configuration
# ===========================================================================
@dataclass
class ModelConfig:
    """Model hyper-parameters.

    The defaults are a small research configuration for the local 6400-token
    Minimind cache.  The architecture defaults from the design document are
    ``d_model=512, n_heads=8, n_layers=10, n_groups=16, n_experts=16``.
    """

    vocab_size: int = 6400
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4  # S, parallel decoder branches
    n_groups: int = 4  # G, routed attention groups
    n_experts: int = 4  # E, MoA up/down experts
    top_k: int = 2  # K, experts per token
    max_seq_len: int = 4096
    moe_ratio: int = 4  # expert hidden dim = d_model * moe_ratio

    router_temperature: float = 1.0
    router_aux_coef: float = 1e-2
    moe_aux_coef: float = 1e-2
    aux_mean: bool = True  # divide balance coefficients by n_layers

    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if self.d_model <= 0 or self.n_heads <= 0:
            raise ValueError("d_model and n_heads must be > 0")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_model < self.n_heads:
            raise ValueError("d_model must be >= n_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for pairwise RoPE")
        if self.router_temperature <= 0:
            raise ValueError("router_temperature must be > 0")
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if self.n_groups < 1 or self.n_experts < 1:
            raise ValueError("n_groups and n_experts must be >= 1")
        if not (1 <= self.top_k <= self.n_experts):
            raise ValueError("require 1 <= top_k <= n_experts")
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be >= 1")
        if self.moe_ratio < 1:
            raise ValueError("moe_ratio must be >= 1")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def moe_hidden(self) -> int:
        return self.d_model * self.moe_ratio

    @property
    def router_coef(self) -> float:
        return (
            self.router_aux_coef / self.n_layers
            if self.aux_mean
            else self.router_aux_coef
        )

    @property
    def moe_coef(self) -> float:
        return self.moe_aux_coef / self.n_layers if self.aux_mean else self.moe_aux_coef

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        """Rebuild a config, ignoring unknown keys from newer checkpoints."""
        names = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in names})


# ===========================================================================
# Load balancing
# ===========================================================================
def balance_loss(
    scores: torch.Tensor,
    assignment: torch.Tensor,
    n_options: int,
    coef: float,
) -> torch.Tensor:
    """Load-balance auxiliary loss.

    ``loss = n_options * sum_o(frac_o * mean_tokens(scores[t, o])) * coef``

    ``assignment`` holds one option id per token (groups) or per token-slot
    (MoA/MoE top-k assignments).  The loss only differentiates through
    ``scores``.
    """
    counts = torch.bincount(assignment, minlength=n_options).to(scores.dtype)
    frac = counts / assignment.numel()
    return n_options * (frac * scores.mean(dim=0)).sum() * coef


# ===========================================================================
# Mixture of active experts (MoA)
# ===========================================================================
class SwiGLUExpert(nn.Module):
    """SwiGLU FFN with separately callable up and down halves."""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.gate_up = nn.Linear(d_model, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def up(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., D) -> SwiGLU activation (..., hidden)."""
        gate, value = self.gate_up(x).chunk(2, dim=-1)
        return F.silu(gate) * value

    def down_proj(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., hidden) -> output projection (..., D)."""
        return self.down(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.up(x))


class MoA(nn.Module):
    """Two-stage token-wise mixture of active SwiGLU experts.

    Stage 1 (up): each token is routed to ``top_k`` up projections and their
    SwiGLU activations are combined with the router weights into one
    ``moe_hidden`` vector.

    Stage 2 (down): that vector is routed to ``top_k`` down projections and
    their outputs are combined with a second router.  The down logits add the
    up-router logits as a residual and ``down_gate`` starts at zero, so the
    initial down assignment matches the up assignment (preserving the base
    model); training then learns where the two stages should diverge.  Both
    routers emit a balance loss.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.coef = cfg.moe_coef
        self.d_model = cfg.d_model
        self.hidden = cfg.moe_hidden
        self.gate = nn.Linear(cfg.d_model, cfg.n_experts, bias=True)
        self.down_gate = nn.Linear(cfg.moe_hidden, cfg.n_experts, bias=True)
        self.experts = nn.ModuleList(
            [SwiGLUExpert(cfg.d_model, cfg.moe_hidden) for _ in range(cfg.n_experts)]
        )

    def _route(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        top_w, top_i = torch.topk(scores, self.top_k, dim=-1)
        top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-8)
        return top_w, top_i

    def _mix_up(
        self,
        x: torch.Tensor,
        top_w: torch.Tensor,
        top_i: torch.Tensor,
    ) -> torch.Tensor:
        """Run the selected up projections on their rows and combine them."""
        rows = x.shape[0]
        pieces: List[torch.Tensor] = []
        sources: List[torch.Tensor] = []
        for e, expert in enumerate(self.experts):
            token_idx, slot_idx = (top_i == e).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            y = expert.up(x.index_select(0, token_idx))
            pieces.append(y * top_w[token_idx, slot_idx].unsqueeze(-1))
            sources.append(token_idx)
        if not pieces:
            return x.new_zeros(rows, self.hidden)
        return x.new_zeros(rows, self.hidden).index_add(
            0, torch.cat(sources), torch.cat(pieces)
        )

    def _mix_down(
        self,
        h: torch.Tensor,
        top_w: torch.Tensor,
        top_i: torch.Tensor,
    ) -> torch.Tensor:
        """Run the selected down projections on their rows and combine them."""
        rows = h.shape[0]
        pieces: List[torch.Tensor] = []
        sources: List[torch.Tensor] = []
        for e, expert in enumerate(self.experts):
            token_idx, slot_idx = (top_i == e).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            y = expert.down_proj(h.index_select(0, token_idx))
            pieces.append(y * top_w[token_idx, slot_idx].unsqueeze(-1))
            sources.append(token_idx)
        if not pieces:
            return h.new_zeros(rows, self.d_model)
        return h.new_zeros(rows, self.d_model).index_add(
            0, torch.cat(sources), torch.cat(pieces)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (N, D) -> MoA delta (N, D), summed balance loss scalar."""
        up_logits = self.gate(x)
        up_scores = torch.softmax(up_logits, dim=-1)
        up_w, up_i = self._route(up_scores)
        h = self._mix_up(x, up_w, up_i)

        # The down router reads the combined up activation and starts from the
        # up assignment (zero-initialised down_gate), then learns to diverge.
        down_scores = torch.softmax(self.down_gate(h) + up_logits, dim=-1)
        down_w, down_i = self._route(down_scores)
        delta = self._mix_down(h, down_w, down_i)

        # Average the two balance terms so the configured coefficient keeps
        # the same total auxiliary scale as the old single-router MoE.
        aux = 0.5 * (
            balance_loss(up_scores, up_i.reshape(-1), self.n_experts, self.coef)
            + balance_loss(down_scores, down_i.reshape(-1), self.n_experts, self.coef)
        )
        return delta, aux
