import math
from typing import Optional, Tuple, Union

import torch
import sys
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F


from .pscan_2d import selective_scan_fn as _selective_scan_cuda
USE_CUDA_PSCAN = True

class Mamba2D(nn.Module):
    """A 2D selective state-space mixer compatible with Vim4Path blocks."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 1,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        layer_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        input_resolution: Optional[Tuple[int, int]] = None,
        num_prefix_tokens: int = 0,
        num_suffix_tokens: int = 0,
        use_2d_mamba: bool = True,
        **_: dict,
    ) -> None:
        super().__init__()

        if not use_2d_mamba:
            raise ValueError("Mamba2D should only be used when 2D mode is enabled.")
        if input_resolution is None:
            raise ValueError("input_resolution must be provided for 2D Mamba.")

        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        if dt_rank in ("auto", 0):
            self.dt_rank = math.ceil(self.d_model / 16)
        else:
            self.dt_rank = int(dt_rank)
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.layer_idx = layer_idx
        self.height, self.width = input_resolution
        self.num_prefix_tokens = num_prefix_tokens
        self.num_suffix_tokens = num_suffix_tokens
        self.num_patches = self.height * self.width

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank ** -0.5 * self.dt_scale
        if self.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif self.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Unsupported dt_init: {self.dt_init}")

        tensor_kwargs = {}
        if device is not None:
            tensor_kwargs["device"] = device
        if dtype is not None:
            tensor_kwargs["dtype"] = dtype
        dt = torch.exp(
            torch.rand(self.d_inner, **tensor_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        a = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device)
        a = a.repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def allocate_inference_cache(self, *args, **kwargs):
        return {}

    def _split_tokens(self, hidden_states: torch.Tensor):
        prefix = None
        suffix = None
        core = hidden_states
        if self.num_prefix_tokens > 0:
            prefix = core[:, : self.num_prefix_tokens, :]
            core = core[:, self.num_prefix_tokens :, :]
        if self.num_suffix_tokens > 0:
            suffix = core[:, -self.num_suffix_tokens :, :]
            core = core[:, : -self.num_suffix_tokens, :]
        return core, prefix, suffix

    def _merge_tokens(self, core: torch.Tensor, prefix: Optional[torch.Tensor], suffix: Optional[torch.Tensor]) -> torch.Tensor:
        output = core
        if suffix is not None:
            output = torch.cat([output, suffix], dim=1)
        if prefix is not None:
            output = torch.cat([prefix, output], dim=1)
        return output

    def forward(self, hidden_states: torch.Tensor, inference_params=None) -> torch.Tensor:
        if inference_params is not None:
            raise NotImplementedError("2D Mamba does not support streaming inference.")

        tokens, prefix, suffix = self._split_tokens(hidden_states)
        batch, seq_len, _ = tokens.shape

        def _is_square(n: int) -> bool:
            s = int(math.sqrt(n))
            return s * s == n

        # Dynamic 2D grid inference to support multi-crop (e.g., 224 and 96)
        height = width = None
        if seq_len == self.num_patches:
            height, width = self.height, self.width
        else:
            # try direct perfect square
            if _is_square(seq_len):
                side = int(math.sqrt(seq_len))
                height, width = side, side
            else:
                # Fallback: some callers may have forgotten to declare CLS as prefix token
                # Try removing a leading CLS token
                tried_head = False
                if seq_len > 1 and _is_square(seq_len - 1):
                    tried_head = True
                    extra_prefix = tokens[:, :1, :]
                    tokens = tokens[:, 1:, :]
                    prefix = extra_prefix if prefix is None else torch.cat([prefix, extra_prefix], dim=1)
                    seq_len = tokens.shape[1]
                # If still not square, try removing a trailing token as suffix
                if not _is_square(seq_len):
                    # revert head removal if it didn't help
                    if tried_head:
                        tokens = torch.cat([prefix[:, -1:, :], tokens], dim=1) if prefix is not None else tokens
                        # remove the last added prefix token from prefix if present
                        if prefix is not None and prefix.shape[1] > 0:
                            prefix = prefix[:, :-1, :] if prefix.shape[1] > 1 else None
                        seq_len = tokens.shape[1]
                    if seq_len > 1 and _is_square(seq_len - 1):
                        extra_suffix = tokens[:, -1:, :]
                        tokens = tokens[:, :-1, :]
                        suffix = extra_suffix if suffix is None else torch.cat([extra_suffix, suffix], dim=1)
                        seq_len = tokens.shape[1]
                # Final check
                if _is_square(seq_len):
                    side = int(math.sqrt(seq_len))
                    height, width = side, side
                else:
                    raise ValueError(
                        f"Token length {seq_len} is not a perfect square; cannot infer 2D shape."
                    )

        x = tokens.view(batch, height, width, self.d_model)
        out = self._forward_2d(x)
        out = out.view(batch, seq_len, self.d_model)
        return self._merge_tokens(out, prefix, suffix)

    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        batch, height, width, _ = x_branch.shape
        x_branch = x_branch.view(batch, height * width, self.d_inner).transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, : height * width]
        x_branch = x_branch.view(batch, self.d_inner, height, width).permute(0, 2, 3, 1)
        x_branch = F.silu(x_branch)

        y = self._ssm_2d(x_branch)
        z_branch = F.silu(z_branch)
        y = y * z_branch
        return self.out_proj(y)

    def _ssm_2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform 2D selective scan operation.
        
        Uses CUDA-accelerated version if available for significant speedup,
        otherwise falls back to PyTorch implementation.
        """
        a = -torch.exp(self.A_log.float()).to(dtype=x.dtype, device=x.device)
        d = self.D.float()  # Force float32 for stability

        delta_bc = self.x_proj(x)
        delta, b_param, c_param = torch.split(
            delta_bc, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        # IMPORTANT: Project delta without bias (bias will be applied by CUDA kernel)
        # This matches the 2DMamba paper implementation
        batch, height, width, _ = delta.shape
        delta = delta.view(batch * height * width, self.dt_rank)
        delta = F.linear(delta, self.dt_proj.weight, bias=None)  # No bias here!
        delta = delta.view(batch, height, width, self.d_inner)

        if USE_CUDA_PSCAN and _selective_scan_cuda is not None:
            # Use CUDA-accelerated selective scan for much faster computation
            y = self._ssm_2d_cuda(x, delta, a, b_param, c_param, d)
        else:
            # Fall back to PyTorch implementation
            delta = F.softplus(delta)  # Apply softplus in PyTorch path
            delta_unsqueezed = delta.unsqueeze(-1)
            a_expanded = a.view(1, 1, 1, self.d_inner, self.d_state)
            delta_a = torch.exp(delta_unsqueezed * a_expanded)
            delta_b = delta_unsqueezed * b_param.unsqueeze(3)
            bx = delta_b * x.unsqueeze(-1)

            hs = pscan_2d(delta_a, bx)
            y = torch.matmul(hs, c_param.unsqueeze(-1)).squeeze(-1)
        
        return y + d * x
    
    def _ssm_2d_cuda(self, x: torch.Tensor, delta: torch.Tensor, a: torch.Tensor,
                     b_param: torch.Tensor, c_param: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """
        CUDA-accelerated 2D selective scan with full forward+backward support.
        
        This uses the SelectiveScanFn which wraps both pscan_cuda.fwd and pscan_cuda.bwd
        for complete training acceleration.
        
        Based on the official implementation in 2DMamba/2DMambaMIL/models/pscan_2d.py
        """
        batch, height, width, d_inner = x.shape
        
        # Reshape for CUDA kernel
        # u: (batch, d_inner, height * width)
        u = x.view(batch, height * width, d_inner).transpose(1, 2).contiguous()
        
        # delta: (batch, d_inner, height * width) - NOT softplus yet, kernel will apply it
        delta_flat = delta.view(batch, height * width, d_inner).transpose(1, 2).contiguous()
        
        # A: (d_inner, d_state)
        A = a.view(self.d_inner, self.d_state).contiguous()
        
        # B: (batch, 1, d_state, height * width)
        B = b_param.view(batch, height * width, self.d_state).permute(0, 2, 1).unsqueeze(1).contiguous()
        
        # C: (batch, 1, d_state, height * width)  
        C = c_param.view(batch, height * width, self.d_state).permute(0, 2, 1).unsqueeze(1).contiguous()
        
        # Call selective_scan_2d - includes both forward and backward CUDA acceleration!
        # This is the complete 2DMamba implementation
        out = _selective_scan_cuda(
            u,                              # input
            delta_flat,                     # delta (before softplus, without bias)
            A,                              # A matrix
            B,                              # B matrix
            C,                              # C matrix
            d.contiguous(),                 # D parameter
            None,                           # z (not used in this version)
            self.dt_proj.bias.float(),      # delta_bias - CUDA kernel applies this!
            delta_softplus=True,            # kernel applies softplus(delta + bias)
            return_last_state=False,        # we don't need the last state
            HH=height,                      # image height for 2D scan
            WW=width                        # image width for 2D scan
        )
        
        # Reshape output back: (batch, d_inner, height * width) -> (batch, height, width, d_inner)
        y = out.transpose(1, 2).view(batch, height, width, d_inner)
        
        return y
