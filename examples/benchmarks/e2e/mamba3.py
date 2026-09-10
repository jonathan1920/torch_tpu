# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mamba-3 sequence mixer layer for benchmarks.

Reference:
  Mamba-3: Improved Sequence Modeling using State Space Principles
  (arXiv:2603.15569).

Mamba-3 Architecture Overview:
------------------------------
State Space Models (SSMs) map a 1D continuous-time input signal x(t) to an
output signal y(t) through an N-dimensional latent state h(t):
    h'(t) = A * h(t) + B * x(t)
    y(t)  = C * h(t) + D * x(t)

Mamba-3 introduces three foundational architectural advancements over prior SSMs
(Mamba-1 and Mamba-2):

1. Exponential-Trapezoidal Discretization:
   Traditional SSMs rely exclusively on Zero-Order Hold (ZOH) discretization,
   which assumes the input x(t) is constant across the interval [t_{k-1}, t_k):
     h_k = exp(A * dt_k) * h_{k-1} + dt_k * (B * x_k)
   While computationally simple, ZOH introduces numerical errors when tracking
   high-frequency oscillations and rapidly varying inputs. Trapezoidal
   (bilinear / Tustin) integration instead approximates the input integral using
   the average of the endpoints:
     0.5 * (B * x_k + B * x_{k-1})
   Mamba-3 introduces a data-dependent, learned trapezoidal blending gate
   `trap_k` in (0, 1) that dynamically interpolates between ZOH and trapezoidal:
     u_k = dt_k * [(1 - trap_k) * (B * x_k) + trap_k * 0.5 * (B * x_k + B *
     x_{k-1})]
   This combines the bounded stability of ZOH with the higher-order numerical
   accuracy of trapezoidal integration.

2. Complex-Valued / Rotary State Space (RoPE in SSMs):
   Standard Rotary Position Embeddings (RoPE) in Transformers rotate paired
   query
   and key coordinates by fixed position frequencies: theta = position * omega.
   In continuous-time SSMs, the time step dt_k varies dynamically per token.
   Mamba-3 integrates angular velocity omega over continuous time:
     theta_k = sum_{s=0}^k dt_s * omega  =  integral_0^{t_k} omega * dt(s) ds
   The accumulated angle theta_k rotates paired coordinates of the state
   projection vectors B and C via 2D Givens rotation matrices:
     [B'_{2j}  ] = [cos(theta_j)  -sin(theta_j)] [B_{2j}  ]
     [B'_{2j+1}]   [sin(theta_j)   cos(theta_j)] [B_{2j+1}]
   This endows the recurrent memory with continuous, input-dependent phase
   dynamics.

3. Multi-Input Multi-Output (MIMO) Recurrence:
   In Single-Input Single-Output (SISO) state spaces, each head processes a
   scalar channel per step through an outer-product state update.
   MIMO generalizes this by projecting each head's channel into a rank-R
   subspace (via `mimo_x`), tracking R interacting state components per head,
   and contracting them back to output space (via `mimo_o`):
     x_r = x * mimo_x   (R sub-channels per head)
     y   = sum_r y_r * mimo_o
   Setting `is_mimo=False` seamlessly recovers standard SISO recurrence (rank
   1).

Computational Flow Diagram:
---------------------------
                 u (batch, seq_len, d_model)
                             │
                             ▼
                         in_proj  (Single batched GEMM)
                             │
       ┌───────────┬─────────┼─────────┬──────────┬──────────┬──────────┬─────────┐
       ▼           ▼         ▼         ▼          ▼          ▼          ▼
       ▼
       z           x       b_raw     c_raw      dd_dt       dd_a     trap_raw
       angle_raw
     (gate)      (SSM)      (B)       (C)        (dt)       (A)       (trap)
     (RoPE)
       │           │         │         │          │          │          │
       │
       │           │      RMSNorm   RMSNorm       │          │          │
       │
       │           │         │         │          │          │          │
       │
       │           │      + b_bias  + c_bias      │          │          │
       │
       │           │         │         │          ▼          ▼          ▼
       ▼
       │           │         │         │      softplus   -softplus   sigmoid
       │
       │           │         │         │          │          │          │
       │
       │           │         │         │          dt         A         trap
       │
       │           │         │         │          │          │          │
       │
       │           │         └────┬────┘          ├──adt─────┘          │
       │
       │           │              │               │                     │
       ▼
       │           │              │               │                     │     dt
       * angle
       │           │              │               │                     │
       │
       │           │              │               │                     │
       cumsum
       │           │              │               │                     │
       │
       │           │              └───────► Apply RoPE
       ◄────────────────┘─────────┘
       │           │                              │
       │           │                      [b_proj, c_proj]
       │           │                              │
       │           └──────────────┬───────────────┘
       │                          │
       │                          ▼
       │                     mamba3_scan
       │               (Exponential-Trapezoidal
       │                  SSM Recurrence)
       │                          │
       │                          ▼
       └────────────────────► Gated SiLU
                                  │
                                  ▼
                              out_proj
                                  │
                                  ▼
                             y (output)

Portability:
------------
This implementation is written in pure PyTorch without custom CUDA or Triton
kernels, ensuring portable, zero-overhead execution across TorchTPU (XLA),
TorchAX (JAX), and GPU platforms.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
  """Root Mean Square Layer Normalization (RMSNorm).

  Normalizes inputs across the last dimension by their root-mean-square:
    y = (x / sqrt(mean(x^2) + eps)) * weight

  RMSNorm is applied to B and C state projections to bound recurrence state
  magnitudes and stabilize gradients across deep unrolled recurrent steps.
  Calculations are executed in float32 for numerical stability across
  lower-precision modes (bfloat16, float16).
  """

  def __init__(self, dim: int, eps: float = 1e-5):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Compute root-mean-square in float32 to prevent half-precision overflow/underflow.
    rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
    # Normalize, scale by learned weight, and cast back to input tensor dtype.
    return (x.float() / rms * self.weight).to(x.dtype)


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
  """Applies Rotary Position Embedding (RoPE) to paired coordinates in x.

  Treats adjacent coordinate pairs (x_{2k}, x_{2k+1}) as 2D vectors (or complex
  numbers x_{2k} + i * x_{2k+1}) and rotates them by angle theta_k:
    [x'_{2k}  ] = [cos(theta_k)  -sin(theta_k)] [x_{2k}  ]
    [x'_{2k+1}]   [sin(theta_k)   cos(theta_k)] [x_{2k+1}]

  Args:
    x: Coordinate tensor of shape (..., 2 * num_angles).
    angles: Rotation angle tensor of shape (..., num_angles).

  Returns:
    Rotated coordinate tensor matching the shape and dtype of x.
  """
  cos = torch.cos(angles)
  sin = torch.sin(angles)

  # Separate into even (real) and odd (imaginary) coordinate components.
  x1 = x[..., 0::2]  # (..., num_angles)
  x2 = x[..., 1::2]  # (..., num_angles)

  # Apply 2D planar rotation matrix multiplication.
  x_rot_1 = x1 * cos - x2 * sin
  x_rot_2 = x1 * sin + x2 * cos

  # Interleave rotated pairs back into original coordinate order:
  # (..., num_angles, 2) -> flatten last two dims -> (..., 2 * num_angles).
  out = torch.stack([x_rot_1, x_rot_2], dim=-1)
  return out.flatten(-2)


def mamba3_scan(
    x: torch.Tensor,
    b_proj: torch.Tensor,
    c_proj: torch.Tensor,
    adt: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    d_skip: torch.Tensor,
    z: torch.Tensor | None = None,
    mimo_x: torch.Tensor | None = None,
    mimo_z: torch.Tensor | None = None,
    mimo_o: torch.Tensor | None = None,
    is_mimo: bool = False,
) -> torch.Tensor:
  """Executes the Mamba-3 SSM sequential scan recurrence.

  SSM Continuous-Time State Formulation:
    h'(t) = A * h(t) + B * x(t)
    y(t)  = C * h(t) + D * x(t)

  Discretization with Exponential Decay and Trapezoidal Blending:
    decay_t = exp(A * dt_t)               # Transition matrix (multiplicative
    decay)
    (Bx)_t  = B_t * x_t                   # Current token state input
    (Bx)_blended = (1 - trap_t) * (Bx)_t + trap_t * 0.5 * ((Bx)_t + (Bx)_{t-1})
    u_t     = dt_t * (Bx)_blended         # Discretized input injection
    h_t     = decay_t * h_{t-1} + u_t     # Hidden state recurrence update
    y_t     = (C_t * h_t + D * x_t) * silu(z_t) # Output projection & gating

  Tensor Dimensions:
    B: batch_size
    L: seq_len
    H: num_heads
    P: head_dim
    D: d_state (SSM latent memory dimension per head)
    R: mimo_rank (number of interacting sub-channels per head)

  Args:
    x: Input values tensor of shape (B, L, H, P).
    b_proj: Key projection B tensor of shape (B, L, R, H, D).
    c_proj: Query projection C tensor of shape (B, L, R, H, D).
    adt: Log decay tensor A * dt of shape (B, L, H).
    dt: Time step dt tensor of shape (B, L, H).
    trap: Trapezoidal blending gate tensor of shape (B, L, H) in (0, 1).
    d_skip: Direct skip connection weight of shape (H,).
    z: Output gate tensor of shape (B, L, H, P).
    mimo_x: MIMO rank expansion parameter for x of shape (H, R, P).
    mimo_z: MIMO rank expansion parameter for z of shape (H, R, P).
    mimo_o: MIMO rank contraction parameter of shape (H, R, P).
    is_mimo: Whether to use Multi-Input Multi-Output recurrence.

  Returns:
    Output tensor y of shape (B, L, H, P).
  """
  batch_size, seq_len, num_heads, head_dim = x.shape
  d_state = b_proj.shape[-1]
  dtype = x.dtype

  # Exponentiate log decay (A * dt) to obtain the multiplicative decay factor:
  # decay_t = exp(A * dt_t) in (0, 1).
  # Shape: (B, L, H) -> (B, L, H, 1, 1) for broadcasting with state h (B, H, P, D).
  decay = torch.exp(adt).unsqueeze(-1).unsqueeze(-1)
  dt_expanded = dt.unsqueeze(-1).unsqueeze(-1)
  trap_expanded = trap.unsqueeze(-1).unsqueeze(-1)

  # ---------------------------------------------------------------------------
  # Phase 1: Precompute B * x outer product across the entire sequence.
  # ---------------------------------------------------------------------------
  # Instead of computing B_t * x_t inside the serial recurrence loop at each
  # time step t (which yields L small, memory-bandwidth-bound matrix multiplies),
  # we precompute bx_curr across all L tokens in parallel upfront.
  # This turns L sequential operations into a single batched, high-throughput
  # GEMM contraction that fully saturates TPU Matrix Multiply Units (MXUs) and
  # GPU Tensor Cores.
  # ---------------------------------------------------------------------------
  x_r = None
  if is_mimo and mimo_x is not None:
    # MIMO mode: Project x into rank-R subspace:
    # (B, L, H, P) x (H, R, P) -> (B, L, R, H, P)
    x_r = torch.einsum("blhp,hrp->blrhp", x.float(), mimo_x.float())
    # Contract rank-R inputs with rank-R B projections:
    # (B, L, R, H, P) x (B, L, R, H, D) -> (B, L, H, P, D)
    bx_curr = torch.einsum("blrhp,blrhd->blhpd", x_r, b_proj.float())
  else:
    # SISO mode: Use primary rank-0 slice of b_proj: (B, L, H, D)
    b_s = b_proj[:, :, 0].float()
    # Outer product between input x and projection B:
    # (B, L, H, P) x (B, L, H, D) -> (B, L, H, P, D)
    bx_curr = torch.einsum("blhp,blhd->blhpd", x.float(), b_s)

  # ---------------------------------------------------------------------------
  # Phase 2: Exponential-trapezoidal discretization blending.
  # ---------------------------------------------------------------------------
  # Shifts bx_curr by 1 step along sequence dimension (dim=1) to obtain (Bx)_{t-1},
  # with zero-padding at the initial time step t=0 (boundary condition (Bx)_{-1} = 0).
  # The learned gate `trap` in (0, 1) blends standard ZOH with trapezoidal integration:
  #   bx_blended = (1 - trap) * (Bx)_t + trap * 0.5 * ((Bx)_t + (Bx)_{t-1})
  # ---------------------------------------------------------------------------
  bx_prev = F.pad(bx_curr[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))
  bx_blended = (1.0 - trap_expanded) * bx_curr + trap_expanded * 0.5 * (
      bx_curr + bx_prev
  )
  u_ssm = dt_expanded * bx_blended  # (B, L, H, P, D)

  # ---------------------------------------------------------------------------
  # Phase 3: Sequential SSM recurrence across the sequence dimension.
  # ---------------------------------------------------------------------------
  # Recurrent state update:
  #   h_t = decay_t * h_{t-1} + u_t
  #
  # Each head maintains a (head_dim, d_state) matrix state representing the
  # accumulated key-value associative memory.
  # All state updates are computed in float32 for numerical stability.
  # ---------------------------------------------------------------------------
  h = torch.zeros(
      batch_size,
      num_heads,
      head_dim,
      d_state,
      device=x.device,
      dtype=torch.float32,
  )
  hs = []
  for t in range(seq_len):
    h = decay[:, t] * h + u_ssm[:, t]
    hs.append(h)
  # Stack unrolled states along sequence dimension: (B, L, H, P, D)
  h_all = torch.stack(hs, dim=1)

  # ---------------------------------------------------------------------------
  # Phase 4: Output projection, skip connection, and SiLU gating.
  # ---------------------------------------------------------------------------
  # Computes y_t = (C_t * h_t + D * x_t) * silu(z_t) across all L steps in
  # parallel using a batched contraction.
  # ---------------------------------------------------------------------------
  if (
      is_mimo
      and mimo_z is not None
      and mimo_o is not None
      and mimo_x is not None
      and z is not None
      and x_r is not None
  ):
    # MIMO Output Contraction:
    # 1. Project state h_all back to head_dim across MIMO rank R:
    #    (B, L, R, H, D) x (B, L, H, P, D) -> (B, L, R, H, P)
    y_r = torch.einsum("blrhd,blhpd->blrhp", c_proj.float(), h_all)
    # 2. Add direct skip connection per rank: D * x_r
    skip = d_skip.view(1, 1, 1, num_heads, 1) * x_r
    # 3. Project gate z into rank-R subspace: (B, L, H, P) -> (B, L, R, H, P)
    z_r = torch.einsum("blhp,hrp->blrhp", z.float(), mimo_z.float())
    # 4. Apply SiLU gating: (y_r + skip) * silu(z_r)
    y_r = (y_r + skip) * F.silu(z_r)
    # 5. Contract rank-R subspace back to head dimension:
    #    (B, L, R, H, P) x (H, R, P) -> (B, L, H, P)
    y = torch.einsum("blrhp,hrp->blhp", y_r, mimo_o.float())
  else:
    # SISO Output Contraction:
    # 1. Contract state h_all with C projection over state dim D:
    #    (B, L, H, D) x (B, L, H, P, D) -> (B, L, H, P)
    c_s = c_proj[:, :, 0].float()
    y = torch.einsum("blhd,blhpd->blhp", c_s, h_all)
    # 2. Add direct skip connection: D * x
    y = y + d_skip.view(1, 1, num_heads, 1) * x.float()
    # 3. Apply SiLU gating if gate z is provided: y * silu(z)
    if z is not None:
      y = y * F.silu(z.float())

  # Cast output back to original input precision (e.g. bfloat16).
  return y.to(dtype)


class Mamba3(nn.Module):
  """Mamba-3 sequence mixer layer.

  Implements exponential-trapezoidal discretization, complex-valued / rotary
  state space (RoPE), and Multi-Input Multi-Output (MIMO) SSM recurrence.

  Attributes:
    d_model: Input and output embedding dimension.
    d_state: Latent state space dimension per head (D).
    expand: Hidden dimension expansion factor (d_inner = expand * d_model).
    headdim: Dimension per head (P).
    nheads: Number of attention/SSM heads (H = d_inner // headdim).
    ngroups: Number of key/value (B/C) head groups (grouped-query SSM).
    num_bc_heads: Number of head groups for B and C projections (= ngroups).
    rope_fraction: Fraction of state space coordinates rotated via RoPE.
    split_tensor_size: Number of coordinates rotated via RoPE.
    num_rope_angles: Number of unique 2D rotation angles (= split_tensor_size //
      2).
    is_mimo: Boolean flag indicating whether MIMO recurrence is active.
    mimo_rank: Rank R for MIMO projection expansions and contractions.
  """

  def __init__(
      self,
      d_model: int,
      d_state: int = 128,
      expand: int = 2,
      headdim: int = 64,
      ngroups: int = 1,
      rope_fraction: float = 0.5,
      dt_min: float = 0.001,
      dt_max: float = 0.1,
      dt_init_floor: float = 1e-4,
      a_floor: float = 1e-4,
      is_mimo: bool = False,
      mimo_rank: int = 4,
      device: torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    """Initializes the Mamba-3 layer.

    Args:
      d_model: Input and output dimension of token embeddings.
      d_state: SSM latent state dimension per head (D).
      expand: Expansion ratio from d_model to d_inner.
      headdim: Dimension of each attention/SSM head (P).
      ngroups: Number of B and C projection head groups (grouped-query SSM).
      rope_fraction: Fraction of d_state coordinates rotated via RoPE (0.5 or
        1.0).
      dt_min: Minimum initial step size dt.
      dt_max: Maximum initial step size dt.
      dt_init_floor: Minimum floor for step size initialization.
      a_floor: Minimum magnitude floor for the negative continuous decay rate A.
      is_mimo: Whether to enable Multi-Input Multi-Output recurrence.
      mimo_rank: Rank dimension R for MIMO state projections.
      device: Optional torch device.
      dtype: Optional torch dtype for weights.
    """
    factory_kwargs = {"device": device, "dtype": dtype}
    super().__init__()
    self.d_model = d_model
    self.d_state = d_state
    self.expand = expand
    self.headdim = headdim
    self.a_floor = a_floor
    self.is_mimo = is_mimo
    self.mimo_rank = mimo_rank if is_mimo else 1
    self.num_bc_heads = ngroups

    # Compute inner dimension and verify divisibility by head dimension.
    self.d_inner = int(expand * d_model)
    if self.d_inner % headdim != 0:
      raise ValueError("d_inner must be divisible by headdim")
    self.nheads = self.d_inner // headdim

    # Ensure head count is divisible by the number of B/C groups (grouped-query SSM).
    if self.nheads % self.num_bc_heads != 0:
      raise ValueError("nheads must be divisible by ngroups")

    # Validate RoPE fraction and calculate coordinate split sizes.
    if rope_fraction not in (0.5, 1.0):
      raise ValueError("rope_fraction must be 0.5 or 1.0")
    self.split_tensor_size = int(d_state * rope_fraction)
    # Ensure split size is even so coordinates can be paired for 2D rotation.
    if self.split_tensor_size % 2 != 0:
      self.split_tensor_size -= 1
    self.num_rope_angles = self.split_tensor_size // 2

    # -------------------------------------------------------------------------
    # Input projection sizing:
    # A single fused linear projection maps d_model to all parallel branches:
    # - z (gate branch): d_inner
    # - x (SSM input branch): d_inner
    # - B projection (Key): d_state * ngroups * mimo_rank
    # - C projection (Query): d_state * ngroups * mimo_rank
    # - dt delta: nheads
    # - A parameter delta: nheads
    # - trapezoidal blending gate: nheads
    # - RoPE angular frequencies: num_rope_angles
    # -------------------------------------------------------------------------
    d_in_proj = (
        2 * self.d_inner
        + 2 * d_state * ngroups * self.mimo_rank
        + 3 * self.nheads
        + self.num_rope_angles
    )
    self.in_proj = nn.Linear(d_model, d_in_proj, bias=False, **factory_kwargs)

    # -------------------------------------------------------------------------
    # Time step dt bias initialization (Inverse Softplus):
    # During forward pass, dt = softplus(dd_dt + dt_bias).
    # To initialize dt to dt_init ~ Exp(Uniform(log(dt_min), log(dt_max))),
    # we solve softplus(dt_bias) = dt_init:
    #   dt_init = log(1 + exp(dt_bias))
    #   exp(dt_bias) = exp(dt_init) - 1
    #   dt_bias = log(exp(dt_init) - 1)
    #           = dt_init + log(1 - exp(-dt_init))
    #           = dt_init + log(-expm1(-dt_init))
    # This stable form prevents floating-point overflow for large dt_init.
    # -------------------------------------------------------------------------
    dt_init = torch.exp(
        torch.rand(self.nheads, dtype=torch.float32)
        * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    ).clamp(min=dt_init_floor)
    dt_bias_init = dt_init + torch.log(-torch.expm1(-dt_init))
    self.dt_bias = nn.Parameter(dt_bias_init)

    # State projection biases for B and C, shaped for automatic trailing broadcast:
    # (mimo_rank, nheads, d_state) broadcasts cleanly to:
    # (batch_size, seq_len, mimo_rank, nheads, d_state) without transposing.
    self.b_bias = nn.Parameter(
        torch.ones(self.mimo_rank, self.nheads, d_state, dtype=torch.float32)
    )
    self.c_bias = nn.Parameter(
        torch.ones(self.mimo_rank, self.nheads, d_state, dtype=torch.float32)
    )

    # RMSNorm applied to B and C prior to head expansion to stabilize state norms.
    self.b_norm = RMSNorm(d_state)
    self.c_norm = RMSNorm(d_state)

    # MIMO projection parameters (rank expansion and contraction).
    # Scaled by 1.0 / mimo_rank to preserve activation variance across ranks.
    if self.is_mimo:
      self.mimo_x = nn.Parameter(
          torch.ones(
              self.nheads, self.mimo_rank, self.headdim, **factory_kwargs
          )
          / self.mimo_rank
      )
      self.mimo_z = nn.Parameter(
          torch.ones(
              self.nheads, self.mimo_rank, self.headdim, **factory_kwargs
          )
      )
      self.mimo_o = nn.Parameter(
          torch.ones(
              self.nheads, self.mimo_rank, self.headdim, **factory_kwargs
          )
          / self.mimo_rank
      )
    else:
      self.mimo_x = None
      self.mimo_z = None
      self.mimo_o = None

    # D skip connection weight per head (continuous feedthrough matrix D).
    self.d = nn.Parameter(torch.ones(self.nheads, **factory_kwargs))

    # Final linear projection from inner dimension back to d_model.
    self.out_proj = nn.Linear(
        self.d_inner, d_model, bias=False, **factory_kwargs
    )

  def forward(self, u: torch.Tensor) -> torch.Tensor:
    """Runs forward pass of Mamba-3 sequence mixer.

    Args:
      u: Input tensor of shape (batch, seq_len, d_model).

    Returns:
      Output tensor of shape (batch, seq_len, d_model).
    """
    # -------------------------------------------------------------------------
    # Step 1: Linear input projection and feature chunk splitting.
    # Executes a single fused GEMM projecting input token embeddings into all
    # parallel branches: gate z, SSM input x, B, C, dt, A, trap, and RoPE angles.
    # -------------------------------------------------------------------------
    batch_size, seq_len, _ = u.shape
    projected = self.in_proj(u)  # (B, L, d_in_proj)
    z, x, b_raw, c_raw, dd_dt, dd_a, trap_raw, angle_raw = torch.split(
        projected,
        [
            self.d_inner,
            self.d_inner,
            self.d_state * self.num_bc_heads * self.mimo_rank,
            self.d_state * self.num_bc_heads * self.mimo_rank,
            self.nheads,
            self.nheads,
            self.nheads,
            self.num_rope_angles,
        ],
        dim=-1,
    )

    # Reshape input and gate tensors to multi-head layout: (B, L, H, P).
    z = z.view(batch_size, seq_len, self.nheads, self.headdim)
    x = x.view(batch_size, seq_len, self.nheads, self.headdim)

    # Reshape raw B and C projections to (B, L, R, G, D).
    b_raw = b_raw.view(
        batch_size, seq_len, self.mimo_rank, self.num_bc_heads, self.d_state
    )
    c_raw = c_raw.view(
        batch_size, seq_len, self.mimo_rank, self.num_bc_heads, self.d_state
    )

    # -------------------------------------------------------------------------
    # Step 2: Parameter activations (decay rate A, step size dt, trap gate).
    # - Decay rate A is strictly negative: -softplus(dd_a) clamped to -a_floor.
    # - Time step dt is strictly positive: softplus(dd_dt + dt_bias).
    # - Log decay factor adt = A * dt.
    # - Trapezoidal blending gate trap is squashed to (0, 1) via sigmoid.
    # -------------------------------------------------------------------------
    a = -F.softplus(dd_a.float()).clamp(max=-self.a_floor)
    dt = F.softplus(dd_dt.float() + self.dt_bias)
    adt = a * dt
    trap = torch.sigmoid(trap_raw.float())

    # -------------------------------------------------------------------------
    # Step 3: Normalize B and C projections and broadcast across head groups.
    # Applies RMSNorm per group to stabilize state magnitudes, then broadcasts
    # group heads G to total heads H.
    # -------------------------------------------------------------------------
    b_normed = self.b_norm(b_raw.float())
    c_normed = self.c_norm(c_raw.float())

    if self.num_bc_heads == 1:
      # Zero-copy memory broadcast for single-group (G=1).
      b_exp = b_normed.expand(
          batch_size, seq_len, self.mimo_rank, self.nheads, self.d_state
      )
      c_exp = c_normed.expand(
          batch_size, seq_len, self.mimo_rank, self.nheads, self.d_state
      )
    else:
      # Interleaved repetition for grouped-query heads (G > 1).
      repeat_factor = self.nheads // self.num_bc_heads
      b_exp = b_normed.repeat_interleave(repeat_factor, dim=3)
      c_exp = c_normed.repeat_interleave(repeat_factor, dim=3)

    # Add learned state projection biases: (mimo_rank, nheads, d_state).
    b_exp = b_exp + self.b_bias
    c_exp = c_exp + self.c_bias

    # -------------------------------------------------------------------------
    # Step 4: Apply continuous-time Rotary Position Embedding (RoPE) to state.
    # In continuous SSMs, time step dt varies per token. Angular increments
    # dt * omega are accumulated along sequence dimension using parallel prefix
    # sum (cumsum): theta_t = sum_{s=0}^t dt_s * omega.
    # -------------------------------------------------------------------------
    # Compute angular increments: (B, L, H, num_rope_angles)
    angle_increments = angle_raw.float().unsqueeze(2) * dt.float().unsqueeze(-1)
    # Prefix-sum accumulation along the sequence dimension (dim=1):
    cumulative_angles = torch.cumsum(angle_increments, dim=1)
    # Expand angles across MIMO rank dimension: (B, L, R, H, num_rope_angles)
    angles_for_rot = cumulative_angles.unsqueeze(2).expand(
        batch_size, seq_len, self.mimo_rank, self.nheads, self.num_rope_angles
    )

    # Rotate the first `split_tensor_size` coordinates of B and C.
    b_rot = apply_rope(b_exp[..., : self.split_tensor_size], angles_for_rot)
    c_rot = apply_rope(c_exp[..., : self.split_tensor_size], angles_for_rot)

    # Concatenate rotated coordinates with remaining unrotated coordinates.
    b_proj = torch.cat([b_rot, b_exp[..., self.split_tensor_size :]], dim=-1)
    c_proj = torch.cat([c_rot, c_exp[..., self.split_tensor_size :]], dim=-1)

    # -------------------------------------------------------------------------
    # Step 5: Core SSM scan recurrence.
    # Executes the exponential-trapezoidal discretization recurrence loop,
    # applies skip connection D * x, and gates with silu(z).
    # -------------------------------------------------------------------------
    y = mamba3_scan(
        x=x,
        b_proj=b_proj,
        c_proj=c_proj,
        adt=adt,
        dt=dt,
        trap=trap,
        d_skip=self.d,
        z=z,
        mimo_x=self.mimo_x,
        mimo_z=self.mimo_z,
        mimo_o=self.mimo_o,
        is_mimo=self.is_mimo,
    )

    # -------------------------------------------------------------------------
    # Step 6: Output linear projection.
    # Flatten multi-head layout (B, L, H, P) back to (B, L, d_inner) and
    # project back to token dimension (B, L, d_model).
    # -------------------------------------------------------------------------
    y = y.reshape(batch_size, seq_len, self.d_inner)
    return self.out_proj(y.to(u.dtype))
