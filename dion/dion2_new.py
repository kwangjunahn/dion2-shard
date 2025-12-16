"""
Dion2 Optimizer - Fully Optimized Implementation

Key differences from Muon:
- Selects top-α fraction of rows (by L2 norm) for orthogonalization
- Only communicates and orthogonalizes the selected submatrix
- Applies error-feedback decay to selected rows after extraction

Communication pattern (same as Muon):
- DDP: all-gather (each rank orthogonalizes one matrix, then gathers results)
- FSDP: all-to-all (shards → full matrix on owner → orthogonalize → shards)

Row selection is done locally on each shard, so:
- DDP: selection on full matrix
- FSDP: selection on each shard independently (slightly different algorithm, similar performance)

Optimizations:
- torch.compile on hot paths for kernel fusion and reduced Python overhead
- foreach operations for batched tensor updates
- Stacked tensor operations for row selection (all matrices in batch have same shape)
"""

import math
import torch
import torch.distributed as dist
from itertools import chain
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Callable, Generator, List, Optional, Tuple, Union

from .newton_schulz_triton import newton_schulz_triton, zeropower_via_newtonschulz5
from .opt_utils import (
    AsyncRuntime,
    AsyncTask,
    create_param_batches,
    pad_batch,
    to_local,
)
from .scalar_opts import adamw_update_foreach_async, lion_update_foreach_async


class Dion2(Optimizer):
    """
    Distributed Dion2 optimizer for PyTorch FSDP2. Also compatible with DDP.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
            Use DeviceMesh for FSDP2 and ProcessGroup for DistributedDataParallel.
        lr: Base learning rate. Scaled based on matrix dimensions.
        fraction: Fraction of rows to orthogonalize per update (0 < fraction <= 1).
        ef_decay: Error-feedback decay factor applied to selected rows.
        betas: Tuple of (beta1, beta2) for AdamW and Lion algorithms.
        weight_decay: Weight decay factor.
        epsilon: Small value to avoid division by zero.
        adjust_lr: How to adjust learning rate ("spectral_norm", "rms_norm", or None).
        flatten: Whether to flatten 3D+ tensors to 2D.
        use_triton: Whether to use Triton kernel for Newton-Schulz.
        newton_schulz_func: Custom Newton-Schulz function.
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        fraction: float = 0.25,
        ef_decay: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        adjust_lr: Optional[str] = "spectral_norm",
        flatten: bool = False,
        use_triton: bool = False,
        newton_schulz_func: Optional[Callable] = None,
    ):
        # Validate hyperparameters
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if ef_decay < 0.0:
            raise ValueError(f"Invalid ef_decay: {ef_decay}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if adjust_lr not in ("spectral_norm", "rms_norm", None):
            raise ValueError(f"Invalid adjust_lr: {adjust_lr}")

        defaults = dict(
            lr=lr,
            ef_decay=ef_decay,
            fraction=fraction,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            epsilon=epsilon,
            flatten=flatten,
            adjust_lr=adjust_lr,
            algorithm="dion2",
            step=0,
        )
        super().__init__(params, defaults)

        # Distributed configuration
        if isinstance(distributed_mesh, DeviceMesh):
            if distributed_mesh.ndim != 1:
                raise ValueError(
                    f"Only 1D DeviceMesh supported, got {distributed_mesh.ndim}D."
                )
            self._device_rank = distributed_mesh.get_local_rank()
            self._world_size = distributed_mesh.size()
            self._process_group = distributed_mesh.get_group()
        elif isinstance(distributed_mesh, ProcessGroup):
            self._device_rank = dist.get_rank(distributed_mesh)
            self._world_size = dist.get_world_size(distributed_mesh)
            self._process_group = distributed_mesh
        elif distributed_mesh is None:
            self._device_rank = 0
            self._world_size = 1
            self._process_group = None
        else:
            raise TypeError(f"Invalid distributed_mesh type: {type(distributed_mesh)}")
        self._distributed_mesh = distributed_mesh

        # Newton-Schulz configuration
        if newton_schulz_func is not None:
            if not callable(newton_schulz_func):
                raise TypeError(f"newton_schulz_func must be callable")
            self._newton_schulz_func = newton_schulz_func
        elif use_triton:
            self._newton_schulz_func = newton_schulz_triton
        else:
            self._newton_schulz_func = zeropower_via_newtonschulz5

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        dion2_groups = []
        lion_groups = []
        adamw_groups = []

        for group in self.param_groups:
            group["step"] += 1
            algo = group["algorithm"]
            if algo == "dion2":
                dion2_groups.append(group)
            elif algo == "lion":
                lion_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        dion2_tasks = self._create_dion2_tasks(dion2_groups)
        lion_tasks = self._create_lion_tasks(lion_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = chain(dion2_tasks, lion_tasks, adamw_tasks)
        runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=3)
        runtime.run()

        return loss

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        """Initialize optimizer state (identical to Muon)."""
        state = self.state[param]
        if not state:
            state["momentum"] = torch.zeros_like(param)
            if algo == "adamw":
                state["variance"] = torch.zeros_like(param)
        return state

    def _create_dion2_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        """Create batched Dion2 update tasks."""
        for group in param_groups:
            assert group["algorithm"] == "dion2"
            assert all(p.ndim >= 2 for p in group["params"]), \
                "Dion2 only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            # Hyperparameters as tensors for torch.compile
            dion2_args = dict(
                lr=torch.tensor(group["lr"]),
                ef_decay=torch.tensor(group["ef_decay"]),
                fraction=group["fraction"],
                weight_decay=torch.tensor(group["weight_decay"]),
                epsilon=torch.tensor(group["epsilon"]),
                flatten=group["flatten"],
                adjust_lr=group["adjust_lr"],
                device_rank=self._device_rank,
                world_size=self._world_size,
                process_group=self._process_group,
                newton_schulz_func=self._newton_schulz_func,
            )

            # Batch parameters by world_size (same as Muon)
            for params in create_param_batches(group_params, batch_size=self._world_size):
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, "dion2") for p in params]
                momentums = [s["momentum"] for s in states]

                # Determine sharding configuration
                shard_dim = None
                is_batch_sharded = False

                if isinstance(params[0], DTensor):
                    if not isinstance(self._distributed_mesh, DeviceMesh):
                        raise RuntimeError(
                            "Must use DeviceMesh for DTensor parameters."
                        )

                    # Find sharded placements (skip size-1 mesh dims)
                    shard_placements = [
                        (i, p)
                        for i, p in enumerate(params[0].placements)
                        if p.is_shard() and params[0].device_mesh.size(i) > 1
                    ]

                    # Check for batch vs matrix dimension sharding
                    if not group["flatten"]:
                        matrix_dims = {params[0].ndim - 1, params[0].ndim - 2}
                        is_batch_sharded = any(
                            p.dim not in matrix_dims for _, p in shard_placements
                        )
                        shard_placements = [
                            (i, p) for i, p in shard_placements if p.dim in matrix_dims
                        ]

                    if len(shard_placements) == 1:
                        shard_dim = shard_placements[0][1].dim
                    elif len(shard_placements) > 1:
                        raise NotImplementedError(
                            "Multiple sharded dimensions not supported."
                        )

                    # Verify mesh alignment
                    if shard_placements:
                        mesh_dim = shard_placements[0][0]
                        if params[0].device_mesh.get_group(mesh_dim) != self._process_group:
                            raise RuntimeError("DTensor mesh doesn't match optimizer mesh.")

                # Handle batch-sharded 3D tensors (each device has different matrices)
                if is_batch_sharded:
                    for x, g, m in zip(params, gradients, momentums):
                        yield AsyncTask(
                            dion2_update_batch_async(
                                X=[x],
                                G=[g],
                                M=[m],
                                shard_dim=None,
                                **dion2_args,
                            )
                        )
                else:
                    yield AsyncTask(
                        dion2_update_batch_async(
                            X=pad_batch(params, self._world_size),
                            G=pad_batch(gradients, self._world_size),
                            M=pad_batch(momentums, self._world_size),
                            shard_dim=shard_dim,
                            **dion2_args,
                        )
                    )

    def _create_lion_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        """Create Lion update tasks."""
        for group in param_groups:
            assert group["algorithm"] == "lion"

            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue

            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, "lion") for p in params]
            momentums = [s["momentum"] for s in states]

            yield AsyncTask(
                lion_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    lr=torch.tensor(group["lr"]),
                    beta1=torch.tensor(group["beta1"]),
                    beta2=torch.tensor(group["beta2"]),
                    weight_decay=torch.tensor(group["weight_decay"]),
                )
            )

    def _create_adamw_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        """Create AdamW update tasks."""
        for group in param_groups:
            assert group["algorithm"] == "adamw"

            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue

            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, "adamw") for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]

            yield AsyncTask(
                adamw_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    V=to_local(variances),
                    lr=torch.tensor(group["lr"]),
                    beta1=torch.tensor(group["beta1"]),
                    beta2=torch.tensor(group["beta2"]),
                    weight_decay=torch.tensor(group["weight_decay"]),
                    step=torch.tensor(group["step"]),
                    epsilon=torch.tensor(group["epsilon"]),
                )
            )


# =============================================================================
# Core Dion2 Update Functions
# =============================================================================

def dion2_update_batch_async(
    X: List[Tensor],  # Parameters (DTensor or Tensor), padded to world_size
    G: List[Tensor],  # Gradients, padded to world_size
    M: List[Tensor],  # Momentum buffers (modified in place), padded to world_size
    lr: Tensor,
    ef_decay: Tensor,
    fraction: float,
    weight_decay: Tensor,
    epsilon: Tensor,
    flatten: bool,
    adjust_lr: Optional[str],
    device_rank: int,
    world_size: int,
    shard_dim: Optional[int] = None,
    process_group: Optional[ProcessGroup] = None,
    newton_schulz_func: Optional[Callable] = None,
) -> Generator[None, None, None]:
    """
    Batched Dion2 update with fractional row selection.
    
    Algorithm:
    1. Update momentum: M = M + G
    2. Select top-α rows by L2 norm, extract submatrix
    3. Apply ef_decay to selected rows in M
    4. Communicate and orthogonalize only the submatrix
    5. Apply weight update to corresponding rows
    
    Communication patterns:
    - FSDP (shard_dim is not None): 
        - Parameters are row-sharded across ranks
        - Each rank selects top-k rows from its local shard
        - All-to-all gathers selected rows to form full submatrix
        - Orthogonalize, then all-to-all scatter back
    - DDP (shard_dim is None, world_size > 1):
        - Each rank has full matrices (batch of different matrices)
        - Each rank orthogonalizes one matrix from the batch
        - All-gather to distribute results
    - Single GPU: direct computation
    """
    assert len(X) == len(G) == len(M)

    # Step 1: Update momentum and select top-α rows (operates on local shards)
    # All matrices in batch have identical shapes, enabling stacked operations
    U_selected, row_indices_list = dion2_pre_orthogonalize(
        G=to_local(G),
        M=to_local(M),
        fraction=fraction,
        ef_decay=ef_decay,
    )

    # Step 2: Communicate and orthogonalize selected submatrices
    # -------------------------------------------------------------------------
    # FSDP path: all-to-all
    # -------------------------------------------------------------------------
    if shard_dim is not None:
        assert len(X) == world_size
        assert process_group is not None
        assert isinstance(X[0], DTensor)
        
        recv_shards = [torch.empty_like(u) for u in U_selected]
        work = dist.all_to_all(recv_shards, U_selected, group=process_group, async_op=True)
        yield
        work.wait()

        # Concatenate along row dimension to form full selected submatrix
        full_submatrix = torch.cat(recv_shards, dim=-2)

        # Orthogonalize the full selected submatrix
        full_submatrix = dion2_newton_schulz(
            full_submatrix, newton_schulz_func, flatten=flatten, epsilon=epsilon
        )

        # Split back into shards
        send_shards = [
            t.contiguous()
            for t in torch.tensor_split(full_submatrix, world_size, dim=-2)
        ]

        # All-to-all: scatter orthogonalized shards back to original owners
        U_ortho = [torch.empty_like(u) for u in U_selected]
        work = dist.all_to_all(U_ortho, send_shards, group=process_group, async_op=True)
        yield
        work.wait()

    # -------------------------------------------------------------------------
    # DDP path: all-gather
    # -------------------------------------------------------------------------
    elif len(U_selected) > 1:
        assert len(U_selected) == world_size
        assert process_group is not None

        # This rank orthogonalizes the matrix at index device_rank
        my_submatrix = dion2_newton_schulz(
            U_selected[device_rank], newton_schulz_func, flatten=flatten, epsilon=epsilon
        )

        # All-gather: collect orthogonalized submatrices from all ranks
        U_ortho = [torch.empty_like(u) for u in U_selected]
        work = dist.all_gather(
            U_ortho, my_submatrix.contiguous(), group=process_group, async_op=True
        )
        yield
        work.wait()

    # -------------------------------------------------------------------------
    # Single GPU path
    # -------------------------------------------------------------------------
    else:
        assert len(U_selected) == 1
        U_ortho = [
            dion2_newton_schulz(
                U_selected[0], newton_schulz_func, flatten=flatten, epsilon=epsilon
            )
        ]

    # Step 3: Compute adjusted learning rate (based on full/global matrix shape)
    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = _adjust_lr_spectral_norm(lr, X[0].shape, flatten=flatten)
    elif adjust_lr == "rms_norm":
        adjusted_lr = _adjust_lr_rms_norm(lr, X[0].shape, flatten=flatten)
    else:
        raise ValueError(f"Unknown adjust_lr: {adjust_lr}")

    # Step 4: Apply weight update to selected rows only
    dion2_post_orthogonalize(
        X=to_local(X),
        U_ortho=U_ortho,
        row_indices=row_indices_list,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
    )


# =============================================================================
# Optimized Pre-Orthogonalize Function (Stacked Operations)
# =============================================================================
#
# KEY INSIGHT: All matrices in a batch have identical shapes!
# This enables stacked/batched tensor operations instead of loops.
#
# OPTIMIZATION 1: Stack into 3D tensor for batched ops
# ----------------------------------------------------
# Stack (N, rows, cols) enables:
#   - Single batched norm instead of N separate norms
#   - Single batched topk instead of N separate topk calls  
#   - Single batched gather instead of N separate index_selects
#
# Why faster:
#   - One kernel launch instead of N launches
#   - Better GPU parallelism
#   - Reduced Python loop overhead
#
# OPTIMIZATION 2: In-place ef_decay via loop (unavoidable)
# --------------------------------------------------------
# torch.stack creates a copy, so we must apply ef_decay to originals.
# However, the loop benefits from torch.compile fusion.
#
# OPTIMIZATION 3: foreach for gradient accumulation
# -------------------------------------------------
# Optimal for in-place batched additions.
# =============================================================================

@torch.compile(fullgraph=True)
def dion2_pre_orthogonalize(
    G: List[Tensor],
    M: List[Tensor],
    fraction: float,
    ef_decay: Tensor,
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Update momentum and select top-α rows for orthogonalization.
    
    All matrices in the batch have identical shapes, enabling stacked operations.
    
    For each matrix M (shape: rows x cols):
    1. M += G (accumulate gradient into momentum)
    2. Compute L2 norm of each row
    3. Select top-k rows where k = ceil(fraction * rows)
    4. Extract selected rows as submatrix (k x cols)
    5. Apply ef_decay to selected rows in M (in-place)
    
    Returns:
        U_selected: List of selected submatrices in bf16 for communication
        row_indices: List of selected row indices for each matrix
    """
    dtype = M[0].dtype
    num_rows = M[0].size(-2)
    num_cols = M[0].size(-1)
    k = max(1, int(math.ceil(fraction * num_rows)))
    
    # OPTIMIZATION 1: foreach for batched gradient accumulation
    # Single fused kernel for all M += G operations
    G_casted = [g.to(dtype=dtype) for g in G]
    torch._foreach_add_(M, G_casted)
    
    # OPTIMIZATION 2: Stack for batched norm and topk
    # Shape: (batch_size, num_rows, num_cols)
    M_stacked = torch.stack(M, dim=0)
    
    # Batched L2 norm: (batch_size, num_rows)
    row_norms = M_stacked.norm(dim=-1)
    
    # Batched topk: indices shape (batch_size, k)
    _, indices = torch.topk(row_norms, k, dim=-1, sorted=False)
    
    # OPTIMIZATION 3: Batched gather for row extraction
    # (batch_size, k, num_cols)
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, num_cols)
    selected_stacked = torch.gather(M_stacked, dim=-2, index=indices_expanded)
    
    # Apply ef_decay to selected rows in original M tensors
    # Must loop because M tensors are separate (stack created a copy)
    # torch.compile will still optimize this loop
    row_indices_list = list(indices.unbind(dim=0))
    for m, idx in zip(M, row_indices_list):
        m[idx, :] *= ef_decay
    
    # Convert to bf16 and unstack for communication
    U_selected = list(selected_stacked.to(dtype=torch.bfloat16).unbind(dim=0))
    
    return U_selected, row_indices_list


# =============================================================================
# Optimized Post-Orthogonalize Function
# =============================================================================
#
# OPTIMIZATION 1: foreach for weight decay
# ----------------------------------------
# Single fused kernel for all X *= (1 - lr * wd) operations.
#
# OPTIMIZATION 2: Batched dtype conversion
# ----------------------------------------
# Convert all U tensors upfront for better memory planning.
#
# OPTIMIZATION 3: Loop with index_add_ (torch.compile optimized)
# --------------------------------------------------------------
# While we can't use foreach for indexed updates, torch.compile
# will fuse operations within each iteration and optimize the loop.
#
# Note: Stacking X would require copy-back which negates benefits.
# The loop approach is cleaner and torch.compile handles it well.
# =============================================================================

@torch.compile(fullgraph=True)
def dion2_post_orthogonalize(
    X: List[Tensor],
    U_ortho: List[Tensor],
    row_indices: List[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
):
    """
    Apply weight decay (to all rows) and update selected rows only.
    
    Weight decay: X = X * (1 - base_lr * weight_decay)  [all rows]
    Update: X[selected_rows] -= adjusted_lr * U_ortho   [selected rows only]
    """
    # OPTIMIZATION 1: foreach for batched weight decay
    torch._foreach_mul_(X, 1 - base_lr * weight_decay)
    
    # OPTIMIZATION 2: Batch dtype conversion upfront
    dtype = X[0].dtype
    U_converted = [u.to(dtype=dtype) for u in U_ortho]
    
    # OPTIMIZATION 3: Precompute scaled updates
    # This allows torch.compile to potentially fuse with index_add_
    neg_lr = -adjusted_lr
    U_scaled = [neg_lr * u for u in U_converted]
    
    # Apply updates to selected rows
    # torch.compile optimizes this loop
    for x, u_scaled, indices in zip(X, U_scaled, row_indices):
        x.index_add_(dim=-2, index=indices, source=u_scaled)


# =============================================================================
# Newton-Schulz Wrapper (unchanged)
# =============================================================================

def dion2_newton_schulz(
    X: Tensor,
    newton_schulz_func: Callable,
    flatten: bool,
    epsilon: Tensor,
) -> Tensor:
    """Apply Newton-Schulz orthogonalization with optional flattening."""
    original_shape = X.shape
    if flatten and X.ndim >= 3:
        X = X.flatten(start_dim=1)
    elif X.ndim >= 4:
        X = X.flatten(end_dim=-3)

    return newton_schulz_func(X, epsilon=epsilon).reshape(original_shape)


# =============================================================================
# Learning Rate Adjustment Functions (unchanged)
# =============================================================================

def _adjust_lr_spectral_norm(lr: Tensor, param_shape: torch.Size, flatten: bool) -> Tensor:
    """Adjust LR based on spectral norm (for scale transfer)."""
    if flatten:
        fan_out = param_shape[0]
        fan_in = math.prod(param_shape[1:])
    else:
        fan_out, fan_in = param_shape[-2:]
    return lr * math.sqrt(fan_out / fan_in)


def _adjust_lr_rms_norm(lr: Tensor, param_shape: torch.Size, flatten: bool) -> Tensor:
    """Adjust LR based on RMS norm (for Adam/AdamW compatibility)."""
    if flatten:
        fan_out = param_shape[0]
        fan_in = math.prod(param_shape[1:])
    else:
        fan_out, fan_in = param_shape[-2:]
    return lr * 0.2 * math.sqrt(max(fan_out, fan_in))