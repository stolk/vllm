# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import os

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

from .base_device_communicator import DeviceCommunicatorBase

logger = init_logger(__name__)

# Host-staged collectives for XPU TP on hosts where cross-device VRAM IPC is
# unavailable (GPUs on separate PCIe root ports: zeMemOpenIpcHandle fails with
# ZE_RESULT_ERROR_INVALID_ARGUMENT for cross-device handles). Payloads at or
# above VLLM_XPU_HOST_STAGED_MIN_BYTES are bounced through pinned host memory
# over the CPU (gloo) group; small payloads stay on the native device path.
_HOST_STAGED = os.environ.get("VLLM_XPU_HOST_STAGED_COLLECTIVES", "0") == "1"
_HOST_STAGED_MIN_BYTES = int(
    os.environ.get("VLLM_XPU_HOST_STAGED_MIN_BYTES", str(1 * 1024 * 1024))
)
# Move the staged all-reduce's data through /dev/shm instead of gloo. gloo's
# transport manages ~0.6 GB/s even between processes on one host, so a
# prefill-sized all-reduce (8192 x 5120 fp16 = 80 MiB) took ~140 ms, capping
# TP=2 prefill near 450 tok/s. Through shared memory the same all-reduce is
# ~32 ms (measured on 2x Arc Pro B70); gloo then only carries a 1-element
# barrier. Requires every rank of the group on one host; falls back otherwise.
_HOST_STAGED_SHM = os.environ.get("VLLM_XPU_HOST_STAGED_SHM", "0") == "1"


class _ShmAllReduce:
    """All-reduce through per-rank /dev/shm segments, summed on the device.

    Each rank copies its tensor into its own segment, all ranks meet at a
    barrier, then each rank copies every peer segment to its device and sums
    all contributions in rank order, so every rank gets identical bits.

    Segments are double-buffered: call n+2 reuses call n's slot, and no rank
    can pass call n+1's barrier until every rank has finished reading call n,
    so one barrier per call is enough.
    """

    def __init__(self, cpu_group: ProcessGroup, rank: int, world_size: int):
        import mmap
        import socket

        self._mmap = mmap
        self.cpu_group = cpu_group
        self.rank = rank
        self.world_size = world_size
        hosts: list = [None] * world_size
        dist.all_gather_object(hosts, socket.gethostname(), group=cpu_group)
        self.usable = len(set(hosts)) == 1
        tag = [f"{os.getpid()}_{id(self)}" if rank == 0 else None]
        dist.broadcast_object_list(
            tag, src=dist.get_global_rank(cpu_group, 0), group=cpu_group
        )
        self.tag = tag[0]
        self.capacity = 0
        self.generation = 0
        self.maps: list = []  # [slot][rank] -> mmap
        self.calls = 0

    def _allocate(self, nbytes: int) -> None:
        # Collective: every rank reaches this with the same nbytes, since
        # all-reduce operands have the same shape on every rank.
        for slot in self.maps:
            for m in slot:
                m.close()
        self.generation += 1
        cap = max(nbytes, 2 * self.capacity, 1 << 20)
        paths = [
            [f"/dev/shm/vllm_xpu_ar_{self.tag}_g{self.generation}_s{s}_r{r}"
             for r in range(self.world_size)]
            for s in range(2)
        ]
        for s in range(2):
            fd = os.open(paths[s][self.rank], os.O_CREAT | os.O_RDWR, 0o600)
            os.ftruncate(fd, cap)
            os.close(fd)
        dist.barrier(group=self.cpu_group)
        self.maps = []
        for s in range(2):
            row = []
            for r in range(self.world_size):
                fd = os.open(paths[s][r], os.O_RDWR)
                row.append(self._mmap.mmap(fd, cap))
                os.close(fd)
            self.maps.append(row)
        dist.barrier(group=self.cpu_group)
        # Everyone is attached: unlink now, so nothing leaks even on SIGKILL.
        for s in range(2):
            os.unlink(paths[s][self.rank])
        self.capacity = cap

    def all_reduce(self, output: torch.Tensor) -> torch.Tensor:
        nbytes = output.numel() * output.element_size()
        if nbytes > self.capacity:
            self._allocate(nbytes)
        slot = self.maps[self.calls & 1]
        self.calls += 1
        views = [
            torch.frombuffer(m, dtype=output.dtype, count=output.numel()).view(
                output.shape
            )
            for m in slot
        ]
        views[self.rank].copy_(output)
        dist.barrier(group=self.cpu_group)
        acc = None
        for r in range(self.world_size):
            part = output if r == self.rank else views[r].to(output.device)
            acc = part.clone() if acc is None else acc.add_(part)
        output.copy_(acc)
        return output


class XpuCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        use_all2all: bool = False,
    ):
        super().__init__(
            cpu_group, device, device_group, unique_name, use_all2all=use_all2all
        )
        self.ca_comm: None = None
        self._init_host_staging()
        if self.use_all2all:
            if self.all2all_backend in ("naive", "allgather_reducescatter"):
                from .all2all import AgRsAll2AllManager

                self.all2all_manager = AgRsAll2AllManager(self.cpu_group)
                logger.info("Using AgRs manager on XPU device.")

            else:  # type: ignore[has-type]
                logger.warning(
                    "`%s` all2all manager is not supported on XPU. "
                    "Falling back to AgRs manager for XPU, "
                    "which is the Default backend",
                    self.all2all_backend,  # type: ignore[has-type]
                )
                from .all2all import AgRsAll2AllManager

                self.all2all_manager = AgRsAll2AllManager(self.cpu_group)
                logger.info("Using AgRs manager on XPU device.")

    def _init_host_staging(self):
        self._staging_cache: dict = {}
        self._shm_ar: _ShmAllReduce | None = None
        if _HOST_STAGED:
            logger.info(
                "XpuCommunicator: host-staged collectives enabled "
                "(threshold=%d bytes, shm all-reduce=%s)",
                _HOST_STAGED_MIN_BYTES,
                _HOST_STAGED_SHM,
            )

    def _needs_staging(self, t: torch.Tensor) -> bool:
        return (
            _HOST_STAGED
            and t.numel() * t.element_size() >= _HOST_STAGED_MIN_BYTES
        )

    def _pinned(self, shape, dtype) -> torch.Tensor:
        key = (tuple(shape), dtype)
        buf = self._staging_cache.get(key)
        if buf is None:
            buf = torch.empty(shape, dtype=dtype, pin_memory=True)
            self._staging_cache[key] = buf
        return buf

    def _staged_all_reduce(self, output: torch.Tensor) -> torch.Tensor:
        if _HOST_STAGED_SHM and self.world_size > 1:
            # Created lazily on the first staged all-reduce, which every rank
            # of the group reaches together.
            if self._shm_ar is None:
                self._shm_ar = _ShmAllReduce(
                    self.cpu_group, self.rank_in_group, self.world_size
                )
                logger.info(
                    "XpuCommunicator: shm all-reduce %s",
                    "active" if self._shm_ar.usable
                    else "unavailable (ranks on several hosts), using gloo",
                )
            if self._shm_ar.usable:
                return self._shm_ar.all_reduce(output)
        buf = self._pinned(output.shape, output.dtype)
        buf.copy_(output, non_blocking=False)
        dist.all_reduce(buf, group=self.cpu_group)
        output.copy_(buf, non_blocking=False)
        return output

    def _staged_all_gather(self, output: torch.Tensor,
                           input_: torch.Tensor) -> None:
        inbuf = self._pinned(input_.shape, input_.dtype)
        inbuf.copy_(input_, non_blocking=False)
        outbuf = self._pinned((self.world_size,) + tuple(inbuf.shape),
                              input_.dtype)
        dist.all_gather(list(outbuf.unbind(0)), inbuf, group=self.cpu_group)
        output.copy_(outbuf.reshape(output.shape), non_blocking=False)

    def _staged_reduce_scatter(self, output: torch.Tensor,
                               input_: torch.Tensor) -> None:
        inbuf = self._pinned(input_.shape, input_.dtype)
        inbuf.copy_(input_, non_blocking=False)
        outbuf = self._pinned(output.shape, output.dtype)
        dist.reduce_scatter_tensor(outbuf, inbuf, group=self.cpu_group)
        output.copy_(outbuf, non_blocking=False)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        output = input_.clone()
        if self._needs_staging(output):
            return self._staged_all_reduce(output)
        dist.all_reduce(output, group=self.device_group)
        return output

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1):
        world_size = self.world_size

        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        output = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )

        if self._needs_staging(input_tensor):
            self._staged_reduce_scatter(output, input_tensor)
        else:
            dist.reduce_scatter_tensor(output, input_tensor,
                                       group=self.device_group)

        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ):
        world_size = self.world_size

        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        if sizes is not None:
            assert len(sizes) == world_size
            assert input_tensor.shape[0] == sum(sizes)
            chunk_size = sizes[self.rank_in_group]
        else:
            assert input_tensor.shape[0] % world_size == 0
            chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        output = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )
        if self._needs_staging(input_tensor):
            self._staged_reduce_scatter(output, input_tensor)
        elif sizes is not None and sizes.count(sizes[0]) != len(sizes):
            # if inputs shape in different ranks is not the same using reduce_scatter
            input_splits = list(input_tensor.split(sizes, dim=0))
            dist.reduce_scatter(output, input_splits, group=self.device_group)
        else:
            dist.reduce_scatter_tensor(output, input_tensor, group=self.device_group)
        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ):
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")
        world_size = self.world_size

        # 'sizes' is not needed if all inputs in the same group have the same
        # shape
        if sizes is not None and all(s == sizes[0] for s in sizes):
            sizes = None

        def _all_gather_single(input_: torch.Tensor, sizes: list[int] | None = None):
            input_size = input_.size()
            if sizes is not None:
                assert len(sizes) == world_size
                assert input_.shape[dim] == sizes[self.rank_in_group], (
                    f"{input_.shape[dim]} != {sizes[self.rank_in_group]}"
                )
                output_size = (sum(sizes),) + input_size[1:]
            else:
                output_size = (input_size[0] * world_size,) + input_size[1:]
            # Allocate output tensor.
            output_tensor = torch.empty(
                output_size, dtype=input_.dtype, device=input_.device
            )

            if self._needs_staging(input_):
                self._staged_all_gather(output_tensor, input_)
            elif sizes is not None:
                all_gather_list = []
                for size in sizes:
                    all_gather_list.append(
                        torch.empty(
                            (size,) + input_.shape[1:],
                            dtype=input_.dtype,
                            device=input_.device,
                        )
                    )
                dist.all_gather(all_gather_list, input_, group=self.device_group)
                output_tensor = torch.cat(all_gather_list, dim=0)
            else:
                dist.all_gather([output_tensor], input_, group=self.device_group)
            return output_tensor

        if isinstance(input_, torch.Tensor):
            return _all_gather_single(input_, sizes)

        output_list = []
        for inp in input_:
            output_list.append(_all_gather_single(inp, sizes=sizes))
        return output_list

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        # For xpu path, gather doesn't work properly together with ray
        # cluster so we use all_gather instead for now.
        input_size = input_.size()
        # Allocate output tensor.
        output_tensor = torch.empty(
            (self.world_size,) + input_size, dtype=input_.dtype, device=input_.device
        )
        # All-gather.
        if self._needs_staging(input_):
            self._staged_all_gather(output_tensor, input_)
        else:
            dist.all_gather_into_tensor(output_tensor, input_,
                                        group=self.device_group)
        if self.rank_in_group == dst:
            # Reshape
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(
                input_size[:dim]
                + (self.world_size * input_size[dim],)
                + input_size[dim + 1 :]
            )
        else:
            output_tensor = None
        return output_tensor

    def broadcast(self, input_: torch.Tensor, src: int = 0) -> None:
        if self._needs_staging(input_):
            buf = self._pinned(input_.shape, input_.dtype)
            if self.rank_in_group == src:
                buf.copy_(input_, non_blocking=False)
            dist.broadcast(buf, src=src, group=self.cpu_group)
            if self.rank_in_group != src:
                input_.copy_(buf, non_blocking=False)
        else:
            dist.broadcast(input_, src=src, group=self.device_group)

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Dispatch the hidden states and router logits to the appropriate device.
        This is a no-op in the base class.
        """

        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch_router_logits(
            hidden_states,
            router_logits,
            is_sequence_parallel,
            extra_tensors,
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Dispatch the hidden states and topk weights/ids to the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch(
            hidden_states,
            topk_weights,
            topk_ids,
            is_sequence_parallel,
            extra_tensors=extra_tensors,
        )

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        """
        Combine the hidden states and router logits from the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.combine(
            hidden_states,
            is_sequence_parallel,
        )

# --- Host-staged functional collectives -------------------------------------
# The MTP proposer's hidden-state all-gather is traced by inductor and reaches
# torch.distributed's python collectives directly, bypassing XpuCommunicator.
# Wrap the python entry points so large XPU payloads are staged through pinned
# host memory over the default group's CPU (gloo) backend. TP-only deployments
# have TP group == WORLD, which is what the staged path uses.
if _HOST_STAGED:
    import torch.distributed.distributed_c10d as _c10d

    _pg_world = None

    def _stage_guard(t):
        return (
            isinstance(t, torch.Tensor)
            and t.device.type == "xpu"
            and t.numel() * t.element_size() >= _HOST_STAGED_MIN_BYTES
        )

    _cpu_group_cache = None

    def _resolve_cpu_group(group):
        # The device (xccl) group cannot carry CPU tensors; use the matching
        # GroupCoordinator's gloo sibling instead.
        global _cpu_group_cache
        if _cpu_group_cache is not None:
            return _cpu_group_cache
        from vllm.distributed import get_tp_group, get_world_group
        for gc in (get_tp_group(), get_world_group()):
            dg = getattr(gc, "device_group", None)
            if group is None or dg is None or group is dg:
                _cpu_group_cache = gc.cpu_group
                return _cpu_group_cache
        _cpu_group_cache = get_world_group().cpu_group
        return _cpu_group_cache

    def _wrap_unary(name):
        orig = getattr(_c10d, name)
        def wrapped(tensor, *args, **kwargs):
            if _stage_guard(tensor):
                cpu = tensor.detach().to("cpu", copy=True)
                kwargs["group"] = _resolve_cpu_group(kwargs.pop("group", None))
                orig(cpu, *args, **kwargs)
                tensor.copy_(cpu, non_blocking=False)
                return None
            return orig(tensor, *args, **kwargs)
        setattr(_c10d, name, wrapped)
        setattr(dist, name, wrapped)

    def _wrap_all_gather_into_tensor():
        name = "all_gather_into_tensor"
        orig = getattr(_c10d, name)
        def wrapped(output, input_, *args, **kwargs):
            if _stage_guard(input_) or _stage_guard(output):
                cpu_in = input_.detach().to("cpu", copy=True)
                cpu_out = torch.empty(
                    (output.shape), dtype=output.dtype, pin_memory=True)
                kwargs["group"] = _resolve_cpu_group(kwargs.pop("group", None))
                orig(cpu_out, cpu_in, *args, **kwargs)
                output.copy_(cpu_out, non_blocking=False)
                return None
            return orig(output, input_, *args, **kwargs)
        setattr(_c10d, name, wrapped)
        setattr(dist, name, wrapped)

    def _wrap_all_gather():
        name = "all_gather"
        orig = getattr(_c10d, name)
        def wrapped(tensor_list, tensor, *args, **kwargs):
            if _stage_guard(tensor):
                cpu_in = tensor.detach().to("cpu", copy=True)
                cpu_list = [
                    torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
                    for t in tensor_list
                ]
                kwargs["group"] = _resolve_cpu_group(kwargs.pop("group", None))
                orig(cpu_list, cpu_in, *args, **kwargs)
                for t, c in zip(tensor_list, cpu_list):
                    t.copy_(c, non_blocking=False)
                return None
            return orig(tensor_list, tensor, *args, **kwargs)
        setattr(_c10d, name, wrapped)
        setattr(dist, name, wrapped)

    def _wrap_reduce_scatter_tensor():
        name = "reduce_scatter_tensor"
        orig = getattr(_c10d, name)
        def wrapped(output, input_, *args, **kwargs):
            if _stage_guard(input_) or _stage_guard(output):
                cpu_in = input_.detach().to("cpu", copy=True)
                cpu_out = torch.empty(
                    output.shape, dtype=output.dtype, pin_memory=True)
                kwargs["group"] = _resolve_cpu_group(kwargs.pop("group", None))
                orig(cpu_out, cpu_in, *args, **kwargs)
                output.copy_(cpu_out, non_blocking=False)
                return None
            return orig(output, input_, *args, **kwargs)
        setattr(_c10d, name, wrapped)
        setattr(dist, name, wrapped)

    _wrap_unary("all_reduce")
    _wrap_all_gather()
    _wrap_all_gather_into_tensor()
    _wrap_reduce_scatter_tensor()
    logger.info("host-staged functional collectives installed")
