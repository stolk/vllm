# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the /dev/shm transport of the XPU host-staged all-reduce.

_ShmAllReduce only moves bytes through shared memory and sums on the tensor's
device, so every test here runs on CPU tensors over a gloo group: no GPU is
needed, and the tests can run while a server holds the cards.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed.device_communicators.xpu_communicator import _ShmAllReduce
from vllm.utils.network_utils import get_open_port


def _run(fn, world_size: int, *args) -> None:
    """Run fn(rank, world_size, *args) on world_size gloo ranks.

    A failure in any rank re-raises in the test with that rank's traceback.
    """
    port = get_open_port()
    mp.spawn(_entry, args=(fn, world_size, port, args), nprocs=world_size,
             join=True)


def _entry(rank, fn, world_size, port, args):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()


def _contributions(shape, dtype, world_size, seed):
    """Every rank's input, generated identically on every rank."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(shape, generator=g).to(dtype) for _ in range(world_size)]


def _rank_order_sum(parts):
    acc = parts[0].clone()
    for p in parts[1:]:
        acc.add_(p)
    return acc


def _assert_ranks_identical(t: torch.Tensor, world_size: int) -> None:
    """Every rank must hold the same bits, not merely close values."""
    gathered = [torch.empty_like(t) for _ in range(world_size)]
    dist.all_gather(gathered, t.contiguous())
    for r, other in enumerate(gathered):
        assert torch.equal(other.view(torch.uint8), t.view(torch.uint8)), (
            f"rank {dist.get_rank()} differs from rank {r}")


def _shm_segments(tag: str) -> list[str]:
    return [f for f in os.listdir("/dev/shm") if tag in f]


# --- correctness -------------------------------------------------------------


def _worker_matches_reference(rank, world_size, dtype):
    ar = _ShmAllReduce(dist.group.WORLD, rank, world_size)
    assert ar.usable
    for i, shape in enumerate([(1, 1), (7, 3), (256, 5120), (1000, 4097)]):
        parts = _contributions(shape, dtype, world_size, seed=100 + i)
        out = ar.all_reduce(parts[rank].clone())
        # Summed in rank order, so the result equals this reference exactly.
        assert torch.equal(out, _rank_order_sum(parts)), f"shape {shape}"
        _assert_ranks_identical(out, world_size)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("world_size", [2, 3])
def test_matches_rank_order_sum_exactly(world_size, dtype):
    _run(_worker_matches_reference, world_size, dtype)


def _worker_result_is_in_place(rank, world_size):
    ar = _ShmAllReduce(dist.group.WORLD, rank, world_size)
    parts = _contributions((64, 64), torch.float32, world_size, seed=7)
    x = parts[rank].clone()
    out = ar.all_reduce(x)
    assert out.data_ptr() == x.data_ptr()
    assert torch.equal(x, _rank_order_sum(parts))


def test_result_is_written_into_the_input_tensor():
    # XpuCommunicator.all_reduce hands in a clone and returns it; the transport
    # must fill that tensor rather than return a fresh one.
    _run(_worker_result_is_in_place, 2)


# --- buffer management -------------------------------------------------------


def _worker_growth_and_reuse(rank, world_size):
    ar = _ShmAllReduce(dist.group.WORLD, rank, world_size)
    shapes = ([(16, 16)] * 3            # fits the initial 1 MiB allocation
              + [(1024, 1024)]          # 4 MiB fp32: must grow
              + [(512, 512)] * 3        # smaller: must reuse, not reallocate
              + [(2048, 1024)])         # 8 MiB: grows again
    generations = []
    for i, shape in enumerate(shapes):
        parts = _contributions(shape, torch.float32, world_size, seed=200 + i)
        out = ar.all_reduce(parts[rank].clone())
        assert torch.equal(out, _rank_order_sum(parts)), f"call {i} {shape}"
        generations.append(ar.generation)
    assert ar.capacity >= 2048 * 1024 * 4
    # One allocation per growth step; same-size or smaller calls reuse it.
    assert generations == [1, 1, 1, 2, 2, 2, 2, 3], generations


def test_grows_when_needed_and_reuses_otherwise():
    _run(_worker_growth_and_reuse, 2)


def _worker_back_to_back(rank, world_size):
    # Many consecutive calls with different data exercise the double buffer:
    # a rank overwriting a slot its peer is still reading would corrupt a sum.
    ar = _ShmAllReduce(dist.group.WORLD, rank, world_size)
    for i in range(200):
        parts = _contributions((128, 257), torch.float16, world_size, seed=i)
        out = ar.all_reduce(parts[rank].clone())
        assert torch.equal(out, _rank_order_sum(parts)), f"call {i}"


@pytest.mark.parametrize("world_size", [2, 3])
def test_back_to_back_calls_do_not_race(world_size):
    _run(_worker_back_to_back, world_size)


def _worker_no_leak(rank, world_size):
    ar = _ShmAllReduce(dist.group.WORLD, rank, world_size)
    for shape in [(16, 16), (1024, 1024), (2048, 1024)]:
        ar.all_reduce(torch.ones(shape))
        dist.barrier()
        # Segments are unlinked once every rank has mapped them.
        segs = _shm_segments(ar.tag)
        # The next shape grows the buffer: a rank that moves on creates the
        # next generation's segments, which a slower rank would still list.
        dist.barrier()
        assert segs == [], segs


def test_segments_are_unlinked_after_attach():
    _run(_worker_no_leak, 2)


# --- applicability -----------------------------------------------------------


def _worker_hosts(rank, world_size, same_host):
    if not same_host:
        # Pretend every rank runs on its own host.
        socket.gethostname = lambda: f"host-{rank}"
    ar = _ShmAllReduce(dist.group.WORLD, rank, world_size)
    assert ar.usable is same_host


@pytest.mark.parametrize("same_host", [True, False])
def test_usable_only_when_all_ranks_share_a_host(same_host):
    _run(_worker_hosts, 2, same_host)


def _worker_tag(rank, world_size):
    ar = _ShmAllReduce(dist.group.WORLD, rank, world_size)
    tags = [None] * world_size
    dist.all_gather_object(tags, ar.tag)
    assert len(set(tags)) == 1, tags  # all ranks name the same segments


def test_all_ranks_agree_on_segment_names():
    _run(_worker_tag, 3)
