# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU TP=2 all-reduce through peer device memory (VLLM_XPU_PEER_ALLREDUCE).

Needs two XPU devices and the xpu_peer extension. Runs two worker processes
with vLLM's real TP setup (oneCCL device group + gloo CPU group) and the
host-staged collective settings used in production.
"""
import os
import traceback
import zlib

import pytest
import torch
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()
         and torch.xpu.device_count() >= 2),
    reason="needs two XPU devices",
)

ENV = {
    "VLLM_XPU_HOST_STAGED_COLLECTIVES": "1",
    "VLLM_XPU_HOST_STAGED_SHM": "1",
    "VLLM_XPU_PEER_ALLREDUCE": "1",
    "CCL_TOPO_P2P_ACCESS": "0",
    "CCL_ZE_IPC_EXCHANGE": "pidfd",
    "CCL_ATL_TRANSPORT": "ofi",
    "ZE_FLAT_DEVICE_HIERARCHY": "COMPOSITE",
}


def _data(seed, shape, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=g).half().to(device)


def _worker(rank, port, break_peer_on_rank, q):
    os.environ.update(ENV)
    try:
        torch.xpu.set_device(rank)
        dev = torch.device(f"xpu:{rank}")
        if break_peer_on_rank == rank:
            import xpu_peer.allreduce as impl

            def broken(*args, **kwargs):
                raise RuntimeError("zeMemOpenIpcHandle failed: 0x78000004")

            impl.PeerMapping = broken

        from tests.utils import init_test_distributed_environment
        from vllm.distributed import (get_tp_group,
                                      tensor_model_parallel_all_reduce)
        init_test_distributed_environment(2, 1, rank, str(port), local_rank=rank)

        res = {}
        for name, shape in [("prefill", (8192, 5120)), ("chunk", (300, 5120)),
                            ("small", (4, 5120))]:
            xs = [_data(zlib.crc32(name.encode()) % 1000 + r, shape, dev)
                  for r in (0, 1)]
            out = tensor_model_parallel_all_reduce(xs[rank].clone())
            res[name] = torch.equal(out, xs[0] + xs[1])
        # The operand itself is not modified.
        x = _data(9, (8192, 5120), dev)
        y = x.clone()
        tensor_model_parallel_all_reduce(y)
        res["operand_kept"] = torch.equal(x, y)

        comm = get_tp_group().device_communicator
        res["peer_active"] = comm._peer_ar is not None
        res["peer_failed"] = comm._peer_ar_failed
        res["shm_active"] = comm._shm_ar is not None
        q.put((rank, "ok", res))
    except Exception:
        q.put((rank, "error", traceback.format_exc()))


def _run(break_peer_on_rank=None):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29700 + os.getpid() % 200 + (0 if break_peer_on_rank is None else 50)
    procs = [ctx.Process(target=_worker, args=(r, port, break_peer_on_rank, q))
             for r in (0, 1)]
    for p in procs:
        p.start()
    out = {}
    for _ in procs:
        rank, status, val = q.get(timeout=600)
        assert status == "ok", f"rank {rank}:\n{val}"
        out[rank] = val
    for p in procs:
        p.join(timeout=60)
    return out


def test_peer_allreduce_used_and_exact():
    for rank, res in _run().items():
        assert res == {
            "prefill": True, "chunk": True, "small": True, "operand_kept": True,
            "peer_active": True, "peer_failed": False, "shm_active": False,
        }, (rank, res)


def test_peer_setup_failure_falls_back_to_shm_on_all_ranks():
    for rank, res in _run(break_peer_on_rank=1).items():
        assert res == {
            "prefill": True, "chunk": True, "small": True, "operand_kept": True,
            "peer_active": False, "peer_failed": True, "shm_active": True,
        }, (rank, res)
