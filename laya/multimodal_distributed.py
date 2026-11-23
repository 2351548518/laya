"""Local synchronous data parallelism for conditional native media training.

Ranks can execute different numbers of native micro-batches, with different audio/vision
projectors active. Explicit optimizer-window reduction avoids forward-count collectives and
averages every real decision exactly once, including an uneven final window.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def main(self):
        return self.rank == 0

    @property
    def device(self):
        if dist.is_initialized() and dist.get_backend() == "gloo":
            return torch.device("cpu")
        return torch.device("cuda", self.local_rank) if torch.cuda.is_available() else torch.device("cpu")

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def broadcast(self, value):
        if self.world_size == 1:
            return value
        values = [value if self.main else None]
        dist.broadcast_object_list(values, src=0, device=self.device)
        return values[0]

    def synchronize_parameters(self, parameters):
        if self.world_size > 1:
            with torch.no_grad():
                for parameter in parameters:
                    dist.broadcast(parameter, src=0)

    def sum_values(self, values, device=None):
        result = torch.tensor(values, dtype=torch.float64, device=device or self.device)
        if self.world_size > 1:
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result.tolist()

    def sum_gradients(self, parameters):
        """Sum already globally normalized gradients, preserving globally unused grad=None."""
        if self.world_size == 1:
            return
        present = torch.tensor([p.grad is not None for p in parameters], dtype=torch.int32, device=self.device)
        dist.all_reduce(present, op=dist.ReduceOp.MAX)
        active = [p for p, flag in zip(parameters, present.tolist()) if flag]
        if not active:
            return
        # Bound temporary FP32 storage even for full training. The default LoRA/head/
        # projector experiment fits in one bucket; large matrices are reduced in slices.
        limit, bucket, count = 8 * 1024 * 1024, [], 0
        def reduce_bucket(values):
            if not values:
                return
            flattened = torch.cat([p.grad.detach().float().reshape(-1) if p.grad is not None
                else torch.zeros(p.numel(), dtype=torch.float32, device=p.device) for p in values])
            dist.all_reduce(flattened, op=dist.ReduceOp.SUM)
            offset = 0
            for parameter in values:
                gradient = flattened[offset:offset + parameter.numel()].view_as(parameter).to(parameter.dtype)
                if parameter.grad is None:
                    parameter.grad = gradient.clone()
                else:
                    parameter.grad.copy_(gradient)
                offset += parameter.numel()
        for parameter in active:
            if parameter.numel() > limit:
                reduce_bucket(bucket)
                bucket, count = [], 0
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                gradient = parameter.grad.reshape(-1)
                for start in range(0, gradient.numel(), limit):
                    part = gradient[start:start + limit].float()
                    dist.all_reduce(part, op=dist.ReduceOp.SUM)
                    gradient[start:start + limit].copy_(part)
                parameter.grad.copy_(gradient.view_as(parameter))
            else:
                if count + parameter.numel() > limit:
                    reduce_bucket(bucket)
                    bucket, count = [], 0
                bucket.append(parameter)
                count += parameter.numel()
        reduce_bucket(bucket)

    def save_training_state(self, directory, optimizer):
        """Save trusted local optimizer tensors and per-rank RNG state without model pickle."""
        directory = Path(directory)
        self.barrier()
        if self.main:
            _atomic_torch_save(optimizer.state_dict(), directory / "optimizer.pt")
        numpy_state = np.random.get_state()
        rng = {"cpu": torch.get_rng_state(), "python": random.getstate(),
               "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]]}
        device = optimizer.param_groups[0]["params"][0].device
        if device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state(device)
        _atomic_torch_save(rng, directory / ("rng_rank%d.pt" % self.rank))
        self.barrier()

    def restore_training_state(self, directory, optimizer):
        directory = Path(directory)
        path = directory / "optimizer.pt"
        if not path.is_file():
            return False
        rng_path = directory / ("rng_rank%d.pt" % self.rank)
        if not rng_path.is_file():
            raise ValueError("optimizer checkpoint is missing this rank's RNG state: " + str(rng_path))
        rng = torch.load(rng_path, map_location="cpu", weights_only=True)
        device = optimizer.param_groups[0]["params"][0].device
        required = {"cpu", "python", "numpy"} | ({"cuda"} if device.type == "cuda" else set())
        if not required.issubset(rng):
            raise ValueError("incomplete RNG state in " + str(rng_path))
        optimizer.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        torch.set_rng_state(rng["cpu"])
        random.setstate(rng["python"])
        state = rng["numpy"]
        np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
        if device.type == "cuda":
            torch.cuda.set_rng_state(rng["cuda"], device)
        return True


def _atomic_torch_save(value, path):
    temporary = path.with_name(path.name + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, path)


def distributed_context():
    """Reuse an existing group, or initialize torchrun's local NCCL/Gloo process group."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not dist.is_initialized() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo",
                                timeout=timedelta(hours=6))
    if dist.is_initialized():
        return DistributedContext(dist.get_rank(), dist.get_world_size(), local_rank)
    return DistributedContext()


def close_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()
