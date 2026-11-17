"""DDP scaling: torchrun --nproc_per_node N ddp_bench.py --B 256. Fixed on-GPU batch per rank (no loader), full
fine-tuning with cftrain defaults (compile, packed text, gathered features + local loss). Rank 0 writes samples/s."""
import argparse, json, os, statistics, time
import torch, torch.distributed as dist
from cftrain.ref import load_reference
from cftrain.ref import text_ids
from cftrain.engine import Stepper, build_model, build_optimizer
from cftrain.model import pack_ids

ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=256); ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--bf16_grads", type=int, default=0); ap.add_argument("--out", required=True)
a = ap.parse_args()
dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size()
local = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local); dev = torch.device("cuda", local)
ref, _, _ = load_reference("cpu")
for m in ref.modules():
    if isinstance(m, torch.nn.Dropout): m.p = 0.0
model = build_model(ref, "full", text_dropout=0, attn_dropout=0).to(dev).train()
ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local], gradient_as_bucket_view=True, static_graph=True)
if a.bf16_grads:
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
    ddp.register_comm_hook(None, default_hooks.bf16_compress_hook)
st = Stepper(model, build_optimizer(model, 1e-5, 0.2), compile=True, world=world, rank=rank, ddp_model=ddp)
gen = torch.Generator().manual_seed(rank)
x = torch.randn(a.B, 3, 224, 224, device=dev)
pk = pack_ids(text_ids(a.B, "pmc", ref.text.config.vocab_size, False, gen))
inp = dict(images=x, packed=[t.to(dev) for t in pk[:5]] + [pk[5]])
for _ in range(5): st(inp)
torch.cuda.synchronize(); dist.barrier()
ts = []
for _ in range(3):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(a.steps): loss = st(inp)
    torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) / a.steps)
ms = statistics.median(ts) * 1e3
if rank == 0:
    r = dict(kind="ddp_bench", world=world, per_gpu_B=a.B, bf16_grads=a.bf16_grads, ms=round(ms, 2),
             samples_per_s=round(world * a.B / ms * 1e3, 1), per_gpu_samples_per_s=round(a.B / ms * 1e3, 1),
             peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), job=os.environ.get("CFTRAIN_RUN_ID"))
    print(json.dumps(r), flush=True); open(a.out, "a").write(json.dumps(r) + "\n")
dist.destroy_process_group()
