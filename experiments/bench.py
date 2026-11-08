"""
CPU efficiency of the compared architectures: parameters, MACs, batch-1 latency at 224x224.
    python bench.py            # writes analysis/efficiency.csv
Models are timed in interleaved rounds so that CPU frequency drift affects all equally.
"""
import csv, os, platform, statistics, sys, time
import torch, torch.nn as nn, torchvision.models as tvm
from thop import profile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from model import BioLiteNet

NUM_CLASSES, ROUNDS, ITERS, WARMUP = 23, 5, 100, 30


def r50():
    m = tvm.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(2048, NUM_CLASSES))
    return m


MODELS = {
    "MobileNetV3-L": lambda: BioLiteNet(NUM_CLASSES, use_cbam=False, pretrained=False),
    "MobileNetV3-L + CBAM": lambda: BioLiteNet(NUM_CLASSES, use_cbam=True, cbam_stages=[3, 6, 12], pretrained=False),
    "ResNet-50": r50,
}


def main():
    x = torch.randn(1, 3, 224, 224)
    nets = {k: f().eval() for k, f in MODELS.items()}
    rows = {}
    for k, m in nets.items():
        macs, _ = profile(m, inputs=(x,), verbose=False)
        rows[k] = {"model": k, "params_M": sum(p.numel() for p in m.parameters()) / 1e6, "GMACs": macs / 1e9}
    for threads in (1, 4):
        torch.set_num_threads(threads)
        times = {k: [] for k in nets}
        with torch.inference_mode():
            for k, m in nets.items():
                for _ in range(WARMUP):
                    m(x)
            for _ in range(ROUNDS):
                for k, m in nets.items():
                    for _ in range(ITERS):
                        t0 = time.perf_counter(); m(x); times[k].append((time.perf_counter() - t0) * 1e3)
        for k, t in times.items():
            t.sort()
            rows[k][f"ms_median_{threads}t"] = statistics.median(t)
            rows[k][f"ms_p90_{threads}t"] = t[int(0.9 * len(t))]
    out = os.path.join(HERE, "analysis", "efficiency.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(rows.values())))); w.writeheader()
        for r in rows.values():
            w.writerow({k: f"{v:.3f}" if isinstance(v, float) else v for k, v in r.items()})
    print(f"CPU: {platform.processor()} | torch {torch.__version__} | batch 1, 224x224, fp32")
    for r in rows.values():
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
    print("wrote", out)


if __name__ == "__main__":
    main()
