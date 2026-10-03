"""
SutraOpt MIPLIB 2017 Benchmark Runner
Downloads easy-tier MIPLIB 2017 and Netlib instances and benchmarks SutraOpt
performance vs known optimal objective values, reporting solve time, status, and gap.
"""
import os
import time
import urllib.request
import shutil
from sutraopt.engine import SutraOptEngine
from sutraopt.parsers.mps_parser import MPSParser

# Easy-tier MIPLIB 2017 instances and their known optimal objective values
MIPLIB_INSTANCES = {
    "afiro":       {"url": "https://miplib.zib.de/WebData/instances/afiro.mps.gz", "known_opt": -464.7531},
    "blend":       {"url": "https://miplib.zib.de/WebData/instances/blend.mps.gz", "known_opt": -30.81},
    "25fv47":      {"url": "https://miplib.zib.de/WebData/instances/25fv47.mps.gz", "known_opt": 5501.846},
}

NETLIB_INSTANCES = {
    "adlittle":    {"url": "https://netlib.org/lp/data/adlittle", "known_opt": 225494.963},
    "brandy":      {"url": "https://netlib.org/lp/data/brandy",   "known_opt": 1518.5099},
}

def download_instance(url: str, local_path: str) -> bool:
    """Download a benchmark instance file if not already cached."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        return True
    try:
        print(f"  Downloading {os.path.basename(local_path)} ...")
        urllib.request.urlretrieve(url, local_path)
        if local_path.endswith(".gz"):
            import gzip
            out_path = local_path[:-3]
            with gzip.open(local_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(local_path)
        return True
    except Exception as e:
        print(f"  [WARN] Download failed: {e}")
        return False


def run_miplib_benchmark(time_limit: float = 120.0, gap_tol: float = 0.01):
    """Run all configured benchmark instances and report results."""
    engine = SutraOptEngine()
    cache_dir = "benches/miplib_cache"
    results = []

    all_instances = {
        **{f"miplib_{k}": v for k, v in MIPLIB_INSTANCES.items()},
        **{f"netlib_{k}": v for k, v in NETLIB_INSTANCES.items()},
    }

    print("\n" + "="*70)
    print(" SUTRAOPT MIPLIB/NETLIB BENCHMARK SUITE")
    print("="*70)
    print(f"{'Instance':<20} {'Status':<12} {'Obj Value':>14} {'Known Opt':>12} {'Gap%':>8} {'Time(ms)':>10}")
    print("-"*70)

    for name, meta in all_instances.items():
        short_name = name.replace("miplib_","").replace("netlib_","")
        ext = ".mps.gz" if ".gz" in meta["url"] else ".mps"
        local = os.path.join(cache_dir, short_name + ext.replace(".gz",""))

        if not download_instance(meta["url"], os.path.join(cache_dir, short_name + ext)):
            results.append({"name": name, "status": "DOWNLOAD_FAIL"})
            print(f"  {name:<18} DOWNLOAD_FAIL")
            continue

        local_mps = local if not local.endswith(".gz") else local[:-3]
        if not os.path.exists(local_mps):
            print(f"  {name:<18} FILE_NOT_FOUND")
            continue

        try:
            t0 = time.perf_counter()
            model = MPSParser.parse_file(local_mps)
            res = engine.solve_model(model)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            known = meta.get("known_opt", float("nan"))
            gap = abs(res.objective_value - known) / max(1.0, abs(known)) * 100 if known == known else float("nan")

            results.append({
                "name": name,
                "status": res.status,
                "obj": res.objective_value,
                "known": known,
                "gap_pct": gap,
                "time_ms": elapsed_ms
            })
            print(f"  {name:<20} {res.status:<12} {res.objective_value:>14.4f} {known:>12.4f} {gap:>7.3f}% {elapsed_ms:>9.1f}ms")
        except Exception as e:
            print(f"  {name:<20} ERROR: {e}")
            results.append({"name": name, "status": "ERROR", "error": str(e)})

    print("="*70)
    passed = sum(1 for r in results if r.get("status") == "OPTIMAL")
    print(f"\nSummary: {passed}/{len(results)} instances solved to OPTIMAL")
    return results


if __name__ == "__main__":
    run_miplib_benchmark()
