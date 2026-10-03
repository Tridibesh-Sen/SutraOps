"""
SutraOpt Command Line Interface (CLI)
Usage:
  python sutra_cli.py solve <input_file> [--certify] [--out <output_json>]
  python sutra_cli.py info <input_file>
  python sutra_cli.py generate <input_file> [--lang python|cpp]
"""

import sys
import os
import argparse
from sutraopt.engine import SutraOptEngine
from sutraopt.parsers.mps_parser import MPSParser
from sutraopt.parsers.json_schema import JSONSchemaParser
from sutraopt.auto.code_generator import StandaloneCodeGenerator


def main():
    parser = argparse.ArgumentParser(description="SutraOpt (BharatOpt) Sovereign Optimization CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # solve command
    solve_parser = subparsers.add_parser("solve", help="Solve an optimization model")
    solve_parser.add_argument("file", help="Path to input file (.mps, .qps, .json)")
    solve_parser.add_argument("--gpu", action="store_true", help="Enable GPU-accelerated sparse linear algebra & SpMV")
    solve_parser.add_argument("--precision", default="mixed", choices=["mixed", "fp64", "fp32"], help="Precision mode (mixed with FP64 refinement)")
    solve_parser.add_argument("--certify", action="store_true", help="Generate Markdown Certificate of Optimality")
    solve_parser.add_argument("--time-limit", type=float, default=None, help="Stop solve after N seconds")
    solve_parser.add_argument("--gap", type=float, default=1e-3, help="Optimality gap tolerance (default: 1e-3)")
    solve_parser.add_argument("--log-level", default="normal", choices=["quiet", "normal", "verbose"], help="Solver log verbosity")
    solve_parser.add_argument("--out", type=str, help="Save solution JSON to destination file")

    # miplib command
    miplib_parser = subparsers.add_parser("miplib", help="Run MIPLIB 2017 / Netlib benchmark suite")
    miplib_parser.add_argument("--time-limit", type=float, default=120.0, help="Per-instance time limit in seconds")
    miplib_parser.add_argument("--gap", type=float, default=0.01, help="MIP gap tolerance")

    # benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run comparative benchmark vs HiGHS/CPLEX & stress tests")

    # info command
    info_parser = subparsers.add_parser("info", help="Inspect and profile model topology")
    info_parser.add_argument("file", help="Path to input file")

    # generate command
    gen_parser = subparsers.add_parser("generate", help="Generate standalone embedded solver code")
    gen_parser.add_argument("file", help="Path to input file")
    gen_parser.add_argument("--lang", default="python", choices=["python", "cpp"], help="Target language")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "benchmark":
        from benches.benchmark_comparison import run_comprehensive_benchmark
        run_comprehensive_benchmark()
        return

    if args.command == "miplib":
        from benches.miplib_runner import run_miplib_benchmark
        run_miplib_benchmark(
            time_limit=args.time_limit,
            gap_tol=args.gap
        )
        return

    if args.command == "solve":
        from sutraopt.linalg.gpu_acceleration import GPULinearAlgebraEngine
        gpu_engine = GPULinearAlgebraEngine(force_cpu=not args.gpu)
        dev_info = gpu_engine.get_device_info()
        print(f"[*] Compute Device: {dev_info['device_name']} [{dev_info['status']}]")
        if args.gpu:
            print(f"[*] GPU SpMV & Tensor-Core Acceleration ACTIVE (Mixed Precision: {args.precision})")
        engine = SutraOptEngine(use_gpu=args.gpu)
        print(f"[*] Ingesting model: {args.file}")
        res = engine.solve_file(args.file, gap_tol=args.gap)

        print("\n" + "=" * 60)
        print(f" SUTRA-OPT SOLVER REPORT: {res.topology.get('name', 'Model')}")
        print("=" * 60)
        print(f" Status:             {res.status}")
        print(f" Problem Class:      {res.topology.get('class', 'LP')}")
        print(f" Objective Value:    {res.objective_value:,.4f}")
        print(f" Iterations / Nodes: {res.iterations_or_nodes}")
        print(f" Solve Time:         {res.solve_time_seconds * 1000:.3f} ms")
        print(f" KKT Residual Bound: {res.certificate.primal_residual:.2e}")
        print(f" SHA-256 Digest:     {res.certificate.sha256_hash}")
        print("=" * 60)

        print("\nDecision Variables (Top Non-Zeros):")
        nonzero_vars = {k: v for k, v in res.solution_vector.items() if abs(v) > 1e-6}
        for k, v in list(nonzero_vars.items())[:15]:
            print(f"  * {k:25s} = {v:12.4f}")
        if len(nonzero_vars) > 15:
            print(f"  ... and {len(nonzero_vars) - 15} more variables.")

        if args.certify:
            cert_filename = f"certificate_{os.path.basename(args.file)}.md"
            with open(cert_filename, "w", encoding="utf-8") as f:
                f.write(res.certificate.to_markdown())
            print(f"\n[+] Cryptographic Certificate written to: {cert_filename}")

        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(res.to_json())
            print(f"[+] Full solution JSON saved to: {args.out}")

    elif args.command == "info":
        if args.file.endswith(".mps") or args.file.endswith(".qps"):
            model = MPSParser.parse_file(args.file)
        else:
            with open(args.file, "r", encoding="utf-8") as f:
                model = JSONSchemaParser.parse_string(f.read())
        topo = model.summarize()
        print("\n--- Problem Topology Profile ---")
        for k, v in topo.items():
            print(f"  {k:20s}: {v}")

    elif args.command == "generate":
        if args.file.endswith(".mps") or args.file.endswith(".qps"):
            model = MPSParser.parse_file(args.file)
        else:
            with open(args.file, "r", encoding="utf-8") as f:
                model = JSONSchemaParser.parse_string(f.read())

        if args.lang == "python":
            code = StandaloneCodeGenerator.generate_python_script(model)
        else:
            code = StandaloneCodeGenerator.generate_cpp_script(model)

        out_name = f"standalone_{model.name}.{'py' if args.lang == 'python' else 'cpp'}"
        with open(out_name, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"[+] Standalone {args.lang.upper()} solver kernel written to: {out_name}")


if __name__ == "__main__":
    main()
