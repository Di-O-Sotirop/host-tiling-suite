import os
from .config_utils import load_config
from .compiler_utils import update_defines, compile_target
from .executor import run_executable

def benchmark_instance(src_path, exe_path, make_target, shift, tiles, threads, repeats, verify, debug, benchmark_dir):
    update_defines(src_path, shift=shift, tiles=tiles, threads=threads)
    compile_target(make_target, cwd=benchmark_dir)
    return run_executable(exe_path, repeats, verify, debug)


def main(config_path):
    cfg = load_config(config_path)

    benchmark_name = cfg["benchmark_name"]
    benchmark_dir = os.path.join("benchmarks", benchmark_name)
    src_dir = os.path.join(benchmark_dir, "src")

    shifts = cfg["parameters"]["shifts"]
    tiles_list = cfg["parameters"]["num_tiles"]
    repeats = cfg["parameters"]["repeats"]
    verify = cfg["parameters"].get("verify", False)
    threads = cfg["parameters"].get("threads_per_block", 256)
    debug = cfg["parameters"].get("debug", False)
    measure_end2end = cfg["parameters"].get("measure_end2end", False)
    if measure_end2end:
        verify = False

    src_baseline = os.path.join(src_dir, cfg["source"]["baseline"])
    src_tiled = os.path.join(src_dir, cfg["source"]["tiled"])
    exe_baseline = os.path.join(benchmark_dir, cfg["executable"]["baseline"])
    exe_tiled = os.path.join(benchmark_dir, cfg["executable"]["tiled"])
    make_target_baseline = cfg["make_targets"]["baseline"]
    make_target_tiled = cfg["make_targets"]["tiled"]

    results = {}
    baseline_cache = {}

    for shift in shifts:
        print(f"\n[RUN] SHIFTS={shift}")
        # Run baseline once per shift
        try:
            print("  [BASELINE]")
            update_defines(src_baseline, shift=shift, tiles=None, threads=threads)
            compile_target(make_target_baseline, cwd=benchmark_dir)
            res = run_executable(exe_baseline, repeats, verify, debug)
            baseline_kernel = res["kernel"][0]
            baseline_e2e = res["end2end"][0] if measure_end2end else None
            baseline_cache[shift] = baseline_kernel
        except Exception as e:
            print(f"[ERROR] Baseline run failed for SHIFTS={shift}: {e}")
            baseline_cache[shift] = None
            continue

        for tiles in tiles_list:
            print(f"  [TILED] TILES={tiles}")
            try:
                res = benchmark_instance(
                    src_tiled, exe_tiled, make_target_tiled,
                    shift, tiles, threads, repeats, verify, debug,
                    benchmark_dir=benchmark_dir
                )
                tiled_kernel = res["kernel"][0]
                tiled_e2e = res["end2end"][0] if measure_end2end else None
                tiled_time = tiled_kernel  # For legacy use

                baseline_kernel = baseline_cache.get(shift)
                speedup_kernel = (baseline_kernel / tiled_kernel) if isinstance(baseline_kernel, (int, float)) and isinstance(tiled_kernel, (int, float)) else None
                speedup_e2e = (baseline_e2e / tiled_e2e) if measure_end2end and isinstance(baseline_e2e, (int, float)) and isinstance(tiled_e2e, (int, float)) else None

                results[(shift, tiles)] = {
                    "tiled_kernel": tiled_kernel,
                    "speedup_kernel": speedup_kernel,
                    "tiled_e2e": tiled_e2e,
                    "speedup_e2e": speedup_e2e
                }

                if isinstance(tiled_kernel, (int, float)):
                    print(f"    [Result] Tiled: {tiled_kernel:.3f} ms | Speedup: {speedup_kernel:.2f}" if speedup_kernel else "    [Result] Speedup: N/A")
                else:
                    print("    [Result] Tiled run failed.")

                if measure_end2end:
                    if isinstance(tiled_e2e, (int, float)):
                        print(f"    [Result] End-to-End: {tiled_e2e:.3f} ms | Speedup: {speedup_e2e:.2f}" if speedup_e2e else "    [Result] End-to-End Speedup: N/A")
                    else:
                        print("    [Result] End-to-End run failed.")

            except Exception as e:
                print(f"    [ERROR] Tiled run failed: {e}")
                results[(shift, tiles)] = {}


    # Print summary
    print("\n=== Summary: Speedup vs Baseline ===")
    header = "SHIFTS\\TILES" + "".join([f"{t:>10}" for t in tiles_list])
    print(header)
    print("-" * len(header))
    for shift in shifts:
        row = f"{shift:>12}"
        for tiles in tiles_list:
            result = results.get((shift, tiles), {})
            speedup_kernel = result.get("speedup_kernel", None)
            if isinstance(speedup_kernel, (int, float)):
                val = f"{speedup_kernel:.2f}"
            else:
                val = "N/A"
            row += f"{val:>10}"
        print(row)

    if measure_end2end:
        print("\n=== Summary: End-to-End Speedup ===")
        header = "SHIFTS\\TILES" + "".join([f"{t:>10}" for t in tiles_list])
        print(header)
        print("-" * len(header))
        for shift in shifts:
            row = f"{shift:>12}"
            for tile in tiles_list:
                speedup_e2e = results.get((shift, tile), {}).get("speedup_e2e", None)
                if isinstance(speedup_e2e, (int, float)):
                    val = f"{speedup_e2e:.2f}"
                else:
                    val = "N/A"
                row += f"{val:>10}"
            print(row)
