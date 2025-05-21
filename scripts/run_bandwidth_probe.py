import os
import yaml
import re
import subprocess
import argparse

CONFIG_KEYS = {
    "min_size_kb": r"#define\s+MIN_SIZE_KB\s+\d+",
    "max_size_mb": r"#define\s+MAX_SIZE_MB\s+\d+",
    "tolerance": r"#define\s+TOLERANCE\s+[0-9.]+",
    "repeats": r"#define\s+REPEATS\s+\d+"
}

def update_define(file_path, pattern, key, value):
    updated = False
    with open(file_path, 'r') as f:
        lines = f.readlines()

    with open(file_path, 'w') as f:
        for line in lines:
            if re.match(pattern, line):
                f.write(f"#define {key.upper()} {value}\n")
                updated = True
            else:
                f.write(line)
    
    if not updated:
        raise RuntimeError(f"Failed to update #define {key.upper()} in {file_path}")

def update_defines_from_config(config, source_path):
    for key, pattern in CONFIG_KEYS.items():
        if key in config["parameters"]:
            update_define(source_path, pattern, key, config["parameters"][key])

def run_bandwidth_probe(executable_path, mode, debug):
    env = os.environ.copy()
    args = [f"./{executable_path}", mode]
    if debug:
        args.append("debug")
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Benchmark execution failed.")
    print(result.stdout)

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    benchmark_dir = os.path.join("benchmarks", config["benchmark_name"])
    src_path = os.path.join(benchmark_dir, "src", "bandwidth_probe.cu")
    executable = "bandwidth_probe.exe"

    # Step 1: Update #defines in source
    update_defines_from_config(config, src_path)

    # Step 2: Compile
    subprocess.run(["make", executable], cwd=benchmark_dir, check=True)

    # Step 3: Run benchmark
    mode = config["parameters"].get("mode", "H2D")
    debug = config["parameters"].get("debug", False)
    run_bandwidth_probe(os.path.join(benchmark_dir, executable), mode, debug)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
