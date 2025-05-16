import re
import os
from scripts.executor import run_command

# Define regex patterns
SHIFTS_PATTERN = re.compile(r'#define\s+SHIFTS\s+\d+')
TILES_PATTERN = re.compile(r'#define\s+NUM_TILES\s+\d+')
THREADS_PATTERN = re.compile(r'#define\s+THREADS_PER_BLOCK\s+\d+')

def update_define(filepath, pattern, define_name, value):
    """Replace a #define in a source file."""
    updated = False
    with open(filepath, 'r') as f:
        lines = f.readlines()
    with open(filepath, 'w') as f:
        for line in lines:
            if pattern.match(line):
                f.write(f"#define {define_name} {value}\n")
                updated = True
            else:
                f.write(line)
    if not updated:
        raise RuntimeError(f"#define {define_name} not found in {filepath}")

def update_defines(src_path, shift=None, tiles=None, threads=None):
    """Convenience wrapper to update all #defines."""
    if shift is not None:
        update_define(src_path, SHIFTS_PATTERN, "SHIFTS", shift)
    if tiles is not None:
        update_define(src_path, TILES_PATTERN, "NUM_TILES", tiles)
    if threads is not None:
        update_define(src_path, THREADS_PATTERN, "THREADS_PER_BLOCK", threads)

def compile_target(target, cwd=None):
    makefile_path = os.path.join(cwd or ".", "Makefile")
    if not os.path.isfile(makefile_path):
        raise FileNotFoundError(f"No Makefile found at: {makefile_path}")
    run_command(["make", "clean"], cwd=cwd)
    run_command(["make", target], cwd=cwd)

