import subprocess
import os
import statistics
import re

# Regex to extract execution time
EXEC_TIME_PATTERN = re.compile(r'Execution Time:\s+([\d.]+)\s*ms')
END2END_TIME_PATTERN = re.compile(r'Total Execution Time:\s+([\d.]+)\s*ms')

def run_command(cmd, env=None, cwd=None):
    """Run a shell command and return stdout."""
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=cwd
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout
    
def run_executable(exe, repeats, verify, debug):
    kernel_times = []
    end2end_times = []
    env = os.environ.copy()
    env["VERIFY"] = "1" if verify else "0"

    for i in range(repeats):
        output = run_command([f"./{exe}"], env=env)
        
        match_exec = EXEC_TIME_PATTERN.search(output)
        match_e2e = END2END_TIME_PATTERN.search(output)

        if match_exec:
            kernel_times.append(float(match_exec.group(1)))
        if match_e2e:
            end2end_times.append(float(match_e2e.group(1)))

    def stats(values):
        return (sum(values) / len(values), statistics.stdev(values)) if values else (None, None)

    kernel_avg, kernel_std = stats(kernel_times)
    e2e_avg, e2e_std = stats(end2end_times)

    return {
        "kernel": (kernel_avg, kernel_std),
        "end2end": (e2e_avg, e2e_std)
    }
