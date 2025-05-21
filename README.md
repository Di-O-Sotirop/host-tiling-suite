This is an ongoing work for a host-tiling suit. 

It includes 
- gemver: the gemver benchmark tiled in stripes for overlap of Kernel_1 and the transfer of A.
- vadd: a 3 vector add benchmark with parameterizable iteration count on a loop performing the computation. Used to emulate memory-compute bount kernels.
- bandwidth_probe: an iterative algorithm to find the minimum transfer size of nominal bandwidth of the current device within a 5% error of the maximum measured bandwidth.

Execute:
**install nvcc, python3.6**
**edit the corresponding configuration file** (/config) to set the parameters of the experiment.
You can run each benchmark via the corresponding python wrapper script: **bandwidth_probe.py  gemver_benchmark.py vadd_benchmark.py** (if it doesn't work, do cat and run the run it from the command line).
