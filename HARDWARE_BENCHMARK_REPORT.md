# Hardware and backend benchmark report

> **Final audit note (2026-11-07):** The hardware measurements below are preserved historical platform/provenance evidence. No additional hardware tests were run for the final audit. DirectML, CPU, and Apple MPS results are not pooled as a controlled comparison; no NPU, ROCm, or new benchmark claim is made.

Inspection evidence is in [hardware_capabilities.md](results/hardware/hardware_capabilities.md); timed microbenchmark rows are in [backend_benchmark.csv](results/hardware/backend_benchmark.csv). Current execution is on the Ryzen AI 7 350 Windows machine with Radeon 860M and XDNA2 NPU. The 2026-11-05 recorded inventory identifies 23.3 GiB RAM, Windows 11, GPU driver 32.0.22024.3004 and NPU driver 32.0.203.314. Integrated GPU WMI adapter memory (512 MiB) is not the total shared-memory limit.

## Measured microbenchmark

The same small MLP and inputs were measured on CPU and Radeon 860M through DirectML. CPU: 171,088 inference samples/s, 0.197 ms median batch latency, 344.3 training steps/s, 0.0011 s model copy/load. DirectML: 215,569 inference samples/s, 0.500 ms median latency, 295.3 steps/s, 0.240 s model copy/load. DirectML’s Adam path warned about a CPU fallback. This is a short synthetic microbenchmark, not a robust throughput estimate or proof of full GPU residency. Per-process GPU utilization and VRAM are unavailable from the recorded instrumentation.

## Accelerator status

PyTorch ROCm is absent from the working environment. DirectML was verified in a separate environment and used for eligible small neural-network workloads; unsupported operations may fall back to CPU. The NPU is enumerated and driver-loaded, but no compatible NPU execution provider was installed and verified, so this project has no NPU inference measurement. AMD ROCm docs describe compatibility requiring newer graphics drivers than the installed version at inspection; no driver replacement was made. Classical estimators and RL ran on CPU.

See `results/hardware/hardware_capabilities.md` for the detailed detection commands, provider checks, sources, and scheduling caveats. No experiment is labeled NPU accelerated.
