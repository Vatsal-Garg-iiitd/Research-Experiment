# Hardware and runtime capability report

Inspection date: 2026-11-05 (local Windows system).

## Detected hardware

| Component | Observed value | Evidence / caveat |
|---|---|---|
| CPU | AMD Ryzen AI 7 350, 8 cores / 16 logical processors | `Win32_Processor` |
| GPU | AMD Radeon(TM) 860M Graphics, driver 32.0.22024.3004 | `Win32_VideoController`; integrated GPU |
| GPU architecture | RDNA 3.5, target gfx1152 | AMD ROCm GPU specification for Ryzen AI 7 350 |
| GPU memory | 512 MiB reported as adapter RAM | WMI's dedicated adapter RAM field; this is not the shared system-memory limit of an integrated GPU |
| NPU | Integrated AMD XDNA 2 NPU in the Ryzen AI 7 350 (Windows names it “NPU Compute Accelerator Device”); AMD specifies up to 50 TOPS for this CPU family | PnP PCI device `VEN_1022&DEV_17F0`, driver 32.0.203.314; device status OK; [AMD product specifications](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-7-350.html) |
| System RAM | 24,422,772 KiB visible (~23.3 GiB); 4,558,632 KiB free at inspection | Windows CIM; availability changes over time |
| Storage | C: had ~861.8 GB free | PowerShell filesystem inventory |
| OS | Windows 11 Home Single Language, 64-bit, build 26300 | Windows CIM |

## Backend checks

- The pre-existing `.venv` contains PyTorch 2.15.0+cpu. `torch.cuda.is_available()` is false and `torch.version.hip` is `None`. No ROCm runtime/device was detected; this Windows installation is not running a ROCm PyTorch build.
- A separate `.venv-dml` was created so the working environment remained intact. It contains `torch-directml==0.2.5.dev240914`, which pins `torch==2.4.1`. Device enumeration returned one DirectML device (`privateuseone:0`), and a tensor matrix multiplication succeeded.
- A 30-step MLP training smoke benchmark on DirectML completed in 0.780 s (~38.5 steps/s). PyTorch DirectML warned that Adam's `aten::lerp.Scalar_out` fell back to CPU.
- A same-model/same-input microbenchmark was saved in `results/hardware/backend_benchmark.csv`. CPU: 171,088 samples/s inference, 0.197 ms median batch latency, 344.3 SGD steps/s. Radeon 860M DirectML: 215,569 samples/s, 0.500 ms median batch latency, 295.3 SGD steps/s. DirectML model copy/load took 0.240 s versus 0.0011 s on CPU. Results are a short synthetic MLP measurement; warm-up and sample size are limited, so they should not be generalized to other models.
- Windows exposes global GPU engine and memory performance-counter paths. The benchmark did not isolate per-process utilization, and the tested DirectML API did not expose VRAM allocation. These fields remain unavailable rather than being estimated.
- DirectML is therefore the available GPU backend for supported PyTorch operations. Unsupported operators may fall back to CPU; a device label alone does not prove every operator ran on the GPU.
- AMD's ROCm 7.14.1 specifications now list Ryzen AI 7 350 / Radeon 860M (gfx1152) for Windows 11 25H2, but require AMD Adrenalin 26.6.4 or OEM driver 26.10.28. The installed display driver is dated 2025-10-30 and reports 32.0.22024.3004; it is older than that listed ROCm driver requirement. I did not replace the system display driver or trigger a reboot. ROCm was therefore not installed in the current OS state; DirectML was used and verified instead. Sources: https://rocm.docs.amd.com/en/docs-7.14.1/compatibility/compatibility-matrix.html and https://rocm.docs.amd.com/en/docs-7.14.1/reference/gpu-specs.html
- The NPU is present and its driver is version 32.0.203.314. No Ryzen AI / VitisAI execution provider, ONNX Runtime NPU provider, or Windows ML Python runtime was found in the project environment. The NPU has not yet run an inference benchmark. It is treated as unavailable to these experiments until a supported runtime and successful provider/device execution are verified.
- No NPU training capability is claimed. AMD documents Ryzen AI primarily as an ONNX deployment/inference flow; operator coverage depends on the runtime and model.

## Scheduling policy used for this project

- Neural-network training is eligible for DirectML only after the model's operators pass a device smoke test; otherwise it runs on CPU and is labeled accordingly.
- Classical scikit-learn estimators and graph algorithms run on CPU.
- Reinforcement-learning environment interactions run on CPU; only neural updates may use an accelerator if the implementation supports it.
- NPU inference is disabled because the provider/runtime is not installed and verified.
- The existing CPU environment is preserved. DirectML uses a separate environment.

## Compatibility sources

- Microsoft documents PyTorch DirectML for DirectX 12-capable GPUs on Windows: https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows
- ONNX Runtime documents DirectML requirements and GPU provider: https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- AMD Ryzen AI / Windows ML documentation describes VitisAI EP use and driver requirements: https://ryzenai.docs.amd.com/projects/WinML/en/latest/installation.html and https://ryzenai.docs.amd.com/en/latest/app_development.html

