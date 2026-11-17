# cuRF-Stream 计划

> 基于自研 USB3 SDR 传输层（OmniCore-SDR-Host），构建 GPU 实时流处理流水线和自研推理引擎，在消费级显卡上实现全带宽实时信号识别。

三条主线：**自研推理引擎**、**实时流水线**、**传输层与 GPU 的跨层协同**。信号识别是负载，不是目的。

## 1. 原则

- **数字先于功能**：每个阶段都有可复现的 bench 和验收指标，结果写入 `docs/results/`。
- **正确性先于性能**：每个 GPU kernel 先写朴素版本并通过 CPU 参考测试，朴素版本保留为 bench 基线。
- **每个阶段结束时都有完整成果**，后面的阶段没做完也不影响前面成果的完整性。
- **模型尽早冻结**：P0 训练出能用的模型后不再改结构，引擎只需支持它用到的算子。
- **两个仓库边界清晰**：传输层改动提交到 OmniCore-SDR-Host，cuRF-Stream 只通过它的 C ABI 依赖它。

## 2. 环境

| | |
|---|---|
| 开发机 | Apple M4 Mac：编辑、CPU 部分编译与测试 |
| CUDA 主机 | 局域网 `192.168.1.107`（笔记本），Ubuntu 24.04.4，Linux 7.0 |
| GPU | RTX 3050 Laptop，4 GB GDDR6，sm_86，16 SM / 64 Tensor Core，功耗上限 60 W（最高 75 W），PCIe 4.0 x16；未驱动显示器 |
| CPU | Intel i5-11400H，6 核 12 线程，**支持 AVX-512**；内存 15 GB |
| 工具链 | CUDA 12.8、驱动 580、cuDNN 9.21、GCC 13.3、CMake 3.28 |
| Python | anaconda3 的 **`curf` 环境**（本项目专用）：Python 3.11.0、PyTorch 2.11.0+cu128、torchaudio 2.11.0+cu128、TorchSig 2.2.0、onnx 1.23.2、onnxruntime 1.32.0（CPU）、numpy 2.4.6、scipy 1.17.1。已有的 `cu128`、`vllm` 环境不动 |
| TensorRT | **11.4.0**（cuda12.9 构建，与 CUDA 12.8 兼容，已验证），装在用户目录 `~/opt/tensorrt-11.4/usr`，未改动系统；`source ~/opt/tensorrt-11.4/env.sh` 设置 `TENSORRT_ROOT`、`PATH`、`LD_LIBRARY_PATH` |

TensorRT 11 的两个注意点：

- **只支持强类型网络**：`trtexec --fp16` 已被移除，精度完全由 ONNX 中的数据类型决定。FP16 基线要导出 FP16 的 ONNX（或在图中显式插入 Cast）。
- **不用它自带的 CMake 配置**：`TensorRT-EnterpriseConfig.cmake` 强制要求 Windows 交叉构建资源包（额外 1.1 GB）。项目自带 `cmake/FindTensorRT.cmake`，用 `find_path` / `find_library` 定位 `NvInfer.h`、`libnvinfer`、`libnvonnxparser`。

冒烟测试（`~/opt/trt-smoke`）：一个两层 conv1d 的小模型，FP16，batch 256，trtexec 测得 2177 qps（GPU 时间中位数 0.46 ms）；C++ 程序能通过 CMake 编译并解析 ONNX。

估算用的算力参考（理论值，以实测为准）：

- GPU FP32 约 8 TFLOPS；FP16 Tensor Core 约 20+ TFLOPS（FP32 累加时减半）；显存带宽约 192 GB/s。
- CPU AVX-512 FP32 理论峰值约 0.7 TFLOPS。
- 笔记本有功耗和温度墙，所有性能结论都要以 300 秒以上长测为准。

## 3. 系统架构

```
CPU（C++ 异步）                          GPU（CUDA）
──────────────────                      ──────────────────────────────
USB 异步收包（OmniCore 中间件）
SPSC 无锁环形缓冲（锁页内存）
        │ H2D（异步，多 stream 重叠）
        └──────────────────────────────▶ ① 12bit 解包 + DC/IQ 校正
                                         ② 宽带 PSD / 瀑布图
                                         ③ 多相信道化（融合 FIR + cuFFT）
                                         ④ 能量检测 + 活跃信道压缩
                                         ⑤ 切窗、归一化，融合进第一层输入
                                         ⑥ 调制识别推理（自研引擎）
                                         ⑦ softmax + 时间平滑投票
        ┌────────────────────────────── D2H（识别结果 + 瀑布图）
调度：动态 batching、背压、丢样计量
显示 / 日志 / bench 统计
```

数据源通过 `ISource` 接口可插拔：

- **合成源**：可超过 USB 速率，用于把系统推到 PCIe 和 GPU 的极限。
- **文件回放源**：保证可复现。
- **OmniCore 实采源**：真实硬件验证。

## 4. 负载估算

输入：2 路 × 61.44 MSPS = 122.88 MS/s。信道化：512 信道 × 120 kHz，2 倍过采样，每信道输出 240 kS/s。

| 环节 | 估算 |
|---|---|
| 解包、PSD | 访存密集，很小 |
| 多相信道化 | 约 23 GFLOPS |
| 推理（小 ResNet-1D，约 50 MFLOP / 1024 点窗口） | 每信道约 234 窗口/秒 |

推理负载随活跃信道比例变化：

| 场景 | 推理次数/秒 | 算力需求 | CPU（AVX-512） | GPU |
|---|---|---|---|---|
| 只做信道化 | — | 约 23 GFLOPS | 可行，但占用多核 | 很轻 |
| 10% 信道活跃 | 约 2.4 万 | 约 1.2 TFLOPS | 超出峰值 | 有余量 |
| 全部 1024 信道 | 约 24 万 | 约 12 TFLOPS | 差一个数量级以上 | 压线，需要 Tensor Core 高利用率 |

这张表的每一格都要在 P3 用实测替换：GPU 和 CPU 基线各自能跑到哪一档，就是"为什么用 GPU"的答案。

延迟预算：攒满一个窗口需要 4.3 ms（算法本身的主导项），USB 请求约 0.5 ms，H2D 几十 µs，推理 batch 约 1–3 ms。目标是端到端 p99 < 15 ms。

## 5. 范围

**做：**

- 流式运行时：锁页 ring、多 stream 流水、CUDA Graphs、背压与丢样计量。
- 自研推理引擎：小 ResNet-1D，FP16，手写 Tensor Core 卷积，静态内存规划，动态 batching，与 TensorRT / cuDNN 对比。
- 信道化：cuFFT 加融合 FIR kernel。
- CPU 基线：用于证明 GPU 的必要性。
- 端到端：真实 SDR 长测，端到端延迟统计。

**不做（除非有明确余力）：**

- Transformer、DOA、TorchSig 大模型。
- 通用 ONNX 支持：引擎只支持 conv1d、bias、ReLU、残差加、池化、全连接、softmax 这几种算子。
- 复杂 UI：只做瀑布图加识别标注。
- 追求模型准确率。

## 6. 阶段计划

### P0 准备（第 1 周）

- 确认 SDR 能接到 CUDA 主机上。（`curf` conda 环境和 TensorRT 11.4 已装好）
- CMake 工程骨架：CMakePresets，Mac 上只编 CPU 部分，CUDA 主机上编全部（`CMAKE_CUDA_ARCHITECTURES=86`）。
- 训练数据：自写 GPU 合成器（10 类，见第 9 节），参数匹配信道化输出（240 kS/s，1024 点窗口）。
- 训练小 ResNet-1D（PyTorch），导出 ONNX 和原始权重，**冻结结构**。

**成果**：环境可用；模型文件和 PyTorch 参考输出（用于后面逐层对齐）。

### P1 流水线骨架（第 2–3 周）

- `ISource` 接口和合成源、文件回放源。
- 锁页环形缓冲（`cudaHostAlloc` / `cudaHostRegister`）。
- 12bit / 8bit 解包 kernel、宽带 PSD。
- 多 stream 流水：H2D、计算、D2H 三级重叠，缓冲深度可配置。
- bench 框架（吞吐、延迟分位数、丢样计数，输出 JSON）；测试框架（Catch2，与 CPU 参考逐样点比对）；NVTX 打点。

**验收**：
- 解包 kernel 达到显存带宽的 70% 以上。
- H2D 达到 PCIe 实测上限的 90% 以上。
- Nsight Systems 时间线显示传输与计算完全重叠。

### P2 推理引擎（第 4–8 周）

**离线部分（Python）**：ONNX → 自写转换器：BN 折叠、常量折叠、conv + bias + 激活 + 残差融合、布局选择，输出自定义二进制格式（图 + FP16 权重）。

**运行时部分（C++ / CUDA）**：
- 按张量生命周期做静态内存规划，热路径零 `cudaMalloc`。
- 固定 batch 档位（如 1 / 8 / 32 / 128 / 512），每档捕获一个 CUDA Graph。
- kernel 注册表：每个形状多个变体，离线自动调优并缓存结果。
- 逐层与 PyTorch 输出对齐，给出误差容限。

**conv1d（implicit GEMM）逐步优化**，每一步都记录 Nsight Compute 数据：
1. 朴素版本
2. shared memory 分块
3. WMMA Tensor Core
4. epilogue 融合（bias、激活、残差）
5. （余力）`mma.sync` + `ldmatrix` + `cp.async` 多级流水

**对比基线**：cuDNN、CUTLASS、TensorRT FP16。

**验收**：
- 三个后端的识别结果一致。
- 有完整的优化步骤数据表（每一步的吞吐、Tensor Core 利用率、瓶颈变化）。
- 与 TensorRT FP16 的差距有数据，并能解释差距来源。

**检查点**：P2 结束时，"流水线 + 自研引擎"已经是一个完整、可展示的项目。

### P3 端到端（第 9–10 周）

- 多相信道化：融合 FIR kernel + 批量 cuFFT，与 cupyx 的实现对比，做 roofline 分析。
- 能量检测 + 活跃信道压缩（手写版本与 CUB 对比）。
- 前处理融合：第一层卷积直接从信道化输出做 gather。
- 截止时间驱动的动态 batching。
- CPU 基线（ONNX Runtime CPU 或 oneDNN，开启 AVX-512）。

**验收**：
- 负载旋钮曲线：活跃信道比例从 0 到 100% 时，GPU 与 CPU 各自能否实时，用实测数据替换第 4 节的估算表。
- 端到端延迟分布（p50 / p99 / p999）。

### P4 实采与收尾（第 11–12 周）

- OmniCore 实采源：2×61.44 MSPS，300 秒零丢样长测。
- TX→RX 线缆回环（必须加衰减器）采集已知调制信号，验证实采识别效果；必要时用回环数据微调模型（不改结构）。
- 最简展示界面：瀑布图 + 识别标注。
- 文档整理（见第 7 节）。

**验收**：实采长测零丢样；端到端 p99 达到目标；所有数字都能用 `bench/` 下的脚本复现。

### 余力项（选一项做深）

- conv kernel 的 `mma.sync` + `cp.async` 深度优化。
- INT8 量化（校准、精度损失与提速对比）。
- 中间件零拷贝改造：给 OmniCore-SDR-Host 加外部 buffer 分配器钩子，让 USB 请求直接写入 CUDA 锁页内存，消除一次主机拷贝。Linux 上需先实测 `libusb_dev_mem_alloc` 的内存能否 `cudaHostRegister`。
- USB 回调线程、ring、CUDA stream 统一 NVTX 打点，产出全链路时间线。

## 7. 交付物

1. **README**：架构图、核心结果表（吞吐、延迟、与 TensorRT 对比、CPU 基线）、Nsight 截图。
2. **优化日志**（`docs/optimization-log.md`）：每个 kernel 从朴素到最终版本，每一步的改动、提升和瓶颈变化。
3. **设计文档**（`docs/design.md`）：关键取舍及理由，比如为什么自研引擎而不用 ONNX Runtime、batch 档位怎么定、丢样策略。
4. **一键复现**：`bench/` 下的脚本能重新跑出所有结果。

## 8. 目录结构（初稿）

```
cuRF-Stream/
  CMakeLists.txt  CMakePresets.json
  include/curf/        公共头文件
  src/
    source/            ISource 及三种实现
    runtime/           锁页 ring、流水线、调度、背压
    kernels/           unpack / psd / pfb / detect（.cu）
    engine/            推理引擎运行时与 conv 等 kernel
    reference/         CPU 参考实现
  tools/convert/       ONNX → 自定义格式转换器（Python）
  model/               训练与导出脚本
  bench/               基准程序与扫描脚本
  tests/
  docs/                设计文档、优化日志、results/
```

## 9. 进度记录

### 2026-11-14 · P0 工程骨架

已完成：

- CMake 工程：`cpu-*` 预设（Mac）与 `cuda-*` 预设（Linux，sm_86，带 TensorRT）；`cmake/FindTensorRT.cmake`；Catch2 v3.7.1 通过 FetchContent 拉取（固定 SHA256）。
- `ISource` 接口（调用方提供目标内存，便于直接写入锁页 buffer）；`SyntheticSource`（确定性单音 + 噪声，可按真实采样率限速或全速输出）；`FileSource`（原始 wire 文件回放，可循环）。
- CPU 参考实现：packed12 / signed8 的打包与解包，语义与 OmniCore codec 一致（满量程 [-1, 1)，四舍五入远离零，饱和，NaN 置零）。
- CUDA 解包 kernel：输出按通道分平面的 complex float；`naive`（每线程一个样点）与 `vec4`（每线程 4 个样点，32/64 位读、16 字节写），不满足对齐时自动回退。
- `tools/remote.sh`：rsync 到 CUDA 主机后编译 / 测试 / 跑 bench（主机配置在 git 忽略的 `tools/remote.env`）。
- 测试：Mac 上 9 个 CPU 测试通过；CUDA 主机上 11 个测试全部通过，GPU 解包与 CPU 参考逐比特一致（2 种格式 × 2 个变体 × 1–4 通道 × 多种长度，加非对齐输入）。

首批实测（RTX 3050 Laptop，cuda-release）：

| 项目 | 结果 |
|---|---|
| 合成源主机侧输出 | 18.8 GB/s，USB 速率的 51 倍 |
| 显存理论带宽 / D2D 实测 | 192 GB/s / 183 GB/s（95.5%） |
| H2D（64 MiB） | 锁页 7.08 GB/s，可分页 5.70 GB/s |
| 解包 packed12，2 通道 × 2^24 帧 | naive 173 GB/s（D2D 的 94.3%）；vec4 180 GB/s（98.4%），约为实时需求的 133 倍 |
| 解包 signed8 | naive 175 GB/s（95.6%）；vec4 179 GB/s（97.6%） |

两个发现，留给 P1：

1. **H2D 明显偏低**：链路实际训练为 PCIe 4.0 **x8**（插槽支持 x16，笔记本常见布线），理论约 15.8 GB/s，实测锁页只有 7.08 GB/s（约 45%），而且可分页内存只比锁页慢一点，不正常。（已查明，见下一节）
2. **解包 kernel 已经顶到带宽上限**：naive 版本就达到 D2D 的 94%，向量化只多出 4%。它的耗时主要来自写出 8 字节/样点的 fp32（输入只有 3 字节）。真正的优化方向不在 kernel 内部，而在**不落地 fp32**：输出 fp16，或者把解包直接融合进信道化 FIR 的输入。

### 2026-11-14 · H2D 偏低的根因：IOMMU 的 IOTLB 缺失

现象：锁页 H2D 在 256 KB 时有 10.7–12 GB/s，1 MB 以上掉到约 7 GB/s；可分页内存也只慢一点。

排查（`bench/bench_pcie.cpp`）：

- 主机开启了 Intel IOMMU（Ubuntu 内核 `CONFIG_INTEL_IOMMU_DEFAULT_ON=y`），显卡所在 IOMMU 组的模式为 `DMA-FQ`，即每次 DMA 都要做地址翻译，没有直通。
- `cudaHostAlloc` 的锁页内存按 4 KB 页映射：1 MB 就要 256 个翻译表项，超出 IOTLB 容量后每次 DMA 都要查页表。256 KB（64 个表项）还放得下，所以小块传输正常。
- 验证：用 2 MB 对齐的内存加 `madvise(MADV_HUGEPAGE)`，由透明大页（THP）支撑，再 `cudaHostRegister` 锁页，IOMMU 就能用 2 MB 超级页映射，表项减少 512 倍。

| 大小 | `cudaHostAlloc`（4 KB 页） | THP + `cudaHostRegister`（覆盖率 100%） |
|---|---|---|
| 1 MB | H2D 6.68 / D2H 6.64 GB/s | H2D 11.8–12.0 / D2H 13.0 GB/s |
| 4 MB | 6.80 / 7.02 | 12.4 / 13.4 |
| 256 MB | 6.96 / 6.77 | 12.3–12.4 / 13.5 |

结论：**H2D 提升约 1.8 倍**，达到 PCIe 4.0 x8 扣除协议开销后的实际上限（理论 15.75 GB/s）。

对设计的影响：

- 锁页环形缓冲**不用 `cudaHostAlloc`**，改为 THP 支撑的内存加 `cudaHostRegister`。
- THP 拿不拿得到取决于内存碎片（这台主机常驻占用较高；一次运行中只有 91% 的页拿到大页，中等大小的传输就明显变慢）。所以运行时要从 `/proc/self/smaps` 读出该内存区域的 `AnonHugePages`，检查覆盖率，不足时告警；可选用 hugetlbfs 预留大页（需要 root）保证稳定。
- 系统层面的替代方案是内核参数 `iommu=pt`（直通，需要 root 和重启），不作为项目依赖，但可以在报告里作为对照。
- 这一点同样适用于 OmniCore 传输层的 USB 缓冲（P5 跨层优化）。

### 2026-11-14 · P0 模型训练完成，结构冻结

**与原计划的偏差**：没有用 TorchSig 生成数据，改为自写 GPU 合成器（`model/curf_model/synth.py`）。原因：TorchSig 2.2 基于 CPU 逐样本生成、配置复杂，采样率和带宽语义还要额外对齐；自写生成器的参数直接对齐信道化输出，训练时在 GPU 上即时生成，数据量不受限。TorchSig 留作以后的交叉验证集。

**数据**（每个窗口模拟一个信道化输出）：fs = 240 kS/s，1024 点（4.27 ms），信号占用带宽 ≤ 120 kHz。10 类：noise、BPSK、QPSK、8PSK、16QAM、64QAM、GFSK、AM、FM、OFDM。随机化：每符号 3–24 个采样点、RRC 滚降 0.2–0.5、定时偏移、载波相位、频偏 ±10 kHz、50% 概率加短多径（最多 4 条回波，8 个采样点内）、SNR −4 至 24 dB（按全 240 kHz 带宽计）；窗口归一化为单位 RMS（即 GPU 前处理要做的事）。

**模型**（`model/curf_model/net.py`，结构已冻结）：ResNet-1D，输入 `[B, 2, 1024]`；stem conv k7 s2 2→32，stage1 32，stage2 64（s2，2 个块），stage3 128（s2，1 个块），全局平均池化 + 全连接。**13.6 万参数，每窗口 50.8 MFLOP**。引擎需要支持的算子：Conv1d（k = 1/3/7，stride 1/2）、BatchNorm（导出时折叠）、ReLU、Add、全局均值、Linear。

**训练**：3 万步，batch 512，AdamW + 余弦退火，bf16 自动混合精度，RTX 3050 Laptop 上用时 37 分钟（GPU 温度 86°C，SM 降频到约 1.4 GHz）。

**结果**（独立种子的 13.1 万个窗口，`docs/results/model_run1_metrics.json`）：

| 指标 | 结果 |
|---|---|
| 总准确率 | 89.5% |
| SNR ≥ 10 dB | 94.7% |
| SNR −4 / 0 / 6 / 12 dB 档 | 65.6% / 79.4% / 90.7% / 94.5% |
| FP16 与 FP32 的 top-1 一致率 | 99.95% |
| ONNX Runtime 与 PyTorch 的 logits 最大误差 | 0.0024 |

高 SNR 下的错误几乎全部来自 **16QAM 和 64QAM 互相混淆**（约 15%）和 QPSK / 8PSK 互混（约 4%）：每符号采样点多时，1024 点窗口里只有约 43 个符号，不足以区分高阶 QAM。其余类别都在 99% 以上。准确率不是项目卖点，这个水平够用。

**导出物**（CUDA 主机 `~/work/cuRF-Stream/model/artifacts/run1/export/`，git 忽略）：`amc_fp32.onnx`、`amc_fp16.onnx`、`weights.npz`、`reference.npz`（256 个固定输入窗口及 FP32 / FP16 的参考 logits，供引擎逐层对齐使用）。

**TensorRT 11.4 FP16 基线**（trtexec，batch 512，CUDA Graph）：275.8 qps，即**每秒 14.1 万个窗口**（有效约 7.2 TFLOPS），GPU 时间中位数 3.37 ms。对照第 4 节：全部 1024 个信道都要识别时需要每秒约 24 万个窗口，TensorRT 只能满足约 59%；10% 信道活跃时则绰绰有余。全负载场景需要更快的 kernel、INT8 或更窄的模型，这正是自研引擎要攻的方向。

### 2026-11-14 · P1 流水线骨架完成

**新增**：

- `HostRing`（`src/runtime/host_ring.cpp`）：固定槽位的 SPSC 环形缓冲，独立的 2 MB 对齐匿名映射 + `MADV_HUGEPAGE`，运行时从 `/proc/self/smaps` 读出大页覆盖率。三个单调计数器（write / submitted / release）支持多个槽同时在 GPU 上，按序释放；无锁快路径，条件变量只用来睡眠。
- `PsdStage`（`src/kernels/psd.cu`）：Welch 功率谱，Hann 窗 → 批量 cuFFT → 跨段累加并做 fftshift，满量程单音在 bin 中心读数为 A²。
- `run_pipeline`（`src/runtime/pipeline.cpp`）：生产者线程从源读入环形缓冲；消费者线程把每个槽派发到 `depth` 个 CUDA stream 之一，每个 stream 独占设备 buffer，按 H2D → 解包 → PSD → D2H 执行；完成后按序回收，用样点时间戳检测缺口。两种满载策略：背压（非实时源，保证零丢失）或丢弃（模拟硬件 FIFO 溢出）。NVTX 标注生产者填充、派发和回收。
- `curf_pipeline` 命令行基准、`tools/nsys_overlap.py`（从 nsys 轨迹算出拷贝与计算的重叠率）。
- 测试：CPU 12 个、CUDA 主机共 17 个，全部通过，零编译警告。新增环形缓冲的跨线程数据完整性测试，以及流水线端到端测试（PSD 峰值位置和功率与合成单音一致、背压下零丢失、时间限制可正常退出）。

**实测**（2 通道 packed12，block = 2^17 帧 = 786 KB，32 个槽，环形缓冲大页覆盖率 100%）：

| 配置 | 吞吐（实时倍数） | wire 速率 | 每块 GPU 时间 p50 | 丢样 / 缺口 |
|---|---|---|---|---|
| depth 1 | 11.7× | 4.30 GB/s | 0.158 ms | 0 / 0 |
| depth 2 | 20.5× | 7.54 GB/s | 0.176 ms | 0 / 0 |
| depth 3 | 23.0× | 8.46 GB/s | 0.220 ms | 0 / 0 |
| depth 4 | **24.5×** | **9.04 GB/s** | 0.270 ms | 0 / 0 |
| depth 3，环形缓冲不锁页 | 11.7× | 4.33 GB/s | 0.489 ms | 0 / 0 |
| depth 3，naive 解包 | 23.3× | 8.57 GB/s | 0.220 ms | 0 / 0 |
| **按真实速率 2×61.44 MSPS，丢弃策略，20 秒** | 1.00× | 0.37 GB/s | 0.160 ms | **0 / 0** |

按真实速率运行时，**从槽填满到结果回到主机的延迟 p50 0.24 ms、p99 0.32 ms、p999 0.36 ms、最大 0.43 ms**。（吞吐模式下的"延迟"是 3–6 ms，但那只是 32 个槽排满后的排队时间，不是流水线本身的延迟，不作为指标。）

**Nsight Systems**（`docs/results/profiles/p1_depth{1,3}.nsys-rep`，git 忽略）：

| | depth 1 | depth 3 |
|---|---|---|
| H2D 时间中与 kernel 重叠的比例 | 0% | **73.4%** |
| GPU 任意部件忙碌 | 75.4% | **96.0%** |
| H2D 引擎忙碌 / SM 忙碌 | 37.2% / 36.5% | 76.8% / 74.8% |

每块 kernel 时间（depth 3）：加窗 27.8 µs、cuFFT 23.0 µs、解包 18.0 µs、累加 12.9 µs，合计约 82 µs；H2D 约 76 µs。

**P1 验收**：

- 解包达到显存带宽的 98%（≥ 70%）：通过。
- H2D 达到 PCIe 实测上限的 90%：单独测 12.4 GB/s；流水线中 depth 1 时每块 0.071 ms，即 11.1 GB/s（89%）；depth 3 时与 kernel 争用显存，降到约 8.4 GB/s。基本达标。
- 时间线显示传输与计算重叠：通过（73%）。

**发现，留给后续阶段**：

1. depth 3 时 GPU 已经 96% 忙碌，**瓶颈是 GPU 自身**，H2D 和 kernel 两条线各占约 75%。
2. **加窗 kernel 是纯开销**（占 kernel 时间 34%）：它把 2 MB 读一遍、乘窗、再写回。融合进解包 kernel（或用 cuFFT 的 load callback）可省掉约 28 µs，kernel 时间降到约 54 µs，低于 H2D 的 76 µs，瓶颈就转移到 PCIe。这是优化日志的第一个融合案例，放在 P3 与信道化一起做。
3. 不锁页的环形缓冲吞吐减半（4.33 对 8.46 GB/s），每块 H2D 从 0.09 ms 涨到 0.34 ms，验证了锁页 + 大页这条路的价值。

### 2026-11-15 · P2 第一轮：引擎端到端可用，优化阶梯前 4 级

**新增**：

- **离线转换器** `model/convert.py`：ONNX → `amc.curf`（自定义二进制：张量表 + 算子表 + 64 B 对齐的权重 blob）。
  - Pass 1 BN 折叠（PyTorch 的 eval 导出已经折过一次，`--selftest` 用 `onnx.helper` 手搭含 BN 的图对照 onnxruntime 验证，误差 7.6e-6）。
  - Pass 2 epilogue 融合：`Conv [→BN] [→Add] [→Relu]` 合成一个 `conv` 算子；`ReduceMean → Gemm` 合成 `gap_linear`。残差 Add 融进**第二个**执行的 conv（第一版贪心地融进了 shortcut conv，导致另一支输入尚未计算，已修正）。
  - Pass 3 布局：激活改为 channels-last（NLC），隐式 GEMM 的 K 维（tap 优先、再通道）在内存中连续；权重变为 `[cout][kpad]` fp16，kpad 补齐到 32。
  - 融合后 12 个算子（11 个 conv + 头），50.79 MFLOP / 窗口。
  - 同时写出 `amc.golden`：256 个参考窗口的输入、标签、fp32 logits，以及前 32 个窗口**每个张量**的 fp32 激活（numpy 执行器算出，与 onnxruntime 最大差 3.8e-6）。
  - 注意：`reference.npz` 里 PyTorch 的 "fp32" logits 是在 GPU 上算的，cuDNN 默认开 TF32，与严格 fp32 差 2.4e-3，所以 golden 改用 onnxruntime CPU 结果。
- **模型加载与静态内存规划** `src/engine/model.cpp`（CPU 侧，Mac 上可测）：加载时校验拓扑序、形状、权重越界；按张量生命周期 [产生, 最后使用] 做按大小贪心的 offset 分配。batch 512 时 arena **48 MiB，不复用需 120 MiB（2.5×）**，48 MiB 正好是 3 个最大激活同时存活的下界。
- **conv1d kernel 注册表** `src/kernels/conv1d.cu`：20 个变体。
  - `naive`：每线程一个输出。
  - `smem`：64×32 分块，fp32 FMA。
  - `wmma.{f32,f16}.{unfused,fused}.{BM}x{BN}`：BM×BN ∈ {64,128}×{32,64,128}，每 warp 32×32。
  - A 矩阵按 8 个连续 k 做 16 B 向量化 im2col（cin % 8 == 0 时）。
  - 输入是 fp32 的第一层在加载时直接转 fp16，不需要单独的 cast kernel。
  - epilogue 经每 warp 的 smem 小块，以 16 B 向量写回，同时做 bias、残差、ReLU。
  - unfused 变体另起一个逐元素 kernel 做 bias、残差、ReLU。
- **引擎运行时** `src/engine/engine.cpp`：
  - batch 档位 1/8/32/128/512，每档独立选 kernel、独立捕获一个 CUDA Graph。
  - 每个 conv 用所有支持的变体实测，取最快的；变体可以按名字前缀限定，限定不到时逐级退回上一级前缀。
  - 调优结果可缓存到文件。
  - 热路径零 `cudaMalloc`。
- **测试**（共 21 个，全部通过）：
  - 内存规划不让同时存活的张量重叠。
  - 模型文件加载与校验。
  - **20 个变体逐层对齐 fp32 golden**：最差层相对误差 naive / smem 1.6e-3，wmma f32 累加 1.9–2.2e-3，f16 累加 3.0e-3。
  - 调优后的完整引擎（内存复用 + Graph）在 256 个窗口上与 fp32 的 top-1 一致率 ≥ 255/256，logit 最大差 0.011；batch 5 走 8 档 Graph 的结果与 256 档一致。
- `curf_bench_engine`：按阶梯逐级测吞吐（JSON），`--ops` 输出每层所选变体与耗时。数据见 `docs/results/p2_engine_ladder.jsonl`，融合后的图见 `docs/results/model_run1_fused_graph.json`。

**优化阶梯**（batch 512，CUDA Graph，每级在前缀内自动调优；笔记本 GPU 温度对结果影响约 15%，两列分别是冷机和连续跑热之后）：

| 步骤 | 窗口/秒（冷） | 窗口/秒（热） | 有效 TFLOPS（冷） | 相对上一级 |
|---|---|---|---|---|
| 1. naive | 1,277 | 1,160 | 0.07 | — |
| 2. smem 分块（fp32 FMA） | 16,503 | 13,607 | 0.84 | 12.9× |
| 3. WMMA Tensor Core，fp32 累加，epilogue 分离 | 79,939 | 68,769 | 4.06 | 4.8× |
| 4. + epilogue 融合（bias + 残差 + ReLU） | 117,866 | 97,349 | 5.99 | 1.47× |
| 4b. + fp16 累加 | **140,848** | 121,988 | **7.15** | 1.20× |
| 全注册表调优（最终全部选中 4b） | 138,450 | 121,974 | 7.03 | — |

**与 TensorRT 11.4 FP16 对比**（同一 ONNX，batch 512，trtexec + CUDA Graph，与本引擎背靠背交替跑两轮）：

| | 第 1 轮 | 第 2 轮（更热） |
|---|---|---|
| 本引擎 | 140.0k 窗口/秒 | 123.8k |
| TensorRT | 160.4k（中位 3.21 ms） | 153.0k（中位 3.35 ms） |
| 比值 | **0.87×** | 0.81× |

（P0 记录的 141k 是另一个热状态下的数，以本节背靠背数据为准。）

**逐层对比**（冷机，µs / batch 512；TensorRT 来自 `--dumpProfile --separateProfileRun`）：

| 层 | 本引擎 | TensorRT | 备注 |
|---|---|---|---|
| 输入重排 | 0 | 47 | 本引擎第一层直接读 fp32 NLC |
| stem（k7 s2，2→32） | 230 | 110 | K = 14 补到 32，浪费 56% 的 MMA；fp32 输入逐元素 gather |
| stage1 conv1 / conv2+res | 294 / 343 | 216 / 287 | |
| stage2.0 shortcut / conv1 / conv2+res | 143 / 233 / 382 | 236（含 Add）/ 196 / 253 | TRT 把 Add 融进了 shortcut |
| stage2.1 conv1 / conv2+res | 338 / 382 | 252 / 302 | |
| stage3 shortcut / conv1 / conv2+res | 151 / 319 / 604 | 150 / 225 / 330（不含 Add） | |
| 头（GAP + 线性） | ≈ 217 | 570 | TRT 为 Myelin 子图做了两次 reformat，各 188 µs |
| **合计** | **≈ 3,640** | **≈ 3,180** | |

**差距来源**：

1. **conv kernel 本身慢约 1.34×**：11 个 conv 合计 3,418 对 2,557 µs，是主因。TRT 的 stage3 conv2 达到 19.5 TFLOPS（远超本 GPU fp32 累加的峰值，说明它也用 fp16 累加），本引擎同一层 10.7 TFLOPS。本引擎 WMMA kernel 的短板有四点：
   - 单缓冲，全局加载延迟没有被隐藏，每个 BK = 32 要做两次 `__syncthreads`；
   - im2col 每个 16 B 块都要做整数除法求 (b, l, tap, ci)；
   - 用 `wmma::load_matrix_sync` 配合 padding，而不是 `ldmatrix` 加 swizzle；
   - BK 小。
   这正是阶梯第 5 级（`cp.async` 多级流水 + `mma.sync` / `ldmatrix`）要解决的问题。
2. **stem 慢 2×**：它是访存型（读 4 MB fp32，写 16.8 MB），用 Tensor Core 不划算。应该写成专用的直接卷积，或者在 P3 与信道化融合，从源头上消掉。
3. **本引擎的头比 TRT 快约 350 µs**（不做 reformat），抵消了一部分差距；如果把 GAP 融进最后一个 conv 的 epilogue，还能省掉约 217 µs（读写 16.8 MB）。

**小 batch**：batch 1 的延迟约 100 µs，Graph 与逐个 launch 相比只省约 5 µs（batch 32 时省 23 µs）。延迟主要是 12 个 kernel 本身：M = 512 行时只有 4 个 128 行的块，16 个 SM 大多空闲。改进方向有三个：split-K、更小的 tile、整网 persistent kernel。

**P2 剩余**：

- 阶梯第 5 级（`cp.async` + `mma.sync` + `ldmatrix`）；
- stem 专用 kernel；
- GAP 融进最后一个 conv；
- Nsight Compute 逐级数据（Tensor Core 利用率、显存带宽、瓶颈变化）；
- cuDNN 与 CUTLASS 基线；
- 小 batch 延迟优化。

### 2026-11-15 · P2 阶梯第 5 级：cp.async + ldmatrix + mma.sync，超过 TensorRT

**新增**：

- `src/kernels/conv1d_mma.cu`：手写 PTX 的隐式 GEMM 卷积，26 个变体，配置为 `mma.{f32,f16}.{BM}x{BN}.w{WM}.s{级数}`。要求输入为 fp16 且 cin % 8 == 0；fp32 输入的 stem 仍走 WMMA。主要改动有四点：
  - `cp.async` 多级环形缓冲，2–4 级，每个 k-tile 只需一次 `__syncthreads`；
  - im2col 的地址计算提到主循环外：每个线程固定负责几行和一列 16 B，(batch, 位置) 只算一次，(tap, 通道) 按增量推进，主循环里没有整数除法；
  - 用 `ldmatrix.x4` 从 80 B 行距（无 bank 冲突）的 smem 取 fragment，`mma.sync.m16n8k16` 的 warp tile 最大到 64×32；
  - epilogue 经 smem 中转后用 16 B 合并读写。
- `curf_bench_engine --profile`：调优后在 `cudaProfilerStart/Stop` 之间跑一次，配合 `ncu --profile-from-start off` 只抓推理本身的 kernel。
- `tools/ncu_summary.py`：把 ncu 报告压成每个 kernel 一行（时间、SM / DRAM / L1 / L2 / Tensor Core 利用率、DRAM 流量、occupancy、寄存器、smem、主要 stall 原因）。数据见 `docs/results/p2_ncu.md`。
- 修正：变体前缀逐级退回时到不了空前缀（`mma.` 之后就停了），导致 stem 找不到可用变体。
- 测试：新增的 26 个变体全部逐层对齐 golden，最差层误差 f32 累加 1.7e-3、f16 累加 4.2e-3。21 个测试全部通过。

**第一版反而更慢，Nsight Compute 定位到原因**（batch 512，`mma.f16` 128.5k 对 `wmma.f16` 143.8k 窗口/秒）：

| 现象（ncu） | 原因 | 改法 |
|---|---|---|
| achieved occupancy 只有 16–32%（WMMA 48–65%） | 3 级环形缓冲每块占 37–45 KB smem，每个 SM 驻留的块数受 smem 限制 | 增加 2 级配置，由调优挑选 |
| L1 吞吐 70–96%，L2 吞吐 65–86%（WMMA 约 45%） | `cp.async.cg` 绕过 L1；k = 3 的卷积每个输入行要被 3 个 tap 读到，每次都回 L2 | 激活用 `cp.async.ca`（经 L1） |
| 带残差的层 long-scoreboard stall 达 62%，stage1 conv2 慢 60% | epilogue 直接从寄存器写，每线程 4 B，残差读也是零散的 | 两遍 epilogue：先 bias（和 ReLU）写进 smem，再 16 B 合并读残差、写结果 |

三项一起改完后，同一测试从 128.5k 升到 **177.5k**（1.38×）。

**更新后的阶梯**（batch 512，CUDA Graph，前缀内自动调优）：

| 步骤 | 窗口/秒（冷） | 窗口/秒（热） | 有效 TFLOPS（冷） | batch 1 延迟 |
|---|---|---|---|---|
| 4b. WMMA + 融合 epilogue + fp16 累加 | 143,680 | 121,980 | 7.30 | 91 µs |
| 5. `mma.sync`，fp32 累加 | 160,374 | 132,031 | 8.15 | 54 µs |
| **5. `mma.sync`，fp16 累加** | **177,472** | 155,539 | **9.01** | **48 µs** |

第 5 级相对第 4b 级：大 batch 提速 1.24–1.28×，batch 1 延迟降低 47%。原因是小 tile 配置加上没有分离的 epilogue，GPU 填得更满。"全注册表调优"的结果略低于只在 `mma.f16` 里调（171k 对 177k），因为单算子孤立计时有噪声，调优时 L2 也是热的，与在 Graph 里连跑的情况不完全一致。这一点后续把 tune_iters 调大、按整图复测来改进。

**与 TensorRT 11.4 FP16 对比**（背靠背两轮，同一 ONNX）：

| | 第 1 轮 | 第 2 轮 |
|---|---|---|
| 本引擎（batch 512） | 177.6k 窗口/秒 | 175.5k |
| TensorRT（batch 512） | 163.2k（中位 3.12 ms） | 155.1k（中位 3.31 ms） |
| 比值 | **1.09×** | **1.13×** |

小 batch：batch 1 本引擎 48 µs、TensorRT 40 µs（中位）；batch 8 本引擎 64 µs、TensorRT 79 µs。

**逐层看，本引擎的 conv 本身仍比 TRT 慢约 15%**（合计 2,946 对 2,557 µs）。总体领先，靠的是头部（113 µs 对 570 µs，TRT 为 Myelin 子图做了两次 reformat）和省掉的输入重排（47 µs）。剩余差距的分布：

- stem 232 对 157 µs（含 TRT 的输入重排）；
- stage3 conv2 510 对 330 µs。ncu 显示这一层 Tensor Core 利用率 63%、DRAM 54%，是 tile 形状 / 流水线还不够好，而不是带宽的问题；
- 其余层差距在 ±10% 以内，有的反而更快：stage1 conv1 202 对 216 µs，stage2.0 conv1 189 对 196 µs。

**ncu 结论：瓶颈已经从"kernel 写得不好"变成了显存带宽**。大部分 conv 的 DRAM 吞吐在 67–91%（锁基频下），Tensor Core 利用率 19–63%。单个 kernel 再优化收益有限，下一步的大头是**减少显存流量**：

- GAP 融进最后一个 conv：省一次 17 MB 的读写，约 110 µs，占 3.5%；
- stem 专用 kernel，或在 P3 与信道化融合：约 230 µs，占 8%；
- 残差块内的两个 conv 融合，中间激活留在 smem：每个块省一次激活的写和读，是最大的一项，也最复杂；
- 激活降到 INT8 / FP8：sm_86 不支持 FP8，INT8 需要校准。

**P2 剩余**：

- 跨层融合（GAP、stem、残差块）；
- cuDNN 与 CUTLASS 基线；
- 用整图复测改进调优；
- batch 1 再压 8 µs 追平 TRT（12 个 kernel 本身的 launch 和尾部效应，可以考虑整网 persistent kernel）。

### 2026-11-15 · P2 阶梯第 6–8 级：跨层融合，由实测决定是否融合

**新增**：

- **第 6 级，GAP 融进最后一层的 epilogue**：`ConvParams::pool` / `BlockParams::pool` 非空时不写输出，按样本对各通道求和，原子加到 fp32 的 `[batch][C]` 缓冲。Graph 开头插一个 memset 节点清零。头部变成 `linear_from_pool`，从 109 µs 降到 40 µs。最后一层 17 MB 的输出张量不再落地。
- **第 7 级，stem 专用直接卷积**（`direct` 变体，cin ≤ 8 时可用）：输入段和转置后的权重放在 smem，fp32 输入直接读。
- **第 8 级，整个 BasicBlock 一个 kernel**（`src/kernels/block_fused.cu`，`block.{f16,f32}.l{64,128}.{cin}x{cout}s{S}`，共 16 个变体）。每个 CTA 负责一个样本的 BL 个输出位置、全部通道，流程如下：
  1. x 段（带 halo）一次 `cp.async` 进 smem；
  2. 1×1 shortcut 先算，结果直接累加进 conv2 的累加器：shortcut 只是 conv2 GEMM 多出的 K，bsc 合进 bias；
  3. conv1 算 BL + 2 行（conv2 的 halo），bias + ReLU 后以 fp16 写回 smem，覆盖 x。越界位置写 0，这就是 conv2 的 zero padding；
  4. conv2 的 A 操作数直接从 smem 取。隐式 im2col 靠每个 lane 给 `ldmatrix` 的行地址实现；
  5. 权重走 2 级 `cp.async` 环（所有 CTA 都要读，常驻 L2）。
  每个残差块的显存流量从约 5 个激活张量降到 2 个。全网每窗口的激活流量从约 840 KB 降到约 260 KB。
- **引擎的融合 pass**（`src/engine/engine.cpp` 重写）：
  - 按图模式找候选，规则是 conv1 → conv2(+残差) [+ 1×1 shortcut]，中间张量只有一个消费者；
  - 执行计划从"算子列表"改为"每个 batch 档位一份 step 列表"；
  - 内存规划支持"不落地"张量：先按全部落地的方案调优，确定融合后再按实际需要的张量重新规划 arena。
- 新增 `EngineConfig::{fuse_blocks, fuse_gap, block_prefix, exclude_prefixes}`、`Engine::fusion_decisions()`；bench 的阶梯步骤写成 `前缀+gap+stem+block`。
- **测试**（22 个，全部通过）：每种 block 变体（f16 / f32 × BL 64 / 128）× GAP 开 / 关，都与 golden 逐个对比落地张量。融合后 logits 误差反而更小（f16 1.5e-3、f32 1.0e-3），因为中间少了几次 fp16 舍入。

**过程中 ncu 定位并修掉的问题**（每一项都是"融合了反而更慢"）：

| 现象 | 原因 | 改法 | 效果 |
|---|---|---|---|
| stem 直接卷积比 WMMA 还慢（314 对 315 µs，ncu 锁频） | 每线程算 1 个位置 × 8 通道，每 8 次 FMA 要 3 次 smem 读，L1/smem 吞吐 94% | 每线程算 4 个位置 × 8 通道，一对 float4 权重喂 32 次 FMA；位置交错排列，让 warp 的 16 B 写覆盖整行 | 202 µs，DRAM 53%，基准测试中 213 → 172 µs |
| GAP 融合后最后一层多出 190 µs | ① 池化循环里每行一次 64 位整数除法求样本号 ② 每列串行求和 ③ 改用 smem float 原子加后，16 路争用 | ① 每线程只除一次，之后按样本边界增量推进 ② ③ 整块属于同一样本时（通常如此）走快速路径：16 B 向量读 8 列，寄存器部分和 + warp shuffle 归约 + 每 warp 部分和经 smem 汇总，每列一次全局原子加 | 带池化 1037 µs，不带 1043 µs，GAP 融合变成零开销 |
| 融合块 v1 中 64 / 128 通道的块比不融合慢（576 对 551 µs、1020 对 900 µs） | barrier + wait stall 占 40–55%：warp 任务只有 16×64，每个 k-tile 16 次 MMA、8 个独立累加器；conv1 的 halo 行使 10 个任务分给 8 个 warp | v2：BL 可选 128，warp 任务改为 32×32（MI = 2、NI = 4）；halo 行拆成小任务，接着主任务轮转分配 | 全部块变快，见下表 |

**融合是否划算由实测决定，而且要放在上下文里测**：

- 第一版用"融合 kernel 单独计时 vs 各 conv 单独计时之和"来决定。结果 GPU 一热，判断就翻转。连跑三轮时，stage3 单测是不融合快（951 对 1035 µs），可端到端反而是融合快。
- 原因有两个：
  - 单测时，不融合的 conv 在同一缓冲上反复启动，输入有一部分还留在 L2 里。真实序列里，这份数据刚被上一个 kernel 写出，大部分已经被挤出 L2。
  - 笔记本 GPU 降频时，SM 时钟下降而显存时钟不变。融合 kernel 偏计算密集（用更多计算换显存流量），降频时优势随之缩小。
- 最终做法：每个 batch 档位、每个块，轮流切换"融合 / 不融合"，测整网时间，两种方案交替测 3 轮、各取最好值。

| 块（batch 512） | 冷机：整网 µs，融合 / 不融合 | 热机：整网 µs，融合 / 不融合 |
|---|---|---|
| stage1（32→32） | 2179 / 2536 | 2810 / 3042 |
| stage2.0（32→64，s2） | 2366 / 2600 | 2833 / 3003 |
| stage2.1（64→64） | 2346 / 2415 | 2834 / 2830（不融合） |
| stage3（64→128，s2，含 GAP） | 2328 / 2369 | 2830 / 2745（不融合） |

batch 1 时只有几个 CTA，stage3 一般选择不融合；batch 8 时四个块全部融合。

**更新后的阶梯**（batch 512。热机列取自同一份 `p2_engine_ladder.jsonl`，阶梯连跑时 GPU 已经发热；冷机列是单独跑的）：

| 步骤 | 窗口/秒（冷） | 窗口/秒（热） | 相对第 5 级（热） | kernel 数 |
|---|---|---|---|---|
| 5. `mma.f16` | 179,088 | 155,767 | — | 12 |
| 6. + GAP 融合 | — | 158,776 | +1.9% | 12 |
| 7. + stem 直接卷积 | — | 164,459 | +5.6% | 12 |
| **8. + BasicBlock 融合（实测决定）** | **224,044** | 187,969 | **+20.7%** | 6（冷）/ 9（热） |

冷机时第 8 级相对第 5 级提升 25%，有效算力 11.4 TFLOPS。arena 显示为 48 MiB：所有档位的计划按最大 batch 取并集，batch 1 不融合时用到的中间张量也算在内。按档位分别规划可以降到 32 MiB，留作后续改进。

**与 TensorRT 11.4 FP16 对比**（背靠背三轮）：

| | 第 1 轮 | 第 2 轮 | 第 3 轮 |
|---|---|---|---|
| 本引擎 batch 512 | 200.6k | 188.0k | 187.9k |
| TensorRT batch 512 | 153.9k | 158.2k | 154.2k |
| **比值** | **1.30×** | **1.19×** | **1.22×** |
| batch 1 延迟（本引擎 / TRT） | 43 / 38 µs | 48 / 42 µs | 48 / 42 µs |
| batch 8 延迟（本引擎 / TRT） | 61 / 73 µs | 73 / 83 µs | 80 / 83 µs |

单独一轮最好成绩是 224.0k，冷机，四个块全部融合，相对 TRT 约 1.4×。batch 1 仍比 TRT 慢约 12%。

**阶段结论**：从 naive 到第 8 级，batch 512 吞吐提升 **175×**（冷机 1,277 → 224,044 窗口/秒），对 TensorRT FP16 稳定领先 1.2–1.3×。全负载每秒约需 24 万个窗口（第 4 节），冷机已达 93%。

**剩余（P2 收尾或后续）**：

- 64 / 128 通道融合块的计算效率只有约 11 TFLOPS，降频时会输给不融合方案。可以试 3 级权重环（权重 tile 改 swizzle 以节省 smem）、BK = 64，以及更大的 warp tile。
- arena 按档位分别规划。
- batch 1：persistent 整网 kernel，或者 stem + stage1 再融合。
- cuDNN 与 CUTLASS 基线。

### 2026-11-15 · P2 基线：cuDNN 9.21 与 CUTLASS 4.5

**接入方式**：两个库都作为普通的 conv 变体注册进同一个注册表（`cudnn`、`cutlass.{f32,f16}.{TBM}x{TBN}`），和自研 kernel 走同一套自动调优、逐层对齐测试和 bench。激活是 NLC 布局，即 H = 1 的 NHWC；权重 `[cout][k][cin]` 就是 KRSC。所以两个库直接在引擎自己的 buffer 上跑，没有任何拷贝或重排。每层都用库自带的融合 epilogue（bias + 残差 + ReLU），不给库额外的 kernel 开销：

- **cuDNN**（`src/kernels/baseline_cudnn.cpp`，旧版 convolution API）：
  - `cudnnConvolutionBiasActivationForward`：fp16 张量、fp32 计算、tensor-op math，残差以 z 输入（alpha2 = 1）；
  - 每个形状用 `cudnnFindConvolutionForwardAlgorithmEx` 让 cuDNN 自己挑算法；
  - 无 ReLU 的层按 cuDNN 的规定只能用 `IMPLICIT_PRECOMP_GEMM`；
  - stem 的 fp32 输入多付一个 cast kernel；
  - batch 1 时 stem 的 Find 全部失败（INTERNAL_ERROR / NOT_SUPPORTED），回退到 `IMPLICIT_PRECOMP_GEMM`。
- **CUTLASS**（`src/kernels/baseline_cutlass.cu`）：
  - 头文件来自 vllm 环境里 flashinfer 自带的 CUTLASS 4.5.0，拷到 `~/opt/cutlass-4.5.0`，没有联网下载；
  - SM80 `Conv2dFprop`，多级 `cp.async` 主循环，3 级，4 种 tile × fp16 / fp32 累加；
  - 无残差的层用 `LinearCombination[Relu]`，bias 作为 stride 为 0 的广播 C 张量；
  - 有残差的层用 `Conv2dFpropWithBroadcast` + `LinearCombinationResidualBlock<Identity, plus, ReLu>`，bias 作为广播向量；
  - 不支持 stem（cin = 2 达不到 8 元素访问对齐，输入又是 fp32），stem 由引擎自己的 WMMA kernel 跑。
- 两个基线都逐层对齐 golden：最差层误差 cuDNN 2.4e-3，CUTLASS 2.6e-3（f32 累加）/ 3.2e-3（f16 累加）。
- bench 默认把库 kernel 排除在自研配置的调优之外，加 `+lib` 才允许混用。修正这一点之前，`mma.f16` 和"全部自研"两项的 stem 都被调优选成了 cuDNN，自研的数字被库 kernel 悄悄污染了。
- 测试 22 个，全部通过。数据见 `docs/results/p2_baselines.jsonl`。

**端到端**（batch 512，CUDA Graph，同一轮背靠背；GPU 已温热，绝对值偏低）：

| 配置 | 窗口/秒 | 自研融合版领先 |
|---|---|---|
| **自研，完整融合**（第 8 级） | **204.4k** | — |
| CUTLASS（stem 用自研 WMMA） | 178.5k | 1.15× |
| 自研 `mma.f16`，逐层不融合 | 176.9k | 1.16× |
| TensorRT 11.4 FP16 | 154.5k | 1.32× |
| cuDNN 9.21 | 113.5k | 1.80× |

小 batch：

| | batch 1 | batch 8 |
|---|---|---|
| 自研融合版 | 42.1 µs | 59.2 µs |
| CUTLASS | 48.3 µs | 66.5 µs |
| cuDNN | 53.6 µs | 124.5 µs |
| TensorRT | 42.0 µs | 81.8 µs |

**逐层**（µs，batch 512，各自调优选出的最快变体）：

| 层 | 自研 `mma.f16` | CUTLASS | cuDNN | 胜者 |
|---|---|---|---|---|
| stem（2→32，k7 s2，fp32 输入） | 217（WMMA）/ 177（direct） | 不支持 | **167**（含 cast） | cuDNN |
| stage1 conv1（32→32） | **195** | 209 | 231 | 自研 |
| stage1 conv2 + 残差 | **291** | 343 | 317 | 自研 |
| stage2.0 shortcut（1×1 s2） | **142** | 187 | 915 | 自研 |
| stage2.0 conv1（s2） | **189** | **189** | 374 | 持平 |
| stage2.0 conv2 + 残差 | 304 | **288** | 341 | CUTLASS |
| stage2.1 conv1 | 244 | **232** | 314 | CUTLASS |
| stage2.1 conv2 + 残差 | 304 | **288** | 341 | CUTLASS |
| stage3 shortcut（1×1 s2） | **143** | 145 | 150 | 持平 |
| stage3 conv1（s2，64→128） | 223 | **214** | 343 | CUTLASS |
| stage3 conv2 + 残差（128→128，K = 384） | 406 | **335** | 777 | CUTLASS（−18%） |
| 除 stem 外的 10 层合计 | 2,441 | 2,430 | 3,513 | 持平 |

**结论**：

1. **单层 GEMM 质量与 CUTLASS 持平**（合计 2,441 对 2,430 µs）。
   - CUTLASS 在 N、K 大的层上占优，stage3 conv2 快 18%。它的 128×128 tile 用 64×64 的 warp tile、3 级流水，单个 fragment 的复用更高；自研最大只有 64×32。
   - 自研在 N = 32 和带残差的小层（stage1）、以及 1×1 stride 2 的 shortcut 上占优：smem 中转的合并 epilogue，加上 im2col 地址只算一次。
2. **自研引擎整体领先 CUTLASS 15%，来自库做不到的部分**：整个残差块融成一个 kernel、GAP 融合、按档位实测决定是否融合。
3. **cuDNN 在这个形状上明显吃亏**：1×1 stride 2 的 shortcut 要 915 µs，是自研的 6.4 倍；带残差的层也普遍更慢。stem 反而是 cuDNN 最快（167 µs，含 cast），说明自研的 direct stem（DRAM 53%）还有空间。
4. 允许自研和库 kernel 混用（`+lib`）在这一轮反而更低（189.3k）。它排在最后跑、GPU 最热，融合决策也因此变了，说明单轮比较有噪声；数据如实记录。

**对自研 kernel 的改进线索**：

- `mma` 加 64×64 warp tile（MI = 4、NI = 8），目标是 stage3 conv2 追平 CUTLASS 的 335 µs；
- direct stem 改成先把整段输入读入、再按位置分块，目标是追平或超过 cuDNN 的 167 µs。

### 2026-11-15 · 补短板：stem 超过 cuDNN，stage3 conv2 追到 CUTLASS 的 2% 以内

**stem**（`direct` 系列）：把 tile 长度、每线程位置数、每 SM 最少驻留块数（即寄存器上限）做成模板参数，注册 5 个配置由调优挑选。第一版（TL 256、Q 4、69 个寄存器、每 SM 3 个块）的问题是：每个块的读 → 算 → 写是串行的，同时驻留的块又太少，互相遮盖不了，DRAM 只用到 53%。调优选中 `direct.t512q4b4`（TL 512、每 SM 至少 4 个块）：**177 → 155 µs**，快于 cuDNN 的 163 µs（含 cast）。

**stage3 conv2**：按 ncu 逐层定位，分三步追：

| 步骤 | ncu 发现 | 改法 | stage3 conv2（ncu 锁基频 / 实测） |
|---|---|---|---|
| 0 | 最好的 64×32 warp、2 级填充版本：TC 62%，CUTLASS 85% | — | 518 / 406 µs |
| 1 | 64×64 warp tile 反而更慢：3 级填充环占 61 KB，每 SM 只放得下 1 个 CTA（occupancy 8%） | 改用 XOR swizzle 的无填充布局：第 r 行的 16 B 块 c 存在 c ^ ((r >> 1) & 3) 处，`ldmatrix` 和 `cp.async` 都没有 bank 冲突；3 级 128×128 环从 61 KB 降到 48 KB，与 CUTLASS 相同 | 877 µs，仍是每 SM 1 个 CTA |
| 2 | 动态 smem 和 CUTLASS 一样都是 48 KB，CUTLASS 能放 2 个 CTA，本 kernel 不能 | 融合 GAP 用的 2 KB 静态 `__shared__` 数组，不池化时也照样分配：48 + 2 + 1（系统保留）KB × 2 超过 100 KB。改为借用已经空闲的动态 smem | 540 µs，每 SM 2 个 CTA |
| 3 | 同样的 occupancy 下 TC 60%，CUTLASS 85% | 寄存器双缓冲 fragment：本步 k16 的 MMA 运行时，下一步的 `ldmatrix` 已经在途；barrier 之后立刻取下一个 k-tile 的第一组 fragment | 64×32 warp 版本 414 µs、TC 78%；实测 **339 µs**，CUTLASS 332 µs |

**副作用及处理**：fragment 双缓冲让只有一个 k-tile 的 1×1 shortcut（K = 32）从 141 µs 变慢到 186 µs。现在两种主循环并存：swizzle 变体（`.z`）用双缓冲，填充变体保留原来的循环，由调优按层挑选。shortcut 回到 141 µs。

**修正后逐层对比**（batch 512，µs）：

| 层 | 自研 | CUTLASS | cuDNN |
|---|---|---|---|
| stem | **155** | 不支持 | 163 |
| stage1 conv1 / conv2 + 残差 | **192** / **292** | 205 / 336 | 221 / 303 |
| stage2.0 shortcut / conv1 / conv2 + 残差 | **141** / 187 / 289 | 187 / 188 / 288 | 893 / 363 / 329 |
| stage2.1 conv1 / conv2 + 残差 | 229 / 289 | 226 / 288 | 291 / 329 |
| stage3 shortcut / conv1 / conv2 + 残差 | **143** / 215 / 339 | 144 / 215 / **332** | 146 / 338 / 760 |
| 除 stem 外 10 层合计 | **2,317** | 2,408 | 3,971 |

每一层都快于库或与库差距在 2% 以内。10 层合计比 CUTLASS 快 3.8%（之前是持平）。不融合的端到端：自研 192.9k，CUTLASS 182.0k。数据见 `docs/results/p2_baselines_after_fix.jsonl`。

**端到端，与 TensorRT 背靠背两轮**：

| | 第 1 轮 | 第 2 轮（热机） |
|---|---|---|
| 自研完整融合，batch 512 | **230.2k** 窗口/秒（11.7 TFLOPS） | 203.3k |
| TensorRT，batch 512 | 163.4k | 155.9k |
| 比值 | **1.41×** | 1.30× |
| batch 1 / batch 8 延迟 | **39.8** / 55.9 µs | 41.0 / 63.3 µs |

batch 1 现在也快过 TensorRT（之前测得 38–42 µs）。冷机 230k 已经是全负载需求（每秒约 24 万窗口）的 96%。测试 22 个，全部通过。

**仍可改进**：融合块 kernel 还在用填充布局和背靠背的 fragment 加载，没有吃到这一轮的两项改进（swizzle 省 smem、fragment 双缓冲）。把它们移植过去，应该能让 64 / 128 通道的块在热机时也稳定优于不融合方案。

### 2026-11-15 · 把 swizzle + fragment 双缓冲移植到融合块：没有带来提升，瓶颈另有所在

**做了什么**（`src/kernels/block_fused.cu`）：

- 新增 `PIPE` 变体（`block.{f16,f32}.l{64,128}p.*`）：
  - 权重环改为 swizzle 无填充布局，并从 2 级加深到 3 级（`cp.async.wait_group 1`）；
  - warp 在一个 k-tile 内要做若干 (任务, k16) 步，下一步的 fragment 在本步 MMA 之前发出（步号在编译期展开，fragment 缓冲留在寄存器里）。
- 原版本保留，两者由调优挑选。swizzle 辅助函数 `sidx` 和编译期循环 `static_for` 移入 `mma_common.cuh`。
- 两种版本都把隐式 im2col 的地址计算提出了逐 fragment 的路径：CIN、COUT 都是 32 的倍数，一个 k-tile 不会跨 tap，所以 tap 和通道基址是每个 tile 的常量。
- bench 新增 `--block-prefix`。16 个块变体全部与 golden 对齐，22 个测试全部通过。

**结果**（ncu 锁基频，batch 512，µs）：

| 块 | 原版 | PIPE 版 | 原版 + 地址外提 |
|---|---|---|---|
| stage1（32→32，BL 64） | 524 | 537 | 514 |
| stage2.0（32→64，s2） | 525 | 601 | 531 |
| stage2.1（64→64） | 636 | 646 | 630 |

端到端两轮：225.1k 对 TRT 153.4k（**1.47×**），热机 203.6k 对 154.2k（1.32×），与上一轮持平。按实测决定融合时，stage2.1 和 stage3 仍然选择不融合。PIPE 版只在一轮中被 stage1 选中过一次。

**为什么没用**（要记下来的教训）：

- 上一节把融合块的瓶颈归因为"barrier + cp.async 等待"，这是看错了指标。ncu 的 `stall_wait` 指的是固定延迟的指令依赖等待，不是 `cp.async.wait_group`。
- 三项改动之后 `wait` 一直稳定在约 25%，说明它和权重加载延迟、地址计算都无关。
- 更可能的原因是 MMA 依赖链太短：融合块的 warp 任务是 32×32，每步只有 8 个独立累加器。而 occupancy 受寄存器限制（约 122 个寄存器，每 SM 16 个 warp），没有足够多的 warp 来遮盖延迟。
- 对照：独立 conv 的 `mma` kernel 用 64×32 的 warp tile，每步 16 个独立累加器，同一层 TC 能到 78%；融合块只有 34–57%。

**下一步若要继续**：

- 64 通道以上的块，warp 任务改为 32×64，每步 16 个独立累加器。为了任务均衡需要 BL = 128，代价是每 SM 只放得下 1 个 CTA，要看 ILP 的增益能否抵消。
- 或者 halo 的 conv1 改用小 tile 单独算，减少寄存器压力。

现阶段收益不确定，暂停在这里。

### 2026-11-15 · P3 第一步：多相信道化

**参数**（第 4 节）：512 信道 × 120 kHz，2 倍过采样（D = M/2 = 256），每信道 240 kS/s，与分类器训练数据一致。原型滤波器为 Kaiser 窗 sinc：L = M·P = 512 × 8 = 4096 个抽头，−6 dB 边缘在 0.75 倍信道间隔，β = 8，直流增益为 1。

**定义与实现**（`include/curf/channelizer.hpp`、`src/kernels/channelizer.cu`）：

- 定义：`y_k[n] = Σ_l h[l]·x[nD−l]·e^{−j2πk(nD−l)/M}`，即先搬到基带、再低通、再抽取。
- 多相形式：`u[m] = Σ_p h[m+pM]·x[nD−m−pM]`，再做 M 点逆 DFT。2 倍过采样带来的 `(−1)^{kn}` 相位，在奇数帧把 u 循环移位 M/2 来实现，不需要额外乘法。
- fold kernel：每个线程负责一个相位 m，P 个抽头放在寄存器里，连续处理多帧。
- 批量 cuFFT：M 点逆变换，原地进行。
- 每路保留最近 L 个输入样点在显存里作为历史，块与块之间是一条连续的流。
- 输入有两种：平面 cf32（解包 kernel 的输出），或**直接读 packed12 / signed8 线格式**——解包融进 FIR，不落地 fp32 中间结果，这正是 P1 的结论"不要物化 fp32"。
- fold 有两个变体：
  - `direct`：每个 tap 都从全局内存读并解码；
  - `smem`：块所需的输入段先一次性解码进 smem（8 帧共 5888 个样点，47 KB），再只做 FMA。
- CPU 侧有原型滤波器设计（GPU 与参考共用）和按定义直接求值的参考实现 `ref::channelize`。

**测试**（共 26 个，全部通过）：

- 原型滤波器：直流增益为 1，系数对称。
- 两种 fold 变体连续跑 3 个块，与直接定义的最大相对误差 < 2e-5，块边界连续。
- 单音放在信道 3 / 200 / 511 的中心：幅度误差 < 0.1%；±2 信道 < −70 dB；±100 信道 < −90 dB。
- 直接读线格式，与"解包 + 平面"两条路径的结果逐位相同（packed12 和 signed8 各测一次，第二次调用覆盖历史缓冲路径）。

**吞吐**（2 路 × 2^17 样点每块，512 信道，P = 8；实时需求为每块 2.13 ms）：

| 路径 | 每块时间 | 每路 MS/s | 实时倍数 |
|---|---|---|---|
| PyTorch（unfold + 乘 + reshape-sum + `torch.fft.ifft`，平面 complex64 输入） | 1.035 ms | 127 | 2.1× |
| 解包 kernel + 平面信道化 | 0.126 ms | 1038 | 16.9× |
| 线格式融合，`direct` fold | 0.124 ms | 1054 | 17.2× |
| **线格式融合，`smem` fold** | **0.111 ms** | **1182** | **19.2×** |

- 相对 PyTorch 快 **9.3×**：PyTorch 的 unfold 加乘法会物化一个 `[S, F, L]` 的 32 MB 中间张量。
- 原计划的 cupyx 对比还没做：主机上没有 cupy，安装需要下载 `cupy-cuda12x`，待确认。

**ncu 分解与 roofline**（锁基频）：

| kernel | 时间 | 判断 |
|---|---|---|
| fold（`direct`） | 95 µs | SM 73%、DRAM 31%，是指令吞吐受限：每个样点被 L/D = 16 帧各解码一遍，每个 tap 还有 64 位下标运算。`smem` 变体把解码降到约 2.9 次每样点，快了 11%。 |
| cuFFT（`vector_fft<512>`） | 48 µs | DRAM 87%，带宽受限：原地读写各 4 MB。 |
| 历史缓冲更新 | 3 µs | — |

- 每块的理论下限：
  - 计算 40 MFLOP（fold 16.8 + FFT 23.6），约 6 µs；
  - 最少显存流量 4.8 MB（线格式输入 0.8 MB + 信道输出 4 MB），约 27 µs。
  - 所以下限约 30 µs，实测 111 µs，差 3.7×。
- 剩下的大头是 fold 输出再被 cuFFT 读写一遍，多了 8 MB 流量。把 512 点 FFT 也放进 fold kernel（在 smem 里做，或用 cuFFTDx / cuFFT 的 load callback），流量可以从约 12 MB 降到约 4 MB，预计再快约 2×。不过目前信道化只占实时预算的 5%，这一项列为余力项。

**P3 剩余**：

- 能量检测 + 活跃信道压缩（手写与 CUB 对比）；
- 切窗与归一化，融进推理输入（第一层直接从信道化输出 gather）；
- 接入流水线 + 截止时间驱动的动态 batching；
- CPU 基线（ONNX Runtime CPU / oneDNN，AVX-512）；
- 负载旋钮曲线；
- 端到端延迟分布。

### 2026-11-15 · P3 第二步：端到端流水线、负载旋钮、延迟与端到端准确率

**新增**：

- **窗口阶段**（`include/curf/cuda/windows.hpp`、`src/kernels/windows.cu`），输入是一个窗口的信道化输出 `[流][1024 帧][512 信道]`：
  - 逐信道平均功率；
  - 噪声底：每路信道功率的 25% 分位数，块内 bitonic 排序求得，活跃比例高时也不受影响；
  - 检测模式：相对噪声底（默认 6 dB）、绝对门限、全部信道；
  - **邻道泄漏抑制**：2 倍过采样时相邻信道中心落在原型滤波器的过渡带，强信号会在 k±1 出现约 −42 dBc 的泄漏。若某个邻道比本信道强 20 dB 以上，本信道不报（频率上首尾相接）；
  - 活跃信道压缩：手写单 block 的 warp ballot 前缀和（保序）与 `cub::DeviceSelect::Flagged` 各一版；
  - gather：只取活跃信道，经 smem 转置把 `[帧][信道]` 变成引擎输入 `[窗口][1024][2]`，读写都合并，同时归一化到单位 RMS（与训练一致）。
- **信道化器**支持输出按流 stride 写入：相邻两个块直接写进同一个窗口缓冲的前后两半，无需额外拷贝。
- **端到端流水线** `run_classifier`（`src/runtime/classifier.cu`）：
  - 生产者线程把数据填进锁页大页环形缓冲（与 P1 共用，生产者抽成了 `runtime/producer.cpp`，P1 流水线也改为使用它）；
  - 两条 CUDA stream：copy stream 做 H2D；compute stream 依次做信道化（解包已融合）→ 窗口缓冲（双缓冲交替）→ 检测 + 压缩 → 活跃数 D2H；
  - 活跃数的事件完成后，按"活跃数 = 本窗口 batch"、以引擎最大档位为块依次 gather → 引擎 → argmax / softmax → 结果 D2H；
  - 消费线程全程不阻塞 GPU。**动态 batch 由 GPU 上的检测器决定**，主机只读回一个计数。
- **合成源**：tone 增加初相位；生成改为 phasor 递推（每 1024 个样点精确重锚），避免数百个 tone × 2^20 个样点的 sin/cos 计算。
- **`curf_classify` 命令行**（合成或文件输入，负载旋钮 `--active N`，结果导出 CSV）。
- `tools/make_scenario.py`：带标注的宽带场景。
- `tools/cpu_baseline.py`：CPU 基线。
- **测试** 29 个，全部通过。新增：检测（两种压缩一致、三种模式）、邻道泄漏抑制、gather 分块转置与归一化。

**负载旋钮**（吞吐模式，合成源每路 N 个 tone 在信道中心，绝对门限检测；GPU 已连续工作处于热态；实时需求为每 4.27 ms 一个窗口）：

| 每路活跃信道（比例） | 每秒需识别窗口数 | GPU 实时倍数 | 每窗口 GPU 时间 p50 |
|---|---|---|---|
| 0（只做信道化 + 检测） | 0 | 10.6× | 0.39 ms |
| 8（1.6%） | 3.7k | 7.2× | 0.32 ms |
| 64（12.5%） | 30k | 3.9× | 0.83 ms |
| 128（25%） | 60k | 2.4× | 1.47 ms |
| 256（50%） | 120k | 1.35× | 2.89 ms |
| 320（62.5%） | 150k | 1.09× | 3.64 ms |
| 384（75%） | 180k | 0.93× | 4.35 ms |
| 512（100%） | 240k | 0.68×（164k 窗口/秒） | 6.16 ms |

- 第一版曲线在 128 / 256、384 / 512 处出现平台。原因是引擎 batch 档位只有 {1, 8, 32, 128, 512}：256 个窗口会按 512 的档位整跑，768 个会跑成 512 + 512。档位加密到 {…, 64, 192, 256, 384, 512} 后曲线恢复单调。
- 全负载 0.68× 与引擎单独测得的冷机 224k 窗口/秒之间的差距，来自信道化、检测、gather 的开销（每窗口约 0.4 ms），以及连续扫描时 GPU 一直处于热态。

**实时模式**（合成源按 61.44 MS/s 限速，满了就丢弃；每档 20 秒，4687 个窗口）：

| 每路活跃 | 丢样 | 端到端延迟 p50 / p99 / p999 / max |
|---|---|---|
| 8 | 0 | 0.49 / 0.57 / 0.62 / 0.71 ms |
| 64 | 0 | 0.95 / 1.05 / 1.08 / 1.15 ms |
| 128 | 0 | 1.68 / 1.76 / 1.79 / 1.85 ms |
| 256（第 1 次） | 0.25%（集中在一次突发） | 2.92 / 35.0 / 35.6 / 36.2 ms |
| 256（复测，环形缓冲 16 槽） | 0 | 2.90 / 2.99 / 3.01 / 3.04 ms |
| 256（环形缓冲 64 槽） | 0 | 2.92 / 3.42 / 3.49 / 3.51 ms |
| 320 / 384 | 持续丢样（超出算力） | 38 / 48 ms（排队） |

- 延迟从"窗口最后一块数据进入环形缓冲"算到"识别结果回到主机"。窗口本身的 4.27 ms 积累时间不计入，它由算法决定。加上这一项，端到端仍远低于目标 p99 < 15 ms。
- 50% 负载时 GPU 利用率约 74%：第 1 次的丢样集中在一次突发，发生在之前连续重负载导致 GPU 最热的时刻；复测为零丢样。

**端到端准确率**（`tools/make_scenario.py`）：

- 场景：2 路 × 61.44 MS/s 的 packed12 宽带录制，每路每隔 4 个信道放一个连续信号，共 128 个；信号用训练用的合成器生成（9 类，不含 noise），SNR 5–20 dB（与训练相同，按 240 kHz 带宽定义）。
- 生成方式是频域插值：基带频谱（±120 kHz，240 kS/s）与宽带频谱（61.44 MS/s，长 256 倍）的 bin 间隔相同，直接放到信道 k 的中心附近。时长 16 个窗口（68 ms）。
- 对照组（direct）：同样的信号切成同样的窗口，加相同 SNR 的白噪声，直接送入 PyTorch 模型。

| | 结果 |
|---|---|
| **流水线准确率**（窗口 0 除外，它包含零历史的建立暂态） | **92.92%** |
| direct 准确率 | 92.86% |
| 检测召回 | 100%（3840 / 3840） |
| 虚警 | 473 个（6 dB 门限）；**全部是真实信号的相邻信道，全部被分类器判为 noise**，没有一个被报成调制信号 |
| 各类准确率 | AM / BPSK / OFDM 100%，FM 99.8%，8PSK 97.7%，GFSK 96.8%，QPSK 89.1%，16QAM / 64QAM 约 79% |

- 12 位量化 → 信道化 → 检测 → 归一化 → fp16 引擎，整条链路几乎没有准确率损失。
- 门限的取舍：3 dB 时虚警 1048、召回 100%；**6 dB 时虚警 473、召回 100%（设为默认）**；10 dB 时召回只有 80%，低 SNR 信道漏检。

**CPU 基线**（i5-11400H，6 核 12 线程，AVX-512）：

- 推理用 ONNX Runtime 1.32 CPU（MLAS，12 线程，同一个 fp32 ONNX），最多 **3,339 窗口/秒**（batch 64）。
- 信道化用 PyTorch CPU 版（MKL FFT）：**0.069× 实时**。这不是优化过的 CPU 实现：按 FLOP 估算，AVX-512 多相滤波 + FFTW 只需这颗 CPU 约 4% 的峰值，理论上能跑实时。

**负载表**（用实测替换第 4 节的估算）：

| 场景 | 推理需求 | GPU（实测） | CPU（实测） |
|---|---|---|---|
| 只做信道化 | — | 10.6× 实时（含检测） | PyTorch 0.07×；优化后估计可行 |
| 10% 信道活跃 | 约 2.4 万窗口/秒 | 实时，p99 约 1 ms | 3.3k 窗口/秒，只够需求的 14% |
| 25% | 6 万 | 实时，p99 1.76 ms | 不可行 |
| 50% | 12 万 | 实时，p99 约 3 ms（一次复测前有突发丢样） | 不可行 |
| 62.5% | 15 万 | 1.09×，临界 | 不可行 |
| 100% | 24 万 | 0.68×（16.4 万窗口/秒） | 1.4%（只能跟踪约 14 个信道） |

结论：**GPU 能实时识别 50–60% 的信道，CPU 只能跟上约 14 个信道**，两者差约 40–50 倍。

**压缩：手写 vs CUB**：手写 warp ballot 3 µs（1 个 kernel，保序）；CUB `DeviceSelect` 2 个 kernel，每个约 4 µs，还需要临时缓冲。两者都可忽略。窗口阶段真正的开销在功率 kernel（69 µs，整读一遍 8 MB 窗口缓冲）和 gather（58 µs）。

**P3 剩余 / 改进点**：

- 功率统计融进信道化输出，省掉一遍 8 MB 读。
- 512 点 FFT 融进 fold kernel（预计信道化再快约 2×）。
- 第一层直接从信道化输出 gather，省掉引擎输入的写读。
- 全负载 0.68× → 1×：上面几项加上冷机状态，预计能接近。
- 时间平滑投票（架构里的 ⑦）。
- cupyx 对比（需要安装 cupy）。

### 2026-11-15 · P3 第三步：INT8 推理，全负载达到实时（热稳态差约 6%）

**先剖析，再动手**：用 nsys（`--cuda-graph-trace=node`，否则 CUDA Graph 里的 kernel 不会出现在统计中）看全负载（`--detect all`，1024 路全部识别）的稳态 2 秒：

- GPU 98.4% 时间在忙；
- **推理引擎占 91.6%**；
- 信道化 4.9%，gather 1.9%，检测 1.5%。

上一步列的几项融合（功率并进信道化、FFT 并进 fold 等）全做完，最多省 8%，到不了 1×。**瓶颈在推理本身**，所以这一步改做 INT8。

**INT8 卷积**（`src/kernels/conv1d_int8.cu`）：

- **量化方式**：对称量化，零点为 0。
  - 权重按输出通道量化，`s_w[n] = max|w[n]| / 127`。
  - 激活按张量量化，scale 来自校准。
  - 反量化系数 `alpha[n] = s_w[n] · s_x` 折进 epilogue：`y = act(acc_i32 · alpha + b (+ s_res · q_res))`，再按输出 scale 重新量化，饱和到 ±127。
- **kernel 与 fp16 版几乎相同**：`mma.sync.m16n8k32.s8` 的 A/B fragment 和 fp16 `m16n8k16` 覆盖完全相同的字节（每行 32 字节，每个线程持有相同的 4 字节字），所以 smem k-tile（每行 64 字节）、swizzle、ldmatrix 寻址、cp.async loader 全部照搬，只是每 16 字节从 8 个 half 变成 16 个 int8。
- **stem**：保持 direct kernel（fp32 输入、fp16 权重、fp32 累加），输出改为直接量化成 int8。
- **GAP 融合**：从 fp16 暂存 tile 求和，与 fp16 版共用 `pool_tile`。
- **校准**（`EngineConfig::int8 / calibration / calib_percentile`）：
  - 引擎内部建一个 fp16 引擎（不融合、保留所有张量、fp32 累加 kernel），跑一遍校准窗口；
  - 第一遍求每个张量的 max|v|，第二遍做 4096-bin 直方图，取 99.99% 分位数映射到 127。

**第一个结果：只快 7%**（batch 512：24.8 万 vs fp16 融合版 23.2 万窗口/秒）。

- 逐层看，INT8 kernel 本身比 fp16 快 1.25–1.5×。
- 但 INT8 下每个张量每窗口 32 KB，未融合的整网每窗口读写约 776 KB；24 万窗口/秒就是 **186 GB/s，已经顶到这块卡的显存带宽（约 192 GB/s）**。
- 算力翻倍后，层间读写成了瓶颈。

**INT8 BasicBlock 融合**（`src/kernels/block_int8.cu`）把 y1 和 shortcut 的结果留在片上，每窗口读写降到约 264 KB。与 fp16 融合块的区别：

- **k-tile 32 字节**：一个 m16n8k32 步；CIN 是 32 的倍数，所以一个 k-tile 不会跨抽头。
- **x 和 y1 各占一块 smem**：x 留着给 shortcut 和 identity 残差用，残差改为从 smem 读，不再回全局内存。
- **shortcut 和 conv2 不能共用 int32 累加器**：两者 scale 不同（`s_y1·s_w2` vs `s_x·s_wsc`）。shortcut 放到 conv2 之后算，复用 conv1 已经用完的累加器寄存器。
- **权重常驻**（`l64r / l128r`）：前三个 block 的权重只有 6–27 KB，和 x 一起一次装进 smem，k 循环完全不需要 barrier。
  - 只用 ring 版时，融合块每个 k32 步就要一次 barrier，只有 12–15 TFLOPS。
  - 常驻后 batch 512 整网从 1790 µs 降到 1636 µs。
- **stage3（64→128）**：权重 88 KB 放不进 smem。试了 64 字节 swizzle ring（每次 barrier 两个 k 步，`k2`），整网里融合仍比拆开跑慢约 3%，所以 stage3 保持三个独立 conv。`k2` 变体保留，正确性有测试覆盖。

**引擎吞吐**（冷机，CUDA Graph，`docs/results/p3_int8_engine.jsonl`）：

| batch | fp16 全融合 | INT8 不融合 | **INT8 + 融合块** |
|---|---|---|---|
| 1 | 39.9 µs | 41.8 µs | **37.2 µs** |
| 32 | 20.0 万/秒 | 22.5 万 | **26.6 万** |
| 128 | 21.6 万 | 23.7 万 | **28.6 万** |
| 256 | 22.7 万 | 24.5 万 | **30.0 万** |
| 512 | 23.4 万 | 24.9 万 | **30.0–31.5 万**（各次运行的 autotune 结果略有不同） |

batch 512 时每个 block 的耗时：

| | 耗时 | 算力 |
|---|---|---|
| stem | 140 µs | 访存受限 |
| block1 | 239 µs | 13.5 TFLOPS |
| block2 | 301 µs | 17.8 TFLOPS |
| block3 | 338 µs | 19.0 TFLOPS |
| stage3 三个 conv 合计 | 510 µs | conv2 + GAP 26 TFLOPS |

**准确率**：

- golden（256 个窗口，前一半用于校准）上与 fp32 的 top-1 一致率：后一半 127/128，全部 254/256；准确率 234 = fp32 的 234。
- 端到端场景（3840 个窗口，见上一步）：

| | 准确率 |
|---|---|
| fp32 直接推理 | 92.86% |
| fp16 流水线 | 92.92% |
| **INT8 流水线** | **92.66%**（召回 100%，473 个虚警仍全部判为 noise） |

- 校准分位数：1.0 → 92.66%，0.99999 → 92.73%，0.9999 → 92.66%，0.999 → 91.82%（截断过多）。默认 0.9999。

**全负载**（`--detect all`，2 路 × 512 信道全部识别，需要 24 万窗口/秒；`docs/results/p3_int8_full.jsonl`）：

| | 结果 |
|---|---|
| 吞吐模式（冷机） | **1.11–1.14×**（26.7–27.3 万窗口/秒）；同时段 fp16 是 0.79× |
| 实时模式 30 秒 | **1.00×，零丢样，端到端延迟 p50 3.8 / p99 4.1 / max 4.2 ms**（不到一个 4.27 ms 的窗口周期） |
| 实时模式 150 秒 | 0.94×（零丢样，生产者被背压）：前约 50 秒保持实时，之后 GPU 温控降频 |

**热稳态**（`docs/results/p3_int8_thermal.csv`）：

- 开始约 45 秒：受 60 W 功耗墙限制，1650–1740 MHz。
- 87 °C 后触发软件温控降频，稳定在 **1240–1370 MHz、39–45 W**。
- 这台笔记本能长期散掉的热量约 40 W，所以持续全负载时真正的指标是每个窗口的能耗；热稳态还差约 6%。

**排查中发现的两件事**：

- **autotune 在热态下不可靠**：每次启动都重新调优，GPU 热的时候计时噪声大，选出来的 kernel 时好时坏，同样的全负载跑出过 0.89–1.09×。用 `--tune-cache` 固定一次冷机调优的结果后，连续 4 次稳定在 1.12–1.14×。部署时应在冷机状态下调优一次并缓存结果。
- **主线程忙等**：`cudaEventSynchronize` 默认自旋，classifier 进程占满一个 CPU 核（90%）。笔记本的 CPU 和 GPU 共用散热，改为 `cudaEventBlockingSync` 事件后降到 30%，延迟不变。

**其他改动**：

- `curf_classify` 新增 `--int8`（用模型目录的 golden 窗口校准）、`--calib-percentile`、`--tune-cache`。
- `curf_bench_engine` 的 step 支持 `+int8`。
- 测试 32 个，全部通过。新增：
  - 每个 INT8 conv tile 逐层对齐 golden；19 种 tile 的误差完全相同，最大相对误差 4.4%。
  - INT8 融合块的 6 种前缀。
  - 调优后的 INT8 引擎准确率与 fp32 对比。

**剩余 / 改进点**（按热稳态下预期能省的比例排）：

- stem 融进 block1：block1 直接从 fp32 输入计算自己需要的 stem 行，省掉 stem kernel，每窗口少写读 64 KB（引擎约 6–8%）。
- stage3 融合块：目前受计算和延迟限制（约 13 TFLOPS），拆开跑反而受显存带宽限制；需要一个更好的 64→128 融合块。
- 信道化、检测、gather 合计约 8%：功率统计并进信道化、FFT 并进 fold、stem 直接从信道化输出 gather。
- 时间平滑投票（架构里的 ⑦）；cupyx 对比（需要安装 cupy）。

## 2026-11-15 · P3 第四步：stem 融进 block1，热稳态 0.99×

**做法**（`block_int8.cu` 的 STEM 模式，变体 `i8.block.stem.{l128r,l64r,l128,l64}.32x32s1`）：

- block1 的 CTA 不再从显存读 x，而是用 cp.async 取自己需要的 fp32 输入行（含 2+2 行 halo，多一行让起点 16 字节对齐），在片上算出 stem 输出并量化成 int8 写进 x tile。stem 输出（每窗口 16 KB，写一次、读两次）不再经过显存，也少一次 launch。
- **第一版用 fp32 FMA 算 stem，几乎没省**：融合 kernel 190 µs，而 stem 单独 80 µs + block1 123 µs = 203 µs。stem 每个位置 × 8 通道要 112 次 FMA 和 42 次 smem 读，CTA 内这一段是串行的，每个 SM 只有 4 个 CTA，藏不住。
- **改用张量核**：K = 14（7 tap × 2 通道）补到 16，一条 m16n8k16 fp16 MMA；A 片段直接从 smem 里的 fp32 行读 float2 转 half2（第 r 行就是从 2 + 4r 起连续 16 个 float），每 CTA 约 36 条 MMA。融合 kernel 降到 **157 µs**（batch 256）。
- 输入舍入到 fp16 的代价：与“stem kernel + block”相比，block1 输出 1.6% 的元素差 1–2 个 int8 步，RMS 相对差 0.7%（INT8 本身对 fp32 的误差约 4%）；准确率不变。
- 引擎的融合决策从“融合 / 不融合”扩成三选一（不融合 / block / stem + block），仍然在整网里交替计时决定。所有 batch 都选了 stem 融合。
- 试过 256 位置的 tile（每 CTA 8 个 warp，halo 占比减半）：161.7 vs 160.5 µs，没有收益，已删除。

**引擎吞吐**（`docs/results/p3_int8_stem.jsonl`）：

| batch | 融合前 | **stem 融进 block1** |
|---|---|---|
| 1 | 37.2 µs | **34.4 µs** |
| 32 | 26.6 万/秒 | **28.1–28.7 万** |
| 256 | 30.0 万 | **31.0–31.9 万** |
| 512 | 30.0–31.5 万 | **31.4–32.2 万** |

每次 launch 7 个 kernel（原来 8 个）。收益约 3%，低于预估的 6–8%：stem 的计算没有消失，只是搬进了 block1，而且在 CTA 内与 conv1 串行。

**全负载**（`--detect all`，2 路 × 512 信道，需要 24 万窗口/秒）：

| | 融合前 | **stem 融进 block1** |
|---|---|---|
| 吞吐模式 | 1.11–1.14×（冷机） | **1.14–1.15×**（GPU 已 75–78 °C） |
| 实时 150 秒 | 0.94× | **0.99×，零丢样**（23.8 万窗口/秒） |
| 热稳态时钟 | 1240–1370 MHz、39–45 W | 1390–1400 MHz、44.7 W（`docs/results/p3_int8_stem_thermal.csv`） |

150 秒里前约 60 秒保持实时，之后积压，p99 延迟升到 42 ms。热稳态还差约 1–2%。

**测试**：33 个，全部通过。新增“INT8 stem 融进 block1 与单独 stem 一致”：4 种 tile，差异元素 ≤ 5%、最大 ≤ 3 步、RMS < 1%，各层对 golden 误差 < 12%，top-1 32/32。

**剩余 / 改进点**：

- stem 段与 conv1 的串行：可以让一部分 warp 先算 halo 以外的行、其余 warp 先开始 conv1（warp 分工），或把 stem 的 36 条 MMA 插进 conv1 的 k 循环。
- stage3 融合块（权重 88 KB，需要更好的 64→128 方案）。
- 信道化、检测、gather 合计约 8%：功率统计并进信道化、FFT 并进 fold、stem 直接从信道化输出 gather（stem 融合后，这一步可以让 block1 直接读信道化输出）。

## 2026-11-15 · P3 第五步：head kernel 与 stage3 shortcut 融合，全负载 150 秒实时（300 秒差约 1.5%）

**profile**（stem 融合后，全负载 2 秒窗口，GPU 忙 94%）：引擎 88.8%、信道化 6.5%、gather 2.6%、检测 1.9%。引擎内 stage3 的三个 conv 占 31.9%；head 的 `linear_from_pool` 每次 28 µs，占 1.6%。

**head kernel**：原来每个样本一个 block、每个类别一个线程：128 个线程里只有 10 个在干活，每个跑 128 步串行 FMA，权重按步长访问。改成一个 warp 算一个（样本, 类别）：合并访问加 shuffle 归约。引擎 batch 512 从约 1590–1630 µs 降到 1544 µs。

**stage3 conv2 + shortcut**（`conv1d_int8.cu` 的 SC 变体 `i8.mma.sc.*`）：

- shortcut（1×1 stride 2，64→128）原来单独一个 kernel（batch 512 时 111 µs，约 9.6 TFLOPS），它的输出每窗口 16 KB，写出后立刻被 conv2 读回。
- 现在 conv2 的 k 循环先跑 shortcut 的 k-tile（A 取 block 输入的偶数位置，B 取 shortcut 权重），然后把 int32 累加器按 shortcut 的 scale 反量化进 fp32 寄存器（每线程多 32 个），清零后再跑 conv2 的 GEMM，epilogue 把两者相加再做 ReLU / GAP。两个 GEMM 的 scale 不同，所以不能共用一个累加器。
- shortcut 不再先量化成 int8，而是以 fp32 加进去：logits 与原路径的相对差 0.4%，对 fp32 的误差不变（4%），top-1 32/32。
- 融合决策增加第四种：conv1 + “conv2 带 shortcut”。整网计时：batch 512 时 1458 µs，对比 stage3 拆成三个 conv 1549 µs、stage3 融合块 1625 µs。所有 batch 都选了它。

**引擎吞吐**（`docs/results/p3_int8_sc.jsonl`）：

| batch | stem 融合后 | **+ head + shortcut 融合** |
|---|---|---|
| 1 | 34.4 µs | **30.1 µs** |
| 32 | 28.1–28.7 万/秒 | **31.1 万** |
| 256 | 31.0–31.9 万 | **33.9 万** |
| 512 | 31.4–32.2 万 | **34.3 万** |

每次推理 6 个 kernel（P3 第三步时是 8 个）。

**全负载**（`--detect all`，2 路 × 512 信道，需要 24 万窗口/秒）：

| | stem 融合后 | **本步** |
|---|---|---|
| 吞吐模式 | 1.14–1.15× | **1.23–1.25×**（GPU 76–82 °C） |
| 实时 150 秒 | 0.99× | **1.00×，零丢样，p50 延迟 3.8 ms**（p99 40.6 ms，时钟掉到 1240 MHz 附近时有短暂积压，随后追上） |
| 实时 300 秒 | — | 0.99×（23.7 万窗口/秒，零丢样）：约 3 分钟后时钟再降到 1237–1310 MHz、37–41 W，队列积满，p50 延迟升到 38 ms，生产者被背压 |

热稳态时钟 1240–1490 MHz、39–47 W（`docs/results/p3_int8_sc_thermal_150s.csv`）。

**测试**：34 个，全部通过。新增“INT8 conv2 融合 shortcut 与单独 shortcut 一致”（6 种 tile）；常驻权重块测试中 stage3 现在走 conv1 + 融合 shortcut 的 conv2，期望的步数和张量数相应更新。

**剩余**：

- 长时间热浸后（GPU 约 37–41 W）还差约 1.5%。信道化 + 检测 + gather 合计约 11%，是剩下最大的一块；gather 可以并进 stem 融合块（直接读信道化输出），省掉每窗口 8 KB 的写和读，以及一次 launch。
- stem 段与 conv1 的串行。

## 2026-11-15 · P3 第六步（负结果）：gather 并进 stem 融合块，没有收益

**做法**（测完后已删除，代码不保留；以下记录实现方式和测量结果）：

- 引擎在显存里有一个 `GatheredInput` 描述符槽（窗口缓冲、power、活跃列表、count、first）。stem 融合块经它直接从信道化输出读样本：每个样本一次 8 字节 cp.async，乘 `rsqrt(power)`，和 gather kernel 的 fp32 归一化相同。
- CUDA Graph 固定了 kernel 参数，所以每个 chunk 的变化部分放在显存里：classifier 为每个（窗口缓冲, chunk）预先建好描述符，每次推理前只做一次 48 字节的 D2D 拷贝，代替 45 µs 的 gather kernel。
- 默认描述符指向引擎自己的 input()（M = 1，power = 1），调优时就用它。
- 测试：与“gather + 引擎”相比 logits **逐位相同**（两个 chunk，含越过 count 的部分 batch）。

**结果**（nsys，全负载，每次推理 512 窗口）：

| | stem 融合块 | gather | 合计 |
|---|---|---|---|
| gather kernel + 引擎 | 331–344 µs | 45.5 µs | 377–390 µs |
| 引擎直接读（第一版） | 389 µs | — | 389 µs |
| 依赖链缩短后 | 387 µs | — | 387 µs |

端到端交替各跑 3 次：直接读 1.20–1.23×，gather kernel 1.22–1.26×，直接读稳定慢 1–2%。

**原因**：信道化输出是 `[stream][t][channel]`，一个窗口的相邻样本相隔 4 KB。stem 块每个 CTA 只读一个窗口，每个 32 字节 sector 只用到 8 字节，而且是在 conv1 之前串行等这些访存（ncu：long-scoreboard stall 翻倍）。gather kernel 每行读 32 个相邻信道，访存是连续的。试过的补救都没有效果：
- 权重拷贝排到依赖链前面；
- grid 改成窗口优先，让同时驻留的 CTA 读同一批行；
- power 移出依赖链。

要让“直接读”成立，需要信道化按信道把时间连续写出（等于把 gather 的转置搬进信道化），或者一个 CTA 同时处理相邻的多个窗口。

**处理**：整条路径已删除（引擎的描述符槽、`set_gathered_input`、stem 块的直接读路径、classifier 的描述符、`--engine-gather` 和对应测试），回到 gather kernel + 引擎。删除后 34 个测试全部通过，全负载吞吐 1.24–1.25×。

## 2026-11-15 · P3 第七步：流水线空档、功率统计、fp16 输入；300 秒满负载实时（在热极限上持平）

**profile**（全负载吞吐模式，按 kernel 之间的空档统计）：GPU 空闲 6.1%，其中 4.8% 是每个窗口一次、平均 163 µs 的空档，出现在“count 回传主机 → gather”之间：主机等 count 回来、被唤醒、再提交 gather 和推理。

**1. 推测提交**（`classifier.cu`）：

- 提交检测时，按上一个窗口的活跃数直接把 gather、推理、后处理和结果回传一起排进队列。
- gather 在 GPU 上按真实的 count 截断，超出部分的行算了也不用；count 回来后，如果比预测的多，再补提交差额。
- 空闲从 6.1% 降到 1.6%，吞吐模式从 1.24–1.25× 升到 **1.31–1.32×**。
- 带标签场景（每个窗口的 count 都不同，两条路径都会走到）：召回 100%，准确率 92.73%。

**2. 功率统计**：原计划并进信道化的 FFT，但 FFT 是 cuFFT 的内部 kernel。存储回调要走 LTO JIT，而且每个块约有 50 万次原子加落在 1024 个地址上，大概率比省掉的那次读还慢，所以没有做。改为让功率 kernel 用满 GPU：原来只有 4 个 block（2 组信道 × 2 路），每个都要循环 1024 帧。现在按帧切成 16 段，各段部分和由 `floor_flag` 按固定顺序求和（结果保持确定，也不增加 launch）。62 µs → 50.5 µs（约 166 GB/s，接近显存带宽）。

**3. fp16 输入**（`EngineConfig::half_input`，INT8 下 `curf_classify` 默认开启，`--fp32-input` 关闭）：

- 融合 stem 进张量核前本来就要把输入舍入成 fp16。现在改由 gather 直接写 fp16（`WindowStage::gather_half`），gather 的写出量和 stem 的读入量都减半。
- 开启后 stem 一定融合，不存在 fp16 输入的独立 stem kernel。
- 测试：与 fp32 输入的引擎相比，logits **逐位相同**（2 种 tile × 2 个 chunk，含不满一个 bucket 的 batch）。
- 交替各跑 3 次：fp16 输入 1.26–1.30×，fp32 输入 1.25–1.28×，每对都是 fp16 快 1–1.5%。
- 场景：召回 100%，准确率 92.76%。

**300 秒满负载实时**（`docs/results/p3_pipeline_overlap.jsonl`）：

| | 吞吐 | p50 延迟 | 热稳态时钟 |
|---|---|---|---|
| 上一步 | 0.99×（23.7 万窗口/秒） | 38 ms | 1237–1310 MHz |
| + 推测提交 + 功率 kernel | **1.00×**（23.98 万），零丢样 | 9.3 ms | 1237–1327 MHz、38–42 W |
| + fp16 输入 | **1.00×**（23.93 万），零丢样 | 31 ms | 几乎一直是 1237 MHz、38.5 W |

两次 1.00× 的差别来自散热：后一次起始温度 61 °C（前一次 57 °C），而且几乎全程停在 1237 MHz。结论：在这台笔记本的热极限（约 38.5 W、1237 MHz）下，流水线刚好持平，队列时满时空，取决于散热状态。

**测试**：35 个，全部通过。

**再往上**：现在受限于能量：每个窗口在 38.5 W 下要 4.27 ms 内算完。kernel 级的小优化（stem 与 conv1 并行、stage2/3 的 tile）还能挤出几个百分点，但要有明显余量，需要减少计算本身：
- 时间复用：类别和功率没变的信道隔窗口才分类；
- 模型瘦身：stage3 变窄、剪枝加蒸馏。

或者换一台散热更好的机器。

## 2026-11-15 · P4 第一步：OmniCore 实采源与展示界面（硬件未接，实采待测）

CUDA 主机的 USB 设备列表里还没有 SDR，所以先做不依赖硬件的两项。

**1. OmniCore 实采源**（`OmniCoreSource`，`-DCURF_OMNICORE=ON`；用已安装的 OmniCore Host，或 `-DCURF_OMNICORE_SOURCE=<源码目录>` 当子项目构建）：

- `read()` 是一次 `och_recv`，应用格式选成 wire 格式本身（12 bit 用 packed12，8 bit 用 sc8），驱动里就是一次 memcpy，直接写进 HostRing 的锁页槽。零拷贝租借 `och_rx_acquire` 同样要一次主机拷贝，而且租借只能整块释放，槽放不下一整块时就得另存一份；recv 能在块中间停下，所以用 recv。
- recv 不会把不同 epoch 拼进同一次返回，每次读只有一个样点时间。`SampleClock` 把它换算到 ISource 的时间轴上：有硬件时间（1 tick = 1 样点）时直接用时间差，丢多少样就是多大的间隔；没有时按驱动报告的丢样数累加。主机溢出、设备溢出、改参都标成 discontinuity；设备时间回退时接着上一次读继续编号，并重新对齐原点。
- `curf_classify --source omnicore [--freq --gain --agc --iq-bits --bandwidth --serial]`，实采时自动用丢样模式（硬件不会等），结果 JSON 里加上驱动统计（USB 字节数、主机丢样、设备溢出、传输错误、主机队列峰值、discontinuity 次数）。
- 测试：`SampleClock` 两个 CPU 测试（硬件时间下的连续、丢样、改参、时间回退后重新对齐；无硬件时间下的丢样累加）。写测试时抓到一个 bug：回退后原点在时间“上方”，无符号比较出错，改成模运算后的有符号差。
- 适配器本身只能编译验证：OmniCore 的 fake 传输层不经过 C ABI。Mac 上把 OmniCore 当子项目编译通过；CUDA 主机缺 `libusb-1.0-0-dev`（只有运行库），还没编。

**2. 展示界面**（`tools/ui_server.py` + `tools/ui/index.html`，只用 Python 标准库）：

- `curf_classify --ui-out PATH [--ui-every 8]` 输出 JSON lines：每 8 个窗口（34 ms）一行，含每个信道这段时间内的最大功率（dBFS，取整）、每路的噪声本底和最后一个窗口的识别结果。数据来自 classifier 新加的 `on_power` 回调：检测之后多一次 4 KB 的 D2H，排在 count 回传之后，不推迟 count。
- `ui_server.py -- curf_classify ...` 实时模式：自己启动 classify，经管道读 feed，用 SSE 推给浏览器；`--replay FEED` 按每行的样点时间回放录下的 feed。
- 页面：每路一个面板，频率轴（有中心频率时显示绝对频率）、识别标注条（相邻同类信道合并成一段，放得下时写类名和概率）、功率瀑布图、按类别着色的识别瀑布图（亮度随概率），顶部是各类别当前的信道数。
- 开销：满负载吞吐模式交替跑 4 次，有 UI 输出 1.31× / 1.26×，没有 1.29× / 1.24×，差异在热漂移范围内。
- 用 scn8 场景的回放在浏览器里验证过：29.6 行/秒，2 × 128 个信号都标注出来了。

**测试**：37 个（CPU 16，CUDA 21），全部通过。

**接硬件需要的**：
- SDR 接到 CUDA 主机的 USB3 口；
- 安装 `libusb-1.0-0-dev`、OmniCore 的 udev 规则，并把用户加入 plugdev 组（都要 sudo）；
- 回环用的线缆和衰减器。

之后按计划做 300 秒实采零丢样长测，再做 TX→RX 回环。回环时 2 RX + 1 TX 在 12 bit 下最高 58.5 MSPS（UHSIF 总线上限）；要 61.44 MSPS 就用 8 bit 2T2R。

## 2026-11-15 · P4 第二步：实采接通，300 秒零丢样（相对门限检测）

**接入**：SDR 接在 CUDA 主机的 USB3 口上（5 Gb/s，固件 SDK_CMD_1.2_20261107_SYNC，FPGA IP 1.3，带硬件时间戳）。
- libusb 开发包用 `apt-get download` 加 `dpkg -x` 解压到 `~/opt/libusb-dev`，没有改动系统包。
- OmniCore 0.8.0 装在 `~/opt/omnicore/0.8.0`。
- udev 规则由用户安装。
- cuRF 的配置命令（`PKG_CONFIG_PATH` 指向 `~/opt/libusb-dev/root/usr/lib/x86_64-linux-gnu/pkgconfig`）：
  `cmake --preset cuda-release -DCURF_OMNICORE=ON -DCMAKE_PREFIX_PATH=~/opt/omnicore/0.8.0`

**传输层对照**：`och bench --rx --channels 1,2 --sample-rate 61.44M --seconds 10` 跑出 368.8 MB/s，零丢样、零溢出。

**接通时发现的问题**：AD9361 的模拟带宽上限是 56 MHz，原来默认把带宽设成采样率（61.44 MHz），`och_configure` 直接拒绝。现在默认取 min(采样率, 56 MHz)。代价是两侧边缘各约 2.7 MHz 被模拟滤波器衰减。

**300 秒实采，2 × 61.44 MSPS，中心 98 MHz，增益 40 dB**（`docs/results/p4_live.jsonl`，温度和时钟记录在 `p4_live_*_thermal_300s.csv`）：

| 检测模式 | 吞吐 | 驱动丢样 / 溢出 | 流水线丢样 | 延迟 p50 / p99 / p999 / max | GPU |
|---|---|---|---|---|---|
| 相对门限（工作模式，平均 318 / 最多 574 个活跃信道） | **1.00×** | 0 / 0 | **0** | 1.74 / 2.47 / 2.66 / 8.4 ms | 平均 86 °C、1377 MHz |
| 全部 1024 个信道 | 0.97× | 0 / 0 | 3.0%（5.5 亿帧） | 34.5 / 44.5 / 45.7 / 46.5 ms | 89 °C、1072–1222 MHz |

- 工作模式达到 P4 的验收标准：实采长测零丢样，端到端 p99 2.5 ms。discontinuity 次数为 0，主机队列峰值 0.9 MiB。
- 满负载在实采时扛不住。合成数据的长测能在 1.00× 持平，实采时 CPU 还要处理 368 MB/s 的 USB 数据（USB 线程，加上生产者往锁页内存的拷贝），而 CPU 和 GPU 共用散热，GPU 稳定时钟从约 1237 MHz 掉到约 1150 MHz，正好差了 3%。驱动那一层一直没丢；丢样发生在 HostRing 满了之后，是丢样模式按设计丢掉的。
- 要让满负载在实采时也够用，还是得减少计算（时间复用、模型瘦身）。另一种办法是零拷贝：USB 请求直接写进 CUDA 锁页内存，省掉一次主机拷贝和相应的 CPU 功耗。

**识别结果还不能说明准确率**：
- 频谱里看不到清楚的 FM 电台。本底约 −68 dBFS，87–98 MHz 只有间歇的宽带突发（高约 20 dB）；101.8–119 MHz 有一段边缘很陡、整体高约 10 dB 的平台。天线情况未知。
- 被检测到的信道 97% 判成 ofdm：在 240 kHz 的单个信道里，宽带类噪声能量和 OFDM 本来就分不开。
- 另外有两个窄带常驻信号（104.00 MHz 和 117.32 MHz，只在 RX2 上），判成了 bpsk。
- 准确率要用已知信号来测，即 TX→RX 回环。

**展示界面**：用这次实采录下的 feed 回放验证过。频率轴显示绝对频率，宽带平台、突发和识别着色都显示正常。

## 2026-11-15 · P4 第三步：接天线后的实地检查

每个频段采 5 秒，相对门限检测，增益 40 dB（FM 段另测了 60 dB），驱动和流水线全部零丢样。汇总脚本是 `tools/live_survey.py`（2026-11-17 已删除，由界面和 `capture_survey.py` 的占用摘要代替）。

| 频段 | 看到的信号 | 识别结果 |
|---|---|---|
| 950 MHz | 934.6–944 MHz 连续占用，占空比 1.0（中国移动 900 MHz 下行，目前主要是 LTE/NR） | ofdm，置信度 0.92–1.00 ✓ |
| 2.44 GHz | 2443–2447 MHz 等处间歇突发，占空比 13–17%，峰值 −15 dBFS（WiFi） | 约 2/3 判成 ofdm，其余主要是 8psk、fm，多出现在突发边缘；另有少量 gfsk，可能是蓝牙 |
| 98 MHz，60 dB | 91.04 MHz 一个持续存在的电台 | fm，0.89 ✓ |
| 98 MHz | 101–119 MHz 一段平坦平台：宽 18 MHz、边缘很陡，接天线前后都在 | ofdm（像是板子或主机自身的干扰，不是广播） |
| 98 MHz | 104.00 MHz 窄带常驻信号，接天线前后都在 | bpsk（像是杂散） |

**结论**：
- 宽带 OFDM（LTE、WiFi）和一个真实 FM 电台都判对了。
- 但"信道里填满了类噪声能量"在 240 kHz 的单个信道里就是 ofdm 的样子，所以 ofdm 的命中只能算定性的一致，不能当准确率。
- 这根天线在 100 MHz 收得很弱，FM 段只有 1 个电台。
- 定量准确率还是要靠 TX→RX 回环（已知调制、已知信道）。

**界面修正**：
- 问题：功率瀑布每一行取的是 8 个窗口的最大功率，识别行却只用了最后一个窗口。WiFi 突发很短，功率图亮的地方识别图常是空的。
- 修正：识别行改为每个信道取这 8 个窗口里功率最大那次的结果。2.44 GHz 实时界面上，两张图已经对齐。

## 2026-11-16 · P4 第四步：TX→RX 线缆回环，实采准确率

**接法与保护**：TX1 用 SMA 线直连 RX1，**没有外接衰减器**。保护全部放在软件里：
- `OmniCoreConfig::kMinTxAttenuationDb = 40`：低于这个值，程序在打开设备之前就抛异常。实测 30 dB 被拒绝。这时进入 RX 的功率不超过 −33 dBm，AD9361 RX 的绝对最大额定值是 +2.5 dBm。
- TX 衰减和 TX 使能在同一次 `och_configure` 里设好，TX2 一直是最大衰减（不再用驱动默认的 10 dB）。
- 扫描从 89.75 dB 开始，每级降 5 dB；RX 峰值超过 −20 dBFS 就停。

**发射波形**（`tools/loopback.py make`）：
- 单路 61.44 MS/s，sc16 格式，32 个窗口的长度。
- 在 ±25 MHz 通带内、|k| ≥ 4（避开本振泄漏）的范围里，每隔 4 个信道放一个随机类别的干净信号，共 104 个，覆盖 9 类，电平相同。
- 总 RMS −15 dBFS，峰值 −3 dBFS。
- 在频域里按整个周期构造，所以循环发射时首尾没有接缝。

**链路**：2450 MHz（ISM 频段），RX 增益 30 dB，每级 5 秒（1165 个窗口）。每一级的设备溢出、TX 欠流、流水线丢样都是 0。

中途 SDR 一度以 USB 2.0（480 Mb/s）重新枚举，只收到 4% 的样点，那一组作废。插回 USB3 之后正常。

**结果**（`docs/results/p4_loopback.jsonl`；SNR 是信号信道的接收功率中位数减去本底）：

| TX 衰减 | SNR 中位数 | 召回 | 准确率 |
|---|---|---|---|
| 89.75–75 dB | 信号还在本底以下 | 0.22–0.24 | 0.16–0.18 |
| 70 dB | 6.9 dB | 0.61 | 0.552 |
| 65 dB | 10.3 dB | 0.92 | 0.758 |
| 60 dB | 14.5 dB | 0.99 | 0.838 |
| 55 dB | 19.0 dB | 1.00 | 0.884 |
| 50 dB | 23.6 dB | **1.00** | **0.914** |

- 89.75–75 dB 这几级的结果和 TX 功率无关。那时检测到的只是和外部信号重叠的信道（几乎都判成 ofdm），表里的"SNR"也是这些信道的，没有意义。
- SNR 每级升约 4.5 dB，和衰减的 5 dB 步长基本一致。
- 50 dB 时分类别的准确率：ofdm 1.00、fm 1.00、bpsk 0.95、8psk 0.95、qpsk 0.90、am 0.87、gfsk 0.87、64qam 0.84、16qam 0.81。
- 作为对照，同样 SNR 下模型直接吃合成数据的准确率约 95%，合成宽带场景走整条流水线是 92.7%。实际射频链路（DAC/ADC、模拟滤波器、本振、IQ 失衡）带来的损失大约 3–4 个百分点。16/64QAM 互相混淆的比例和模型本身的混淆矩阵一致。
- 扫描最低只用到 50 dB 衰减，RX 峰值一直在 −43 到 −45 dBFS（这个峰值也不是我们的信号，不随 TX 功率变化）。

**误报**：每个窗口约 155–165 个。
- **约一半在频带边缘**（|f| > 25.5 MHz，每窗口约 80 个）：最强的在 ±30.5 MHz，比本底高约 13 dB，和 TX 功率无关。这是模拟滤波器以外的边缘噪声被抬高，相对门限把它当成了信号。
- **紧挨信号的信道**（每窗口 49–69 个）：信道化是 2 倍过采样，240 kHz 宽的信号有一半落进相邻信道。合成场景里也有这类误报。
- **其余带内**（每窗口 20–30 个）：外部的 2.4 GHz 信号经板子或线缆泄漏进来。

**下一步可以做的两项检测改进**：
- 屏蔽模拟通带以外的信道，即 |f| > 带宽/2 的那些；
- 相邻信道抑制：只保留局部最大，或者比邻居低 6 dB 以上的就不算。

两项都是检测 kernel 里的小改动，预计能去掉大约 80% 的误报。

## 2026-11-16 · P4 第五步：检测加频带屏蔽，相邻抑制阈值按实测重定，回环误报降 78%

**依据**（用第四步的回环数据离线模拟后再改）：
- **频带边缘**：不发信号时（TX 衰减 89.75 dB），噪声不是在边缘突然抬升，而是从约 20 MHz 起逐渐升高。20–24 MHz 比中心高约 5 dB，到 ±30 MHz 高约 12 dB。这符合 AD9361 sigma-delta ADC 的噪声整形，再加上 56 MHz 模拟滤波器以外的部分。
- **相邻信道**：实际信号在 k ± 1 信道里只比自身低约 10 dB（中位数，10% 分位是 5 dB）。原来的 20 dB 阈值是按合成数据（约 −42 dB）定的，对实采数据从来不起作用。

**改动**（`DetectConfig`，在 `floor_flag_kernel` 里实现）：
- `band`：只上报中心 ± band/2 以内的信道，噪声本底也只用这些信道估计（带外的值在排序时置为 +inf）。`curf_classify --band-mhz B`，默认全带宽，omnicore 源默认 52 MHz。
- `neighbor_reject_db` 从 20 改为 **6**：邻居比自己强 6 dB 以上就不上报；电平平坦的宽带信号不受影响。`--neighbor-db` 可以调。
- 新增 2 个测试：−10 dB 的泄漏被抑制、8 个信道宽的平坦信号保留；带外信道既不上报，也不参与本底估计。测试共 39 个，全部通过。

**回环复测**（同一个波形，2450 MHz，`p4_loopback_detect`）：

| TX 衰减 | SNR | 误报/窗口（前 → 后） | 召回（前 → 后） | 准确率（前 → 后） |
|---|---|---|---|---|
| 70 dB | 6.9 dB | 156 → 80 | 0.612 → 0.680 | 0.552 → 0.574 |
| 60 dB | 14.6 dB | 155 → 72 | 0.990 → 0.998 | 0.838 → 0.834 |
| 50 dB | 24.2 dB | 165 → **36.8** | 1.000 → 1.000 | 0.914 → 0.914 |

低 SNR 时召回反而提高，因为本底不再被边缘的高噪声抬高。

合成场景 scn8：准确率 92.76%，召回 100%，误报 16 个，全部紧挨信号，没有一个被判成信号类别。

**剩下的误报**：主要在 20–26 MHz，噪声已经开始抬升，但还在屏蔽范围以内。根本的解决办法是按频率估计本底，比如在 ±32 个信道的滑动范围内取 25% 分位数，代替每路只用一个本底值。

## 2026-11-16 · P4 第六步：AD9361 RX FIR（×2 抽取）

**背景**：接收链原来是 ADC 491.52 MHz → HB3 → HB2 → HB1 → 61.44 MSPS，可编程 FIR 旁路。模拟带宽早已开到最大（56 MHz，`rx.rf-bandwidth` 的上限）。设备里原来存的那组 64 tap 系数不是低通：总和只有 20，0.2 fs 处还有 +70 dB 的峰。好在它一直是关的。

**设计**（`tools/fir/design_rx_fir.py`，输出 `tools/fir/rx_dec2_61m44_64tap.ftr`）：
- remez 等波纹设计，输入 122.88 MHz，抽 2，通带 ±24 MHz，阻带从 30.72 MHz 起；
- 系数总和归一到 32768，GAIN 0。浮点下通带波动 0.08 dB，阻带 −78 dB。

**装载**：
- 128 tap 被 SDK 拒绝（`fir_rate_invalid`）：61.44 MSPS 下的乘加预算只够 64 tap。
- 64 tap 用 `och fir rx load` 装载、`och fir rx on` 启用后，SDK 自动把链改成跳过 HB1：ADC 491.52 → 245.76 → 122.88 → FIR 抽 2 → 61.44。
- `och clock set` 不能先改链，因为链和 FIR 因子必须一致。
- 之后 `curf_classify` 的 configure 会保留 FIR 和时钟链；启动信息里会写明 "RX FIR on (dec 2, 64 taps)"。
- FIR 状态存在设备 RAM 里，断电后丢失。
- 直流跟踪和 IQ 跟踪保持开启。实测开着时直流信道只比本底高 5–6 dB；关掉的话，本振泄漏会成为常驻误报。

**纯噪声对比**（TX 不发射，2450 MHz，增益 30 dB；中位功率相对本底，dB）：

| \|f\|（MHz） | 0–10 | 10–20 | 20–22 | 22–24 | 24–25 | 25–26 | 26–27 | 27–28 | 28–29 | 29–30 | 30–31 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FIR 关 | +1 | +3 | +6 | +9 | +10 | +11 | +12 | +12 | +13 | +13 | +13 |
| FIR 开 | +3 | +5 | +6 | +8 | +10 | +9 | +7 | +2 | −5 | −13 | −15 |

- 27 MHz 以外被压下去了：这部分是 HB1 过渡带折叠进来的，FIR 在抽取之前把它滤掉了。
- 20–26 MHz 的抬升没变，是通带内的噪声（ADC 噪声整形或模拟链），FIR 管不到。61.44 MSPS 下 ADC 已经是能用的最高时钟。
- 中间那段"高了 2 dB"是假象：25% 分位数现在包含了边缘那些很低的信道，本底估计被拉低了。

**回环**（52 MHz 屏蔽，`p4_loopback_fir`）：

| TX 衰减 | 误报/窗口（FIR 关 → 开） | 召回 | 准确率 |
|---|---|---|---|
| 50 dB | 36.8 → 35.0 | 1.000 → 1.000 | 0.914 → 0.915 |
| 60 dB | 72.4 → 71.1 | 0.998 → 0.992 | 0.834 → 0.836 |
| 70 dB | 79.7 → 78.0 | 0.680 → 0.632 | 0.574 → 0.558 |

- 70 dB 这档的 SNR 约 7 dB，正好压在检测门限上，召回的差别属于正常起伏。
- FIR 开着时，把屏蔽放宽到 56 MHz，误报会从 35 升到 42.6。

**结论**：
- 屏蔽本来就排除了 FIR 能清理的那段，所以 FIR 对这次回环几乎没有影响。
- 它的价值在实采：频带外 28–36 MHz 的强信号不会再折叠进频带边缘，能压下去约 78 dB。保持开启。
- 剩下的误报来自 20–26 MHz 通带内的噪声抬升。要解决，需要让本底随频率变化。

## 2026-11-16 · P4 第七步：校准前端噪声形状，按形状白化后再检测

**为什么不用局部分位数**：边缘的噪声抬升是前端硬件（AD9361 在 61.44 MSPS 下）的固定特性，和场景无关。如果按局部频段取 25% 分位数当本底，比窗口还宽的信号（十几 MHz 的 LTE、WiFi）会被当成本底，从而漏检。所以改为先校准一次噪声形状，检测时按形状白化、用全局本底判断。

**实现**：
- `WindowStage::set_noise_shape`：输入 [streams][M] 或 [M]。
- `floor_flag_kernel`：先用 p_k / shape_k 估计全局本底（仍是 25% 分位数），再把信道 k 的门限设为 floor × shape_k × 10^(dB/10)；相邻抑制仍按原始功率比较。
- `ClassifierConfig::noise_shape`。
- `curf_classify --calibrate-noise F`：输入端没有信号时运行，取每个信道前最多 4096 个窗口功率的中位数，再除以通带内信道的中位数。
- `curf_classify --noise-shape F`：加载形状文件。
- 测试：模拟边缘噪声逐渐抬高 10 dB。不用形状时，边缘噪声被大量误报；用形状后，只报出中心信号、比当地噪声高 10 dB 的边缘信号，以及 100 个信道宽的宽带信号；去掉形状后结果恢复原样。CUDA 测试 24 个、CPU 测试 16 个，全部通过。

**校准**（`tools/cal/omnicore_2450M_g30_fir64_shape.f32`，2450 MHz，增益 30 dB，RX FIR 开，TX 不发射，2807 个窗口）：

| \|f\|（MHz） | 0–10 | 10–20 | 20–22 | 22–24 | 24–26 | 26–28 | 28–31 |
|---|---|---|---|---|---|---|---|
| shape（dB） | −0.4 | +0.4 | +5.6 | +7.9 | +9.2 | +4.6 | −13.7 |

直流信道是 +4.3 dB，也一并被吸收进形状。形状和增益、采样率、FIR 有关，换了配置要重新校准。

**回环**（FIR 开，`p4_loopback_shape`）：

| 配置 | 衰减 70（SNR 约 7 dB） | 衰减 60（约 15 dB） | 衰减 50（约 24 dB） |
|---|---|---|---|
| 无形状，6 dB，52 MHz | 召回 0.632 · 误报 78 | 0.992 · 71 | 1.000 · 35 |
| 有形状，6 dB，52 MHz | 0.471 · 23 | 0.861 · 21 | 1.000 · 17 |
| **有形状，4 dB，60 MHz** | **0.687 · 28** | 0.926 · 28 | 1.000 · 18 |

（误报单位：个/窗口。）三种配置的准确率依次为：70 dB 0.558 / 0.699 / 0.708；60 dB 0.836 / 0.871 / 0.848；50 dB 0.915 / 0.909 / 0.910。

- 误报降了一半到三分之二。白化后频带边缘不再有问题，屏蔽可以放宽到 60 MHz。
- 召回的下降全部集中在边缘。以衰减 60 dB、6 dB 门限为例，20–23 MHz 从 0.977 降到 0.632，23–25.5 MHz 从 0.973 降到 0。
- 那里的噪声本来就高 8–9 dB，FIR 过渡带（24 MHz 起）还会再衰减信号，所以这些信号的真实 SNR 低于门限。以前能检出，是因为全局本底被低估了，本质上是同一种误判。
- 门限降到 4 dB 后，SNR 约 7 dB 的那档召回比无形状时还高（0.687 对 0.632），误报只有三分之一。
- 推荐的实采配置：`--noise-shape <校准文件> --threshold-db 4 --band-mhz 60`。

**剩下的误报**：每窗口约 18–28 个，主要来自外部 WiFi（2437 MHz 的 6 信道、2462 MHz 的 11 信道）漏进 RX 那一侧。实时界面上能看到这两处的横向突发，并且都被识别为 ofdm。它们是真实信号，只是不在发射标签里。

## 2026-11-16 · P4 第八步：界面的跟踪与可读性

**问题**（用户反馈）：标签跳得厉害，识别数量也一直在变，而且看不出每个信号是什么调制。

**原因**：
- 界面每 34 ms 一行，每行只取一个窗口的识别结果，而单窗口准确率约 91%，所以每个信号大约每 10 行就会错一次。
- 图例统计的是每一行的信道数：突发信号时有时无，宽带信号又占好几个信道，这个数自然一直跳。
- 104 个信号挤在一起，标签条上没地方写字；bpsk、qpsk、8psk 的颜色也太接近。

**投票的收益**（回环数据，每个信号在不同时间跨度内按概率加权投票）：

| TX 衰减 | 单窗口（4 ms） | 34 ms | 128 ms | 512 ms |
|---|---|---|---|---|
| 50 dB | 0.910 | 0.959 | 0.962 | 0.963 |
| 60 dB | 0.848 | 0.916 | 0.926 | 0.923 |
| 70 dB | 0.708 | 0.787 | 0.796 | 0.780 |

超过 128 ms 后基本不再提高。剩下的错误是个别信号持续性的混淆（主要是 16/64QAM），投票消除不了。

**改动**：
- **feed**（`curf_classify --ui-out`）：每行改为对这一行的 8 个窗口按概率加权投票，输出 `[stream, channel, class, 得票占比 %, 检测到的窗口数]`。
- **界面**（`tools/ui/index.html`）：
  - **信道跟踪**：每个信道维护各类别得分的指数衰减和（时间常数可选 0.15 / 0.5 / 2 s，默认 0.5 s），显示得分最高的类别和它的占比。
  - **出现和消失**：用检测率的指数平均加迟滞，高于 0.35 出现、低于 0.12 消失。
  - **合并**：相邻且类别相同的信道合并成一个信号，图例统计的是信号个数。
  - **信号列表**：按频率排序，列出频率、带宽、类别、置信度、SNR 和持续时间，4 Hz 刷新，点击一行会放大到该信号。
  - **放大**：在瀑布图上拖动可以放大频段，双击恢复全带宽。
  - **悬停提示**：显示频率、信道号、功率、高于本底多少、类别、置信度和持续时间。
  - **图例**：点击可以隐藏某一类。
  - **识别瀑布图**：只给已跟踪的信道着色，原始检测显示为淡灰色。
  - **配色**：9 类改用色相差别明显的一组颜色。

**实测**（实时回环，衰减 50 dB）：
- 已知信号的置信度稳定在 95–100%，持续时间连续增长，没有中断。
- 放大到 13 MHz 宽时，每个信号的类别名都能直接显示。
- 夹在已知信号之间的 ofdm 条目，是 WiFi 6 信道（2427–2447 MHz）被已知信号切成的几段，标成 ofdm 是对的。

**第二轮反馈**："标签还是一直在闪，看不清；而且好像只有 ofdm。"

原因：
- 全带宽时，160 多个信号挤在约 770 像素里，每个不到 5 像素。原来的标签条从左往右"放得下就写一个"，结果大约 10 个信号里只有 1 个能显示标签。
- WiFi 碎片时有时无，每一行抢到位置的信号都不一样，所以会闪；而 WiFi 碎片数量多，抢到位置的往往是它们，看起来就"全是 ofdm"。

实际上识别结果和发射标签一致，ofdm 多出来的约 55 个都是外部 WiFi 的碎片。

改动：
- **区分突发信号**：最近各行里检测率低于 80% 的信号标为"突发"（WiFi 这类），在颜色条和列表里画成半透明。新增开关"只看持续信号"，默认开启。开启时，图例与发射标签对得上：qpsk 15/15、fm 13/13、64qam 15/13、ofdm 15/17、8psk 9/8 等。
- **标签排布**：每个信号都有一根类别色的颜色条。类名分两行放：先试第一行，放不下再试第二行，两行都放不下就不写。信号集合不变，排布就不变，所以不再闪。标注了多少个信号显示在面板标题上。
- **实测**：全带宽时 102 个信号标出 34 个；放大到约 11 MHz 宽时，23 个标出 22 个。隔几秒截两次图，标签完全相同。

## 2026-11-16 · P4 第九步：RX2 收真实空中信号

**改动**：
- `--rx 1|2|1,2`（`OmniCoreConfig::rx_mask`）可以单独选 RX2。启动信息和界面面板会标出用的是哪一路接收机。
- `DetectConfig::dc_guard` / `--dc-guard`：直流附近的这些信道一律不上报，omnicore 源默认 1，即只排除 channel 0。原因是 RX2 在增益 40 dB 下的本振直流残余比 RX1 校准时强，被判成了 fm。
- 新增测试：DC guard 只屏蔽 channel 0，紧挨着的 channel 1 照常上报。测试共 41 个（CUDA 25，CPU 16），全部通过。

**噪声形状随增益变化**：最开始在 RX2 上用 RX1 增益 30 dB 的形状，结果每个频段每窗口都检出 260–290 个信道，几乎全判成 ofdm。原因是增益更高时热噪声占主导，边缘抬升只有约 5 dB，而不是 9 dB；用增益 30 dB 的形状白化后，边缘被压得过低，25% 分位数取到边缘，本底偏低，于是中间的纯噪声大片超过门限。

RX1 接着不发射的 TX，相当于输入端端接，就用它在增益 40 dB 下重新校准，再把形状用到 RX2（同一颗芯片）：

| 形状 | 0–10 | 10–20 | 20–24 | 24–26 | 26–28 | 28–31 MHz |
|---|---|---|---|---|---|---|
| 2450 MHz，增益 30 | −0.4 | +0.4 | +6.6 | +9.2 | +4.6 | −13.7 |
| 950 MHz，增益 40 | −0.2 | +0.1 | +3.2 | +4.9 | +0.0 | −18.6 |
| 2450 MHz，增益 40 | −0.1 | +0.0 | +2.8 | +4.5 | −0.4 | −19.0 |

形状主要由增益决定，频率的影响很小。增益一变，就要重新校准。

**实测**（RX2 接天线，增益 40 dB，门限 4 dB，带宽 60 MHz，每个频段 5 秒）：
- **950 MHz**：934.5–943.5 MHz 一段 9 MHz 宽、比本底高 19 dB 的连续信号，判成 ofdm，一致率 0.99，是移动 900 MHz 的 LTE/NR ✓。954.8–958.5 MHz 一段较弱的 ofdm（+6 dB）也像 LTE/NR。945–968 MHz 还有几段只比本底高约 5 dB 的 ofdm，来源不确定：可能是很弱的真实信号，也可能是 RX2 和 RX1 的噪声形状略有差异。
- **2450 MHz**：WiFi 突发共 458 个信道，几乎全判成 ofdm ✓。另有两个常驻的窄带信号（2440.04、2457.56 MHz）判成 gfsk，来源不明。
- **433 MHz**：信号很少。415–417 MHz 有几段很弱的 ofdm（+5 dB）；439.96 MHz 一个窄带信号判成 gfsk。
- **98 MHz**：101–119 MHz 那段宽带平台在 RX2 上也有，判成 ofdm，像是板子或主机自己产生的干扰。91.04 MHz 的 FM 电台这次判成 ofdm（之前用 RX1、增益 60 dB 时判成 fm，置信度 0.89）。

**结论**：
- 宽带 OFDM 在真实环境里识别可靠。
- 真实的 FM 广播（±75 kHz 频偏，带立体声和 RDS）和 GSM 与合成训练数据差别较大，这是域差异。硬件伪影（直流残余、杂散、101–119 MHz 平台）也会被当成信号并分到某一类。
- 改进方向：
  - 用真实采集的数据微调模型（P4 原计划里"必要时用回环数据微调"）；
  - 查清 101–119 MHz 平台的来源；
  - 给识别结果加一个"未知"类别：置信度低或像噪声的不强行归类。

## 2026-11-16 · P5 方向：ADS-B 解码、数据集重做（GNSS 延后）

**run2 实采复查**（2450 MHz，RX1+RX2，300 秒，零丢样，p99 2.61 ms）：
- 8psk 几乎消失（每路接收机 < 0.01%）。
- 但 bpsk 占了 26–33%，主要落在 WiFi 频段内部（2453–2470 MHz）和 BLE 广播信道（2426、2480 MHz）。峰值 −17 dBFS，不是削波。
- 结论：合成数据里的 wide 类和窄带类分布与真实信号不对应，模型在真实宽带切片上会随意落到某个窄带类。逐信道补丁不能解决，数据集要重做。

**设计决定**：
1. **检测网格和分类输入分开**：信道化器只负责检测（信号的中心频率、带宽、起止时间）。分类时按信号把它从原始 61.44 MS/s 数据里下变频、重采样到与其带宽成比例的速率，再送入一个覆盖所有宽度的模型。wide 不再是调制类别，而是带宽属性（次级标签）。
2. **标签体系**：主标签是调制族；次级是带宽和突发；协议名由规则推断（例如 OFDM + 20 MHz + WiFi 信道中心，推断为 WiFi）。另设 unknown 类，用 outlier exposure 训练。
3. **仿真数据**：用开源项目 [OnAir](https://github.com/adswill/OnAir) 的发射生成器产生符合标准的协议信号（WiFi、BLE、Zigbee、LoRa、DMR、ADS-B、LTE 等），再由我们自己的管线叠加信道损伤和 AD9361 接收链路特性。**OnAir 是 GPL-3.0（另有署名条款），只作为外部工具运行，生成数据，不复制代码进本仓库。**
4. **真实数据**：RX2 接天线，在各频段直接录 SigMF。能用 OnAir 解码器解出来的（BLE 广播、ADS-B、WiFi 信标等），据此自动标注；最终准确率以实采测试集为准。暂不使用专门的发射设备。
5. **协议插件**：ADS-B 先做。自研 CUDA 解调（下变频 + 低通抽取 + 脉冲匹配 + 前导检测 + 逐比特判决 + CRC-24 与单比特纠错），作为分类管线的旁路 tap，在同一路原始数据上运行；主机端做字段解码与 CPR 定位，界面显示飞机列表。
6. **GNSS 多系统捕获**（北斗 B1I、GPS/Galileo L1、GLONASS L1 同时）列为后续计划，需要有源天线和偏置供电。

## 2026-11-16 · P5 第一步：ADS-B 旁路解调

**实现**：
- `cuda::AdsbDemod`（`src/kernels/adsb.cu`）每块数据依次运行：
  - 解包后用 32 位相位累加器混频（按采样时间相位连续）；
  - 64 抽头低通（±1.5 MHz），4 倍抽取到 15.36 MS/s，再取幅度；
  - 0.5 µs 匹配滤波；
  - 前导检测：4 个脉冲分别比 12 个静默时隙均值和噪声高 threshold 倍，并取局部最大；
  - PPM 判决，CRC-24，DF17/18 用综合征表纠正单比特错误。
- 保留上一块最后 2048 个抽取后样本，每个起点在相邻块中恰好处理一次，跨块边界的帧也不会漏。
- 主机端 `curf/adsb.hpp`：字段解码（呼号、气压/GNSS 高度、地速/空速、航向、升降率、应答码），CPR 全局/本地定位，`Tracker` 检查 DF0/4/5/16/20/21 的地址（只接受近期在 DF11/17/18 中出现过的地址）。
- 作为分类管线的 `BlockTap` 运行，和信道化、识别同时处理同一块原始数据。`curf_classify --adsb [HZ]`，消息以 `{"adsb":{...}}` 行写入显示 feed；界面新增飞机表和航迹图。
- 测试 46 个，全部通过。新增 5 个：
  - CPU：1090 MHz Riddle 的标准测试帧（CRC、呼号、高度、速度、CPR 全局/本地解码）；
  - GPU：5 帧（含跨块帧、单比特错误帧、−34 dBFS 弱帧）全部找到，时间误差 ±1 个采样；纯噪声 0 误报。

**实测**（RX2，中心 1100 MHz，增益 45 dB，门限 1.6）：
- 收到 781CA6 / **CCA4104**（国航），12150 ft，290 kt，下降 896 ft/min，应答码 5154；另有一次运行解出 3 个 CPR 位置。
- 信号只有 −43 到 −48 dBFS，飞机偶尔能收到。瓶颈是天线和室内位置，不是解调：3 分钟里收到 0 到 1 架飞机。
- 实时 180 秒，零丢样，p99 0.48 ms。

**修复了一个潜伏 bug**：
- 分类管线按上一窗口的活跃数预先提交推理，超出本窗口实际数量的条目，会在 `postprocess_kernel` 里拿活跃表中从未写过的位置做下标读 `power[id]`。
- 以前显存碰巧是 0，所以一直没出事；加上 ADS-B 的分配后，布局变了，实时运行约 47 秒就出现非法访问。compute-sanitizer 下因为运行变慢反而复现不了，最后用 `CUDA_LAUNCH_BLOCKING=1` 定位到这个 kernel。
- 修复：postprocess 和 gather 一样，按设备端计数截断。

## 2026-11-16 · P5 第二步：真实空中数据（第一批）

**工具**：
- `tools/capture_survey.py`：RX2 按频段录制 SigMF，每段 1 秒，61.44 MS/s。发现削波就降 10 dB 重录。
- 占用统计（原 `tools/sigmf_survey.py`，已并入 `capture_survey.py`：录完自动打印，`--survey-only` 只统计）：给出每段信号的中心、宽度、SNR 和占空比。
- 注意：`och rx --format sc16` 输出的是 12 位样本左移 4 位，满幅是 32768，不是 2048。

**第一批**（`~/dataset/raw/20261116_2104`，11 个频段，2.6 GB，增益 40 dB；2440 和 1840 MHz 削波，自动降到 30 dB）：

| 频段 | 实际看到的信号 |
|---|---|
| 2440 MHz | WiFi 突发，宽 7–11 MHz（95 分位），SNR 40 dB 以上，占空比 0.1 |
| 1840 MHz | LTE 下行，连续：约 15 MHz、5 MHz 各一，还有一个 2.4 MHz |
| 2140 MHz | LTE 5 MHz，连续 |
| 935 MHz | GSM 200 kHz 载波群、LTE 5 MHz、920.0 MHz 窄带常发信号 |
| 433 MHz | 3 个常发窄带载波（440.0、435.7、433.35 MHz） |
| 400 MHz | 两段约 500 kHz 的突发群（集群对讲） |
| 118–137 MHz | 航空波段 AM 话音突发，120–180 kHz，多个频点 |
| 98 MHz | FM 广播只有 95.6 MHz 一个较强（天线不适合 VHF） |
| 1100 / 3500 / 5785 MHz | 1 秒内无明显信号 / NR 很弱 / 无 |

**对数据集设计的含义**：
- 实际频谱以 **宽带 OFDM（WiFi 突发、LTE 连续）**、**GSM 的 GMSK**、**窄带 FM/AM 话音**、**数字集群（4FSK、π/4-DQPSK）**、**ISM 的 OOK/FSK** 为主。
- 教科书式的 16/64QAM、8PSK 单载波几乎看不到。
- 第一批仿真类别按此确定：OFDM（WiFi/LTE/NR 参数）、GMSK（GSM）、GFSK（BLE）、4FSK（DMR/PDT）、π/4-DQPSK（TETRA）、O-QPSK/DSSS（Zigbee）、chirp（LoRa）、OOK、AM、FM（含立体声/RDS）、PPM 脉冲（ADS-B）、噪声，再加 unknown。

## 2026-11-16 · P5 第三步：用 OnAir 生成协议级仿真信号

**安装**：
- OnAir 4.2.0 Linux portable 包，装在用户目录 `~/opt/onair-4.2.0`，未改动系统，未装 udev 规则。
- 远端主机直连 GitHub 下载会中断，所以在开发机下载，核对 SHA256 后再拷过去。

**隔离网络**：
- OnAir 的桌面程序会把解码出的 ADS-B、AIS 等数据上传到它的地图服务器。
- Ubuntu 24.04 的 AppArmor 禁止非特权网络命名空间（`unshare -rn` 失败），所以改用 `~/opt/onair-4.2.0/libnonet.so`：一个 LD_PRELOAD 拦截库，只放行回环和 Unix socket，拦截所有对外的 connect、sendto、sendmsg 和域名解析，并记日志。
- 所有调用都经过 `onair-offline` 包装。实际运行时 CLI 从未尝试外连。

**许可证边界**：
- OnAir 是 GPL-3.0-or-later，另有署名条款。我们只把它当外部程序运行，用它生成数据，不复制、不链接它的代码。
- 生成的 IQ 是程序输出，不受 GPL 约束。仓库里只有调用它的脚本 `tools/onair_gen.py`。

**生成方式**：
- `onair-cli --synthetic --standard <id> --snr 60 --record <id>.cf32 --secs N` 用该模式自带的测试信号生成器，按模式自己的采样率录成 SigMF。
- 模式 ID 等于 `core/src/<folder>/<id>_rx.cpp` 的文件名前缀。
- **注意**：不认识的 ID 不会报错，而是悄悄退回一个 9.143 MS/s 的默认信号（cell_lte、cdma、chv、is95、pmr 都是这样）。`tools/onair_gen.py` 会统计每个信号的 99% 占用带宽和占空比，靠这个能发现这种情况。

**结果**：37 种模式，每种 2 秒，共 3.1 GB，在 `~/dataset/onair`（`index.json` 记录调制族、采样率、带宽、占空比）。

| 调制族 | 模式（99% 占用带宽） |
|---|---|
| OFDM | wifi 18.7 MHz 突发，cell（LTE 10 MHz）8.9 MHz，nr 15.9 MHz，dtmb 7.5 MHz，dab 1.53 MHz，halow 3.0 MHz，nbiot 316 kHz |
| GMSK / GFSK / CPFSK | gsmr 1.8 MHz（多载波），ais 60 kHz（双信道），ble 1.05 MHz，btc 12.7 MHz（跳频），nrf24 0.91 MHz，zwave，ant，uat 1.14 MHz |
| 4FSK | dmr 8.8 kHz，dpmr 4.9 kHz，nxdn 9.3 kHz，p25 7.8 kHz，m17 10.3 kHz |
| PSK 类 | tetra（π/4-DQPSK）21.5 kHz，vdl2（D8PSK），dvbc（单载波 QAM）7.3 MHz |
| 扩频 | zigbee（O-QPSK）3.8 MHz，umts 6.3 MHz（偏宽，待核对），cdma2000 1.31 MHz |
| chirp | loraraw 478 kHz（BW 500k），lorawan |
| OOK / FSK | sensors 39 kHz，ism，pager（POCSAG/FLEX） |
| 模拟 | airband（AM，多信道），fm（广播立体声 + RDS），frs（窄带 FM）14.6 kHz，acars（AM-MSK） |
| 脉冲 | adsb 2.9 MHz（占空比 0.02），modeac |

**交叉验证 ADS-B 解码**：
- 用 OnAir 生成的 ADS-B 文件（4 MS/s）跑我们的 GPU 解码器，解出 11 架飞机。ICAO、呼号、高度、速度和 OnAir 一致；CPR 位置一致到 1e-4 度（如 84ED3A：25.7961, 53.0293）。
- 漏了 DLH44，有几架的消息数也比 OnAir 少：4 MS/s 时每个脉冲只有 2 个采样，而解调器是按 15 MS/s 设计的。实采走 61.44 MS/s，没有这个问题。
- 为此加了两样东西：`--adsb` 按采样率自动选择抽取倍数；`tools/make_scenario.py --convert` 把 SigMF/cf32/cs16 转成 packed12，供 `--source file` 使用（原 `tools/iq_to_wire.py`，已合并）。

**局限与下一步**：
- 每种模式目前只用了默认参数，例如 LTE 只有 10 MHz 一种带宽。要增加多样性，需要用 `--sopt k v` / `--sval k v` 调每个生成器的参数（见各模式的 `<mode>_gen.h`）。
- 多信道场景（airband、ais、gsmr、lorawan）里同时有多个发射源，构建数据集时要先用我们的检测器把单个信号切出来。
- 下一步写 `tools/make_dataset.py`：重采样到与信号带宽成比例的速率；放进 61.44 MS/s 宽带场景；叠加频偏、相位噪声、多径、AD9361 接收链路（RX FIR、12 位量化、实测噪声形状）和相邻信号；按“调制族 + 带宽 + 突发”生成标签。真实测试集来自 RX2 实采，用 OnAir 解码器（离线）辅助标注。

## 2026-11-16 · P5 第四步：第二级（信号级）数据集

**做法**：`model/make_dataset.py` + `model/curf_model/widesig.py`。不在 61.44 MS/s 宽带场景里合成，而是直接模拟运行时“按信号提取”之后的样子：第二级模型看到的是被提取出的信号，只要提取的误差和损伤模拟对了，两种做法等价，而后者快得多。

1. **切分发射**（`emissions()`）：对每个 OnAir 录音做 STFT（采样率 ≤ 2.5 MS/s 时用 4096 点，否则 1024 点），平滑后取阈值和连通域，每个突发、载波或跳频就是一个发射。为了不出错，加了几条处理：
   - 阈值同时要求高于底噪 20 dB、并且在最强格子的 45 dB 以内，否则窗函数旁瓣会把多信道场景里的所有发射连成一片；
   - 先在时间和频率上平滑 5×5，再取阈值，否则 LTE、NR、DAB 的测试信号空资源多，一个载波会被切成上千个碎片；脉冲类只在频率方向平滑，因为 ADS-B 帧比一个 STFT 帧还短；
   - 峰值比最强发射低 25 dB 以上的分量丢掉；
   - 带宽在分量峰值周围的连续区域内测 99% 占用带宽，不受掩码边界限制，否则弱发射的带宽会被低估；
   - 中心和宽度相近、时间上相连的碎片合并；贴着更宽发射的窄碎片并入那个发射。
   - 结果（每种模式的发射数和中位带宽）：wifi 743 个突发，18.7 MHz；cell 1 个，8.96 MHz；nr 15.96 MHz；umts 6.26 MHz；dtmb 7.55 MHz；gsmr 2 个载波，249 kHz；ble 570 个，1.04 MHz；dmr 8.8 kHz；p25 7.8 kHz；tetra 22 kHz；lorawan 128 kHz；adsb 81 帧，1.5–2.9 MHz；atv 4.6 MHz。
2. **提取**（`extract()`）：模拟运行时的估计误差。带宽估计乘以 U(0.7, 1.4)（对数均匀），中心偏差 ±10% B。混频到 0 Hz 后，FFT 重采样到 fs = 2B（不低于 8 kHz）。窗口落在与发射重叠的任意位置：突发比窗口短时，整个突发都在窗口内。
3. **损伤**（`impair()`）：
   - 多径：指数功率时延谱，均方根时延扩展最大 1 µs，只对宽信号起作用；
   - 残余频偏 ±2% fs，相位噪声（随机游走，线宽 0–100 Hz）；
   - IQ 不平衡 ±0.5 dB / ±3°，直流偏置；
   - 带内 SNR 在 −5 到 30 dB 之间均匀取值；
   - 最后归一化到单位 RMS。
4. **类别**（13 个）：noise、ofdm、fsk（含 GMSK/GFSK/CPFSK/2FSK）、4fsk、psk（π/4-DQPSK、D8PSK、QPSK、BPSK）、qam、dsss（O-QPSK、W-CDMA、CDMA2000）、chirp、ook、am（含 ACARS）、fm、pulse、unknown。
   - unknown 用离群模式训练：模拟电视（VSB）、各种 MFSK（Horus、JS8、FSQ、IFKP、FT8）、RPR、选呼音。
   - 49 种 OnAir 模式全部用上，每个类别内各模式平均分配样本，避免某一类被最宽或最忙的协议主导。
5. **划分**：按时间切分，每个录音的前 75% 用于训练、后 25% 用于验证。突发全落在一侧的录音改为按突发划分。
6. **输出**：`x [N, 2, 1024]` float16（和第一级模型的输入布局相同），`y`，`log_bw`（带宽估计，作为附加输入），`snr`，`mode`；另有 `meta.json` 和 `emissions.json`。

**检查**：高 SNR 下，各类别 80–99% 的功率落在归一化频带中间的一半（纯噪声为 50%）。示例时频图里，HaLow 的 OFDM 符号块、M17 的 4 个频率电平、LoRa 的 chirp 都清楚可见。

**构建**：`~/dataset/ds_v1`，训练 130000 条（每类 10000）、验证 32500 条，共 0.67 GB；4 个进程，耗时 414 秒。

**可学习性检查**：现有的 ResNet-1D 结构，训练 6 轮（约 70 秒）。这只是检查数据集，不是第二级模型的正式训练。

| | 全部 | −5–0 dB | 0–10 dB | 10–20 dB | 20–30 dB |
|---|---|---|---|---|---|
| 只用 IQ | 0.794 | 0.404 | 0.769 | 0.872 | 0.884 |
| IQ + log10(带宽估计) | **0.839** | 0.569 | 0.831 | 0.896 | **0.908** |

- 带宽输入有用，chirp 准确率从 0.84 升到 0.97，fsk 从 0.69 升到 0.82。
- 主要混淆集中在**宽带类噪声信号**：ofdm、qam、dsss 之间互判 7–11%，ofdm 本身只有 0.58。最弱的模式是 wifi、halow、dab、dvbs（约 0.5）。原因是 1024 个点在 fs = 2B 下只包含几个 OFDM 符号，模型的感受野和训练量都不够。另外每个类别的来源较少：qam 只有 dvbc 一种，dsss 只有 3 种。
- pulse 判成 noise 13%：ADS-B 窗口里帧只占一部分时间。
- uat 0.42：它只有 3 个突发，验证集只用到最后一个。

## 2026-11-16 · P5 第五步：第二级模型第一版（验证集 92.0%，≥ 10 dB 95.1%）

**发现并修正了一个标签错误**：OnAir 的 `wifi` 默认生成的是 2.4 GHz **802.11b DSSS 信标**（DBPSK、Barker-11，11 Mchip/s），不是 OFDM，见 `wifi_gen.h`。上一版数据集把它标成了 ofdm，这就是 WiFi 只有 0.53、ofdm 和 dsss 互相混淆的主要原因。现在 `wifi` 标为 dsss；OFDM 的 WiFi 改用 `--sopt 0 1`（5 GHz）和 `--sopt 0 2`（6 GHz）生成，测得 16.4 MHz，符合 802.11a。

**生成器参数变体**：`tools/onair_gen.py` 新增 `VARIANTS`，同一个 OnAir 模式可以带不同参数生成，分别标注调制族：
- LTE 1.4 / 5 / 20 MHz（`cell --sopt 0 2/1/3`，测得 1.15 / 4.51 / 17.7 MHz）；
- DTMB 16QAM；
- DVB-S QPSK 1 Msym/s、DVB-S2 8PSK 2 Msym/s（psk）；
- DVB-S2 16APSK 3 Msym/s、32APSK（qam）。
- `--dvbc-qam` / `--dvbc-sr` 只作用于 DVB-C 接收端，对测试信号没有影响，所以没有采用。

合计 59 种模式和变体。

**模型** `SignalResNet1D`（`model/curf_model/net.py`）：
- 351k 参数，90.9 MFLOP/窗口；
- 只用引擎已经支持的算子：带宽估计作为第三个输入通道（常数 (log10 B − 5)/2），第一层卷积把它当作随带宽变化的偏置；
- 比第一级多一级步长 2，感受野约 250 个采样。

**训练**：`model/train_stage2.py`。整个数据集以 fp16 放在显存里；每批随机旋转载波相位、随机做频谱翻转（共轭）；label smoothing 0.05；OneCycle 学习率，40 轮，约 14 分钟。

**修正突发起止时间**：切分时的时间平滑会把发射的起止各撑开约 2 帧。WiFi OFDM 突发只有 0.1–1 ms，而提取窗口只有 31 µs，结果很多窗口落在静默上却被标成信号。现在用未平滑的带内功率重新确定起止帧。

| | 全部 | −5–0 dB | 0–10 dB | 10–20 dB | 20–30 dB | ≥ 10 dB |
|---|---|---|---|---|---|---|
| 快速检查（ds_v1，小模型，6 轮） | 0.839 | 0.569 | 0.831 | 0.896 | 0.908 | — |
| s2_run1（ds_v2：修正 WiFi 标签、加变体） | 0.905 | 0.772 | 0.910 | 0.938 | 0.948 | 0.932 |
| **s2_run2**（ds_v3：再修正突发起止） | **0.920** | 0.773 | 0.927 | 0.955 | 0.959 | **0.951** |

各类别准确率（s2_run2）：noise 0.91、ofdm 0.86、fsk 0.92、4fsk 0.92、psk 0.90、qam 0.93、dsss 0.88、chirp 0.99、ook 1.00、am 0.96、fm 0.86、pulse 0.86、unknown 0.97。

剩下的混淆（≥ 10 dB）：
- 4fsk→fsk 9%：NXDN 只有 0.56，它的 4FSK 频偏小，和低频偏的 2 电平 GFSK 很接近；
- fm→unknown 12%：FRS 窄带 FM 话音和 MFSK 音调类信号容易混；
- pulse→noise 7%：短帧占窗口比例小。

模型存放在 `model/artifacts/s2_run2/`（远端，`best.pt`、`metrics.json`、`train.log`）。

## 2026-11-17 · P5 第六步：实采测试集（第一版）

**工具**：`model/real_testset.py`，分 scan / label / cut / build / eval 五步，输入是 RX2 的 11 个频段、每段 1 秒的录音。

**切分**：实采数据和仿真数据差别很大，前后改了 4 轮：
1. **按噪声谱白化**：底噪不平坦，单一阈值会在 5785 MHz 切出上百个约 16 MHz 宽的假信号。新增 `capture_survey.py --gains-from`：用 RX1（接着不发射的 TX，相当于输入端接负载）在同频点、同增益下录噪声参考，再用 `noise_spectrum()` 白化。
   - **更正**：当时以为这是“本振附近的隆起”，用户怀疑是直流校正，实测两者都不是（见下一节）。真正原因是 `och rx` 沿用了设备上 18 MHz 的模拟带宽，±9 MHz 以外被滤掉了。
2. **时间合并**：低负载的 LTE 只有部分子帧有数据，会被切成相隔几 ms 的碎片，合并间隔放宽到 12 ms。
3. **窄带二次扫描**：在 240 kHz 网格上各做一次 960 kS/s 的窄带流，再细扫，每个网格只做一次。
4. **稳态载波**：先对整段录音求平均谱，持续存在的载波（FM、LTE、杂散）各作为一个覆盖全程的完整发射；和它在频率上大部分重叠的突发片段丢掉。原因是 FM 的瞬时频谱会随音频漂移，按短时谱切会被切成很多中心不同的小段。另外，VHF 频段有大量 10–18 MHz 宽的脉冲干扰（就是之前记录的 101–119 MHz 宽带平台），实采数据要关闭“窄发射并入覆盖它的宽发射”这条规则，否则 FM 和 AM 会被当作干扰的一部分。

**标注**：按频段规划和实测特征给候选标签，只有 high 和 confirmed 计入准确率：
- FM 广播：87.5–108 MHz，80–300 kHz；
- 1.8 GHz LTE 和 3.5 GHz NR：OFDM（中国这两个频段没有 UMTS）；
- 5 GHz WiFi：只有 OFDM；
- 2.4 GHz WiFi：按频谱形状判断。中间与两肩功率比在 ±2 dB 内为 OFDM（平顶），+8 到 +13 dB 为 802.11b DSSS（带发射滤波后中间更突出）；
- 900 MHz 和 2.1 GHz 的宽信号用 OnAir 解码器（离线）确认：939 MHz 约 9 MHz 宽的信号，cell 解码器识别出了 1 个 LTE 小区；2.1 GHz 的 LTE 和 UMTS 解码器都没解出（可能是 NR）。解码器还会输出网络名和地址，这些都不保存，只记录数量。
- **一处纠错**：133.31–133.36 MHz 那一串一开始按规则标成了 AM，实际是一排间隔约 11 kHz、宽 1.4–2.6 kHz 的常发窄载波（杂散），模型判成 unknown 其实是对的。AM 规则已收紧为 4–25 kHz、且必须是间断的。这 1 秒里没有录到真正的航空话音。

**结果**（s2_run2；1327 个有标签窗口，38 个发射机）：

| 类别 | 窗口准确率 | 主要错成 |
|---|---|---|
| noise | 0.81 | fsk、pulse |
| ofdm | 0.66 | noise、qam、pulse（弱 NR、部分 LTE） |
| dsss（2.4 GHz 802.11b） | 0.83 | ofdm |
| fm（95.6 MHz 广播） | **0.02** | dsss |
| **合计** | **0.651** | 按发射机投票 0.579 |

对比仿真验证集的 92%，差距主要来自三处：
1. **FM 域差异**：训练里只有一个 OnAir 的干净立体声 FM 源；真实电台 SNR 约 17 dB、有多径，在 2B 采样下看起来像类噪声的扩频信号。需要更多样的 FM 仿真（频偏、单声道/立体声、节目内容、多径），并用实采 FM 微调。
2. **弱信号切分**：3.5 GHz 的 NR 只比底噪高约 10 dB，仍会被切成小块，模型只看到一部分。
3. **覆盖面**：每个频段只录了 1 秒，能可靠标注的只有 4 类。需要录更长的时间，并为 VHF 换更合适的天线。

## 2026-11-17 · 底噪“隆起”的真正原因：采集时模拟带宽只有 18 MHz

用户怀疑是直流校正的问题，于是关掉校正采几帧对比（5785 MHz，增益 40 dB，每次 0.3 秒；数值是相对 12–18 MHz 平均噪声的 dB）：

| 设置 | ±30 kHz | ±0.3 MHz | ±1 MHz | ±5 MHz | ±10 MHz | DC 频点 |
|---|---|---|---|---|---|---|
| RX1，直流 + IQ 校正开（默认） | +16.3 | +10.0 | +9.1 | +8.2 | +5.3 | +19.3 |
| RX1，直流校正关 | +23.0 | +10.3 | +9.4 | +8.5 | +5.5 | +28.5 |
| RX1，直流 + IQ 校正都关 | +31.1 | +10.3 | +9.4 | +8.6 | +5.5 | +37.0 |

- 直流和 IQ 校正只影响正中心 ±30 kHz 以内，宽约 ±10 MHz 的平台在三种设置下不变。
- 这个平台在 1100 MHz 和 5785 MHz、增益 20/40/60 dB 下都存在，和频率、增益基本无关。
- 原因：`och get rx.rf-bandwidth` 返回 **18000000**。`och rx` 不传 `--bandwidth` 时会沿用设备上的旧值，12–18 MHz 的参考区落在滤波器过渡带上，于是看起来中心“鼓起”了 8–9 dB。
- 设成 56 MHz 重测：RX1 在 1100 MHz 和 5785 MHz 下，1–15 MHz 范围平坦到 ±0.6 dB；中心 ±30 kHz 剩下的 +4 到 +7 dB 是本振泄漏，由直流校正处理。

**影响范围**：
- 实时管线不受影响：`OmniCoreSource` 会把带宽设成 min(采样率, 56 MHz)。
- 受影响的是用 `och rx` 录的数据：`~/dataset/raw/20261116_2104` 这批实采和它的噪声参考，±9 MHz 以外的信号都被衰减了，比如 5768 MHz 的 WiFi（−17 MHz）和 1847 MHz 的 LTE（+7 MHz 及以外）。
- `capture_survey.py` 现在固定传 `--bandwidth 56M`，并把带宽写进元数据。设备当前的带宽也已经是 56 MHz。

**另一个发现**：2440 MHz 时 RX1 中心 ±1 MHz 高出 36 dB，说明 RX1 在 2.4 GHz 并不是真的接了负载，附近的 WiFi 能漏进来。2.4 GHz 的噪声参考需要真正的 50 Ω 负载。

**下一步**：
- 用 56 MHz 带宽重录实采数据集（2.4 GHz 的噪声参考要接 50 Ω 负载），突发频段录 10 秒以上；
- FM 和蜂窝信号的仿真增加多样性；用部分实采数据微调，另一部分留作测试；
- 突发频段（2.4 GHz、400 MHz、航空波段）录 10 秒以上；
- 2.1 GHz 加一个 NR 解码器来确认。

**下一步**：
- 实采测试集：RX2 录音按同样方式切分发射，用 OnAir 解码器离线辅助标注、人工核对。这才是真正要报的准确率；
- 运行时的 CUDA 提取 kernel（按检测结果混频、重采样到 2B、切窗）和第二级引擎接入；
- 导出：`convert.py` 支持 3 通道输入后，第二级模型即可进入 INT8 引擎。

## 2026-11-17 · P6 计划：演示版 v1（信号级识别 + 可信的显示）

### 目标与原则

v1 的目的是**演示**，不是把模型调到最好，模型性能以后再慢慢提升。

- 显示出来的标签要可信：宁可显示“未识别”，也不要出现不可能的信号（例如 2.4 GHz 里出现 FM、8PSK）。误报一多，观众就分不清是硬件的问题还是模型的问题，反而影响对硬件的印象。
- 每个信号只给一个标签，不论宽窄：显示“调制族 + 带宽 + 协议推测”，不再让一个宽信号被切成很多信道、各判各的。
- 实测数据暂停：v1 只用仿真数据训练，实采只做定性检查。实采测试集等换了天线、准备好 50 Ω 负载后再继续（见 P5 第六步、底噪一节）。

### 演示场景

| 场景 | 内容 | 状态 |
|---|---|---|
| 室内 | 2.4 GHz WiFi（OFDM/DSSS）、蓝牙/BLE、5.8 GHz WiFi、4G LTE（1.8 GHz 或 900 MHz，看现场哪个信号强） | v1 主场景 |
| 飞机 | 1090 MHz ADS-B：飞机表 + 地图，只显示 CRC 校验通过的结果 | 已可用，v1 联动显示 |
| 室外 | FM 广播频道 | 换 VHF 天线后再做 |

### 架构：模型和显示策略分开

```
原始 61.44 MS/s ─┬─ 信道化 → 逐信道能量检测（现有）
                 │        ↓
                 │   相邻活跃信道合并成信号：中心、带宽、起止（从界面的 JS 移到 C++）
                 │        ↓
                 └─ 按信号提取（新 CUDA kernel，FFT 滤波器组）：混频 → 重采样到 2B → 1024 点窗口
                          ↓
                    第二级模型（同一 INT8 引擎，3 通道输入）→ 13 个调制族的概率
                          ↓
                    显示策略：频段先验 → 置信门限 → 时间滞回 → 协议规则 → 杂散表
                          ↓
                    feed 和界面（演示视图 / 工程视图）
```

**按信号提取**：
- 每块做一次大 FFT，每个信号取 [fc ± B] 的频点，按尺寸分组批量做 IFFT（重叠保留）。宽度任意，计算量和信号数、带宽成正比。
- 窄信号要二次估计：先按信道宽度粗提取，用 FFT 测 99% 占用带宽，再精提取。
- GPU 上保留约 100 ms 的原始数据环形缓冲（packed12 约 18 MB）。窄信号识别需要几十 ms 的数据，宽信号是毫秒级。

**显示策略**（`curf_classify` 或 ui_server 里实现，参数可配置）：
1. **频段先验**：每个频段只允许合理的类别，模型输出按允许集合重新归一化。

   | 频段 | 允许 |
   |---|---|
   | 2400–2483.5 MHz | OFDM / DSSS（WiFi）、FSK（蓝牙、BLE）、DSSS（Zigbee） |
   | 5150–5850 MHz | OFDM（WiFi） |
   | 国内蜂窝下行 B1/B3/B8/B34/B39/B40/B41、n41/n78/n79 | OFDM（LTE/NR）、FSK（GSM） |
   | 87.5–108 MHz | FM |
   | 1090 MHz | 只显示 ADS-B 解码结果 |
   | 其他 | 全部类别，门限更高 |
2. **置信门限**：允许集合内的最高概率低于门限、或和第二名差距太小时，显示“未识别信号 · 带宽”。
3. **时间滞回**：同一信号连续几次判断一致才显示，之后有滞回；BLE 这类短突发按频点在几秒内累积投票。
4. **协议规则**：OFDM + 16–20 MHz + 2.4/5 GHz 突发 → WiFi；FSK + 约 1 MHz + 短突发 → 蓝牙/BLE；OFDM + 1.4–20 MHz + 连续 + 蜂窝频段 → 4G LTE，n41/n78/n79 → 5G NR。
5. **硬件杂散表**：用一次接 50 Ω 负载的校准录音，记下常驻的直流残余、杂散梳状谱、VHF 宽带干扰；演示视图里不显示，或标为“本机杂散”。
6. **工程视图开关**：显示模型原始输出、逐信道结果和置信度。答疑时可以把硬件层（频谱、零丢样、SNR）和识别层分开讲。

演示时主动说明“识别结合了频段规划”，这是行业惯例。

### 里程碑

| | 内容 | 验收 |
|---|---|---|
| M1 | 数据集只补演示相关的种类：WiFi（OFDM/DSSS 多参数）、BLE/经典蓝牙、LTE 各带宽、NR（不同子载波间隔和带宽）、GSM、Zigbee、FM（多种频偏、单声道/立体声）；损伤加入 RX FIR、12 位量化、直流残余、低 SNR 和多径。重训第二级；`convert.py` 支持 3 通道输入，INT8 校准，引擎加载 | 仿真验证集上演示相关类别的准确率 |
| M2 | 信号级链路：信道合并成信号、FFT 滤波器组提取（含窄带二次估计）、第二级接入管线 | 实时零丢样，延迟分布（宽、窄分开统计） |
| M3 | 显示策略 1–6 和工程视图；feed 新增信号级记录；ADS-B 结果在 1090 MHz 联动显示 | — |
| M4 | 仿真室内场景生成器（WiFi + BLE + LTE + 噪声，按随机频点、功率、时间放进 61.44 MS/s）→ `--source file` 端到端 | **主指标：显示出来的标签中错误的比例 < 2%**；“未识别”不算错，单独统计占比 |
| M5 | 室内实时试跑（2.4/5.8 GHz、一个 4G 频段、1090 MHz），长测，录演示视频 | 定性检查，加上已确认过的信号：5.8 GHz WiFi、939 MHz LTE、2.4 GHz 802.11b |

### 2026-11-17 · M1 完成：第二级模型 s2_run3，进入自研引擎

**数据**（ds_v4，13 类，每类训练 10000 条）：
- OnAir 新增变体：NR n28（15 kHz 子载波间隔，3.8 MHz）、GSM 单小区和三小区（其中一个弱 12 dB）。
- OnAir 的 BLE 只能切换广播信道，调制不变，所以没加；FM 的频偏、单声道/立体声在 OnAir 里不能调。
- **自研 FM 生成器** `tools/synth_sources.py`（代码完全自写，不涉及 GPL）：
  - 广播 FM 12 份：音乐或语音类节目、50 µs 预加重、广播式压缩（软限幅，压缩强度随机）、单声道或立体声（19 kHz 导频、38 kHz 差信号）、可选 RDS、频偏 55–80 kHz；99% 占用带宽 99–134 kHz。
  - 窄带 FM 对讲 12 份：300–3400 Hz 语音、频偏 1.5–5 kHz、可选 CTCSS、按键发射；带宽 5–10 kHz。
  - 第一版没有压缩时，带宽只有 73–91 kHz，远小于实测电台的 180–220 kHz：真实电台的平均调制度很高。

**训练**：s2_run3，40 轮，13.5 分钟。

| | 全部 | 0–10 dB | 10–20 dB | 20–30 dB | ≥ 10 dB |
|---|---|---|---|---|---|
| s2_run2（ds_v3） | 0.920 | 0.927 | 0.955 | 0.959 | 0.951 |
| **s2_run3（ds_v4）** | **0.920** | 0.927 | 0.960 | 0.967 | **0.958** |

- 演示相关的模式（≥ 10 dB）：WiFi DSSS 0.98，5/6 GHz WiFi 0.99；BLE、经典蓝牙 1.00；nRF24 0.99；Zigbee 0.95；LTE 1.4/5/10/20 MHz 0.97/0.92/0.93/0.97；NR 1.00、n28 0.93；GSM 1.00；自研 FM 广播大多 1.00（最低 0.72）；FM 类整体 0.86 → 0.89。
- 有一个异常：nfm_8 为 0.00，可能是这份录音的验证段正好是没按键的静默期，待查。

**导出和引擎**：
- `model/export_stage2.py`：导出 ONNX（输入 [batch, 3, 1024]）、256 个参考窗口（每类 20 个）和 metrics.json；然后用 `convert.py --name s2` 转换。PyTorch 的 fp32 和 fp16 在整个验证集上 top-1 一致率 0.9996。
- `convert.py`、模型加载、golden 格式都是通用的，3 通道不用改。INT8 的“stem 融合进 block1”kernel 写死了 2 通道，所以第二级走不融合 stem 的路径，也不能用 fp16 输入。第二级负载很低，这没有影响。
- `curf_bench_engine` 新增 `--name`、`--check`（跑 golden 窗口，报告和 fp32 的 top-1 一致率与 logit 误差）、`--no-tune`、`--no-reuse`。
- 引擎测试可以用环境变量 `CURF_TEST_MODEL_DIR` / `CURF_TEST_MODEL_NAME` 指向其他模型。融合后的启动次数和张量数这类结构断言只对第一级检查，数值断言对所有模型都检查。第二级：2273 个断言通过，1 个跳过（fp16 输入）。完整套件 46/46。

| 引擎路径（第二级，256 个参考窗口） | top-1 与 fp32 一致 | logit 相对误差 | 批量 256 吞吐 |
|---|---|---|---|
| FP16（wmma / mma，融合 block 和全局池化） | 100% | 1.3–2.1e-3 | 13.4 万窗口/秒 |
| INT8 | 99.2% | 0.13 | 19.2 万窗口/秒 |

**排查记录**：一开始 `--check` 只测出 78–96% 的一致率，而且每次结果都不同。查下来是检查代码自身的竞争，不是引擎的问题：同步 `cudaMemcpy` 从可分页内存拷贝时，可能在数据真正到达显存前就返回，而 bench 的计算跑在 non-blocking 流上，不会等它。改成在同一个流上用 `cudaMemcpyAsync` 后，一致率是 100%。排查过程中顺便确认了显存复用、CUDA Graph 和自动调优都没有问题。

**留给 M2 的一处不一致**：运行时用 FFT 滤波器组提取，每个信号取的频点数要从固定的几种 FFT 尺寸里选，所以实际采样率会在 2B 到 3B 之间；而训练时固定 fs = 2B（只有带宽估计在抖动）。M2 定下提取方案后，`extract()` 要按同样的分布采样，然后重训一次。

### 2026-11-17 · M2 第一小步：已知参数的提取正确性闭环

- 新增独立的 `cuda::SignalExtractor` overlap-save 基线和连续 CPU 直接卷积参考；输入是已知中心频率、整数抽取倍率的信号，输出未归一化 IQ，显式记录 FIR 群延迟和样点时间。
- `curf_check_extraction` 从 wire 打包、GPU 解包到提取和 IQ 导出；`tools/run_extraction_checks.py` 一键运行 10 组案例，包括宽/窄带、跨块突发、带外干扰、两路输入、packed12/signed8、非二次幂抽取、长历史和近 Nyquist 信号。
- 远端 RTX 3050 全部通过，最大逐点误差 3.16e-7（门限 3e-6）；完整回归 51/51 通过，Compute Sanitizer memcheck 为 0 errors。单音复现确认原独立 FFT 裁剪路径的块边界失真。
- 复现方式、参数契约、结果和接入限制见 `docs/m2-extraction.md`，完整数据在 `docs/results/m2_extraction.jsonl`。
- **M2 尚未完成**：现有 SignalStage 尚未替换提取器；此基线先固定整数抽取，原有 2B–3B 的有理数速率网格需要另行对齐；多信号实时性、检测/跟踪、第二级窗口和延迟统计待接入验证。

### 2026-11-17 · M2 信号级链路完成受控回放验收

**模型**：采用 `s2_run4`（ds_v5，结构不变）。在同一 ds_v5 验证集上，run3 → run4 的总准确率为 89.29% → 91.55%，≥10 dB 为 94.23% → 95.91%。已导出到远端 `model/artifacts/s2_run4/export`。默认 FP16，保留 `--stage2-int8`；模型概率不在 M2 内修改。

**链路**：
- `--stage2` 默认让第一级只检测，不再加载或运行逐信道模型；`--stage1-classify` 保留旧工程路径。
- 原始 packed ring 约 102 ms；每路共享 4×hop FFT，有限 FIR 重叠保留，按 W/batch 分组的小 IFFT，保留 run4 的 2B–3B 有理数速率网格。所有 plan 有界初始化，共用工作区。
- 窄带在粗候选区内精估中心和带宽，之后局部精扫；完整粗组件做载波关联，临时桥接的相邻载波按之前的间隔拆分。宽载波保持完整精扫范围，带宽使用按源时间计算的 2 秒包络，避免退化到 OFDM 局部资源块。
- 窄带 1024 点攒窗保留余量；绝对样点时钟维持相位连续；实际模型输入与连续 CPU 参考对齐，含归一化和第三个带宽通道。
- 新增短时间片峰值检测、PFB 启动屏蔽、文件实时限速、离线第二级背压和结束 flush。feed 包含源时间 `t0/t1`、处理延迟、观察窗长度；宽窄延迟分别统计。

**最终验收**（RTX 3050 Laptop，双路 61.44 MS/s，WiFi OFDM + BLE + LTE 5 MHz + 窄带 FM，FP16，300.001 秒受控文件回放）：

| 指标 | 结果 |
|---|---|
| 样点丢失 / 时间戳缺口 | 0 / 0 |
| 第二级 busy / ring / capacity 跳过 | 0 / 0 / 0 |
| 预期载波 / 轨迹跨载波 / 频带范围异常 | 8/8 全出现 / 0 / 0 |
| 宽带处理 p50 / p99 / max | 4.627 / 4.957 / 6.683 ms |
| 窄带处理 p50 / p99 / max | 4.621 / 4.953 / 6.665 ms |
| 第二级结果（宽 / 窄） | 61260 / 17574 |

处理延迟从最终原始 hop 进入 host ring 算起，不含模型观察窗的积累时间。此处为两路交织的仿真文件回放，**不是新链路的实采长测**。

**验证与复现**：远端 54/54 测试通过，Mac CPU 21 通过、1 因无模型跳过，Python 评分测试 4 个通过；滤波器组和检测到第二级集成路径 memcheck 0 errors。`tools/run_signal_checks.py` 一键生成场景、回放、评分和 300 秒验收；完整说明见 `docs/m2-pipeline.md`，数据在 `docs/results/m2_signal_300s.json` / `m2_signal_acceptance.json`。

**仍按阶段推进**：原始候选不做显示策略。有限场景 75 条原始记录中，61 条正确、1 条类别错误、13 条未匹配；未经筛选的错误/未匹配比例 18.67%。M3 的先验、置信门限、滞回、协议规则、杂散表和工程视图尚未完成，M4 的显示错误率 <2% 也尚未验收。下一步用这些带真值记录实现和检查 M3，同时报告“未识别”比例；M5 再做室内实采定性检查。

### 2026-11-17 · M3 软件链路与仿真接入检查

- 实时服务器与离线工具共用 `tools/signal_policy.py`：频段先验、原始置信度与允许集合总量门限、后验差值、源时间滞回、短突发按 RX / 频点累积投票。原始概率保持原值，单类别先验不能把弱输出放大为可信标签。参数在 `tools/policy/default.json`，可通过 JSON 覆盖。
- feed 写入完整 13 类概率、源时钟、活动能量占比、处理 / 观察时间；文件显式 `--freq` 也进入 RF 元数据。策略追加 `display`、原因与独立未识别统计。无完整概率的旧记录保持未识别。
- 演示使用完整载波标签，工程视图检查原始概率、先验总量、滞回原因、逐信道结果和被抑制候选。已有瀑布图、频率缩放继续使用；有界快照支持晚加入，循环重放重置状态，断流保留上次数据并提示重连。
- 协议按调制族 / RF / 带宽 / 活动证据推测；缺证据不猜。B41 / n41 显示 `LTE / NR（推测）`，不把同一频段直接当成 NR 识别。1090 MHz 模型标签让位于 CRC 通过的 DF17/18；元数据携带接收路，飞机与信号频带联动。
- 杂散表生成、接收设置匹配、标记 / 隐藏已实现。**实际 50 Ω 校准尚未采集**；没有把接天线或 RX 接 TX 线缆的录音当成负载参考，默认不启用任何实测杂散表。
- 三个独立 RF 场景（2440 MHz WiFi + BLE、5800 MHz WiFi、2135 MHz LTE）0.5 秒的真实 GPU / run4 检查，已命名分别 36 / 18 / 16 条，均匹配正确；未识别分别 4 / 2 / 14 条，占 10% / 10% / 46.7%。这是接入冒烟，**不代替 M4 <2% 验收**。
- ADS-B 双 RX 仿真回放：RX2、+10 MHz 偏移，25 条 CRC 通过，24 条输出位置；没有射频发射。2.4 GHz 额外 20.001 秒循环实时回放，1562 次二级分类，零丢样 / 缺口 / 二级跳过，处理 p99 4.758 ms。
- C++ 54/54，阶段二 memcheck 0 errors；Python 策略 / feed 22 个检查和原有评分 4 个通过。界面检查覆盖双视图、原始概率、逐信道详情、未识别、空筛选、ADS-B 联动及窄屏。详见 `docs/m3-display.md`；结果在 `docs/results/m3_*`。

**下一步**：M4 扩大随机场景与错误类别 / 噪声注入，分别报告显示标签错误率与未识别比例；50 Ω 杂散校准等硬件条件具备后补测。M5 实采尚未进行。

### 2026-11-17 · M4 随机室内仿真验收

M3 已分五批提交（`9e01ee5`、`78a14ae`、`51d47e7`、`d911b0f`、`073bd01`），无协作者署名。

- 新增分块场景生成器 `model/make_room_scene.py`：61.44 MS/s packed12，四个独立 RF 调谐配置，
  WiFi OFDM / DSSS、BLE、可选 Zigbee、LTE，随机信道 / 中心、功率、启停、频偏、弱双径、IQ 不平衡、
  直流残余及 12 位量化；双 RX 独立生成。另有纯噪声与 FM / PSK 先验外控制。
- 新增 `tools/run_room_checks.py` 和 `tools/room_evaluation.py`：真实 GPU / run4、`--source file --realtime --drop`
  端到端，按 RX 匹配真值，同时检查调制族与已显示的协议推测，错误不重复计数，未识别单独报告。
  冻结配置、代码、二进制与模型哈希，保存源快照、原始流、真值、feed、命令与运行计数。
- 开发中发现宽载波在低 SNR 下被分割、直流窄候选被高置信判为 FSK；增加提取窗 SNR、带宽及 WiFi
  信道栅格证据，进入置信门限 0.90、保持门限 0.85、提取窗 SNR ≥10 dB。原始模型概率仍完整保留。
- 首轮验收计划 104 场，完成 100 场后因窄带 FM 的验收切片没有发射而中断；已命名 1704 条、错误 21 条（1.23%），
  其中 2.4 GHz 错误为 19/808（2.35%），未通过逐频段门限。保留原始失败记录。
- 开发集复现了同类 WiFi DSSS 裙边误标，增加同 RX 近期宽候选的几何保护；40 场复核变为
  498 条已命名、0 错误，未识别 2952 条（85.57%），高 SNR 载波覆盖仍为 27/34。
  换用连续广播 FM 控制源，并在运行前检查全部可能源的切片；8 个新负控制均无命名输出且运行检查通过。
  使用 33001 起始的新种子重新冻结后，104 场验收全部完成：**1741 条已命名、4 条错误（0.23%）**，
  四频段分别为 4/747、0/261、0/383、0/350，全部 <2%。未识别 11257 条（86.61%）；
  ≥14 dB 的载波中 78/91 曾得到正确标签。8 个负控制没有命名输出，丢样 / 缺口 / 二级跳过 / ADC 削顶均为零。
  C++ 54/54、Python 41/41 通过，现有 M3 三频段与 ADS-B 接入复核通过。M4 显示错误率验收通过；
  详细口径与结果见 [M4 验收说明](docs/m4-evaluation.md)。

**边界**：开发 / 验收使用不同种子与不同源时间切片，但共享干净协议录音；模型窗彼此相关，不是独立统计试验。
本轮载波占用包络不重叠，不覆盖同频碰撞、完整射频链路或真实天线场景。50 Ω 杂散校准和 M5 实采仍待进行。
剩余 4 条错误集中在一个 2.4 GHz 场景的 DSSS / Zigbee 推测；几何歧义和高未识别比例仍需后续改进。

### 2026-11-18 · M5 实采长测与录像，定性验收仍有缺项

M4 已分四批提交（`ea9620c`、`341d0fd`、`73e50ed`、`acae5cc`），无协作者署名。

- OmniCoreSDR 实际 RX、61.44 MS/s，完成 2.4 GHz、5.8 GHz、939 MHz、1090 MHz ADS-B 和双 RX
  五场各 300 秒长测，全程不创建 TX 流。原始首轮结果保留；2.4 GHz 单 / 双路虽零丢样，却分别有
  77426 / 110176 次二级容量跳过，明确判失败。
- 扩展二级显式容量参数，按原有有界 IFFT 计划分块，分别统计请求 / 轨迹 / 工作区不足；默认容量不变。
  以 256 请求 / 512 轨迹 / 每 pending 128 MiB 工作区补跑三场各 300 秒。2.4 GHz 使用现场增益 10 dB、
  检测门限 24 dB；5.8 GHz 调谐至 5770 MHz、门限 18 dB。三场全部零驱动异常、零丢样 / 缺口、
  零二级 busy / ring / capacity 跳过，宽带处理 p99 为 7.06 / 6.85 / 7.47 ms。
  提高检测门限会漏掉弱候选，零跳过只证明这一配置组合；候选碎片化仍需改进。
- 5.8 GHz 实测约 5765 MHz、16.4 MHz OFDM / WiFi 推测；939 MHz 约 9 MHz OFDM / LTE 推测。
  ADS-B 收到 126 条 CRC 通过的 DF17/18、43 条位置消息、1 架飞机，可在实采回放地图检查。
  **2.4 GHz 未稳定显示宽 DSSS / 802.11b，完整 M5 定性验收未通过**；不把窄带 DSSS 或 FSK 标签当作该项完成。
- 新增有界实采检查与 Playwright 录像工具，录制演示 / 工程视图；界面明确区分“SDR 实采”和“SDR 实采回放”。
  汇编视频包含四频段及双 RX；ADS-B 片段为明确标注的 10× 实采 feed 回放。
- C++ 55/55、Python 44/44 通过；二级 Compute Sanitizer 为 6573 个断言、3 用例、0 errors。
  新容量配置复用原 M4 104 场波形回归：1734 条命名、3 错误（0.17%），逐频段均 <2%，
  未识别 11265 条（86.66%）。这是原验收集回归，不是新独立准确率测量。

详细命令、首轮失败与补测、视频来源和限制见 [M5 实采记录](docs/m5-live.md)。
下一步先确认 802.11b 发射并采集足够长的 IQ，检查检测 / 合并 / 提取 / 模型各层；实物 50 Ω 校准仍待补齐。
本轮没有实采分类真值，不报告实采准确率。

### 2026-11-18 · M5 现场定性检查、长测与视频完成（运行记录）

- 重新采集 10 秒 RX2 IQ，独立检查 2442 MHz：外部 WiFi 接收器读到信标，Barker-11 相干能量峰值
  0.980；DSSS / OFDM 对照分别为 0.987 / 0.492。这是物理层与协议观察，未建立实采准确率真值。
- 修复历史窄轨迹锁住完整宽突发的问题：有界地复查当前连通宽组件，保留原有轨迹和模型概率。
  固定 10 秒输入中，目标 WiFi DSSS 推测由 0 条变为 114 条，十个整秒区间均出现。
- 6 dB 检测门限下发现同步显示处理阻塞接收管线。完整模型记录改为独立落盘，网页按轨迹每 200 ms
  合并信号观察，保留全部瀑布行和 ADS-B 消息；界面明确标注计数口径，文件和内存有硬上限。
- 低门限单路 300 秒通过，保留完整模型结果 1173199 条。双路先出现 23 次 busy，扩为 8 个在途窗口后
  清除了 busy，但仍有 8 次请求数上限跳过；继续扩展为 512 请求 / 1024 轨迹后，双路 300 秒复验通过，
  保留完整模型结果 1157863 条。**最终两场零驱动异常、零丢样 / 缺口、零二级 busy / ring / capacity 跳过**，
  宽带处理 p99 为 11.08 / 15.30 ms。全程 `tx_samples=0`。
- 2442 MHz 附近宽 DSSS 在单 / 双路 RX2 上命名 433 / 653 条，五个源时间分钟均出现；最长命名空窗
  39.22 / 21.40 秒。5.8 GHz WiFi、939 MHz LTE、ADS-B 沿用首轮长测并做了新版本接入短测。
  **当时只完成运行与视频检查；识别质量按后续复查仍未通过，未报告实采准确率。**
- 最终 C++ 58/58、Python 55/55 通过，新增 GPU 路径 memcheck 零错误；334 请求双路测试和重排的
  512 窗口 FP32 参考检查通过。512 配置的原 M4 104 场完整模式回归为 0/1720 错误、86.98% 未识别，
  高 SNR 覆盖 79/91。200 ms 显示分支的短场景回归只有 132 条命名，虽无错误，未达到每频段 100 条门槛，
  没有把它记成新的完整 M4 验收。
- 120 秒演示视频覆盖四频段和双 RX；ADS-B 为明确标注 10× 的实采回放，2.4 GHz 使用最终通过的现场录像。
  旧失败试验、源码 / 模型 / 二进制快照和哈希保持可追溯，归档压缩经过解压哈希核对。

复现参数、版本差异、完整结果和视频来源见 [M5 最终复验](docs/m5-dsss.md)。
后续工作为实物 50 Ω 校准、减少候选碎片与未识别、评价抽样显示的短突发召回，以及独立实采真值集。

### 2026-11-18 · M5 重新打开：识别质量未通过

- 用户明确要求不能接受大量未识别。此前的现场出现标签、零丢样和完成录像不足以作为识别交付标准。
- 已修复先抽样后投票导致的突发证据丢失：同一 M4 feed 的高 SNR 覆盖从 60/91 恢复到 79/91，
  全部 1720 条已命名观察保留，原始未知总计不因显示合并而减少。
- 完整宽组件不再重复分类其历史碎片，单条窄历史也不能锁住后续宽提取，拆分后仍可恢复独立轨迹。
  104 场回归为 0/1703 错误、79/91 高 SNR 覆盖、86.87% 原始未识别；运行与负样本检查通过。
- 相同 10 秒 IQ、相同容量与实时参数对照：未知候选记录 24641 → 11443，目标宽 DSSS 命名 69 → 90，
  最长命名空窗 1.84 → 0.61 秒；原始未知比例仍 98.21%。中心平滑试验使真值覆盖降到 77/91，已撤回。
- 新录音在 2442 / 2462 MHz 均有独立 Barker 证据，完整识别链路仍漏识别；调中心频率的三场 30 秒短测
  原始未知比例仍约 98%。不能据此把 M5 标为识别质量合格。
- 当前优先修完整载波定位、相邻宽信号分离与模型选窗，再判断是否需要调整训练数据；50 Ω 校准等待实物条件。
  后续同时报告原始未知总量、带真值覆盖和实采标签空窗，不靠隐藏未知或降低置信门限过关。

复现与未解决样例见 [M5 识别质量复查](docs/m5-recognition.md)。

### 以后再做

- FM 室外演示：换 VHF 天线。
- 实采测试集 v2：56 MHz 带宽重录；噪声参考接 50 Ω 负载（RX1 接 TX1 的 SMA 线在 2.4 GHz 会像天线一样漏进信号）；突发频段录 10 秒以上；2.1 GHz 加 NR 解码器确认。
- 第一级改成检测器（信号 / 噪声 / 边缘），代替能量门限。
- GNSS 多系统捕获（需要有源天线和偏置供电）。
- 用实采数据微调第二级，FM 域差异是重点。

## 10. 待确认

- [x] 新建 `curf` conda 环境（2026-11-14）。注意：PyPI 上的 torchaudio 是 CUDA 13 版本，必须从 PyTorch 的 cu128 源安装，否则 TorchSig 导入失败。
- [x] 安装 TensorRT 11.4（2026-11-14，用户目录安装，见第 2 节）。
- [ ] 下载的 deb 包 `~/opt/tensorrt-debs`（1.4 GB）可以删除；主机磁盘已用 80%。
- [x] SDR 接到 CUDA 主机上，udev 规则已装（2026-11-15）。
- [ ] CUDA 主机内存：检查时 15 GB 中已用约 12 GB，可用约 3 GB，长测前需确认是否有常驻进程占用。
- [x] 回环：TX1 用 SMA 线直连 RX1，没有衰减器，靠软件把 TX 衰减锁在 ≥ 40 dB（2026-11-16）。
