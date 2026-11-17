# cuRF-Stream

**一块自研 USB3 SDR + 一台游戏本，实时识别 2 × 61.44 MHz 带宽内全部 1024 个信道的调制方式。**

数据来自自研 USB3 SDR 及其传输层 OmniCore-SDR-Host。GPU 侧依次做解包、多相信道化、能量检测，再用**自研推理引擎**（手写 Tensor Core / INT8 kernel，不依赖 TensorRT）完成识别。整条链路跑在一块 60 W 的 RTX 3050 笔记本 GPU 上。

<!-- TODO: 实时界面录屏 docs/img/demo.gif（瀑布图 + 识别标注 + 角落常驻 MS/s、丢样数、p99 延迟） -->

| | |
|---|---|
| 输入 | 2 路 × 61.44 MSPS，12 bit，USB3（368.6 MB/s） |
| 信道化 | 512 信道 × 120 kHz / 路，2 倍过采样 |
| 识别 | 9 类调制 + 噪声，每信道每 4.27 ms 一次 |
| 实采长测 | **300 秒零丢样**，端到端延迟 p50 1.74 ms / p99 2.47 ms |
| 推理引擎 | INT8 **34.3 万窗口/秒**；FP16 版对 TensorRT 11.4 领先 **1.3–1.4×** |
| 全负载（1024 信道全部识别） | 吞吐模式 1.31×；合成源 300 秒实时持平（受笔记本约 38 W 散热上限约束） |
| 硬件 | RTX 3050 Laptop（4 GB，sm_86，60 W）、i5-11400H |

---

## 目录

- [硬件：OmniCore SDR](#硬件omnicore-sdr)
- [系统架构](#系统架构)
- [自研推理引擎](#自研推理引擎)
- [端到端结果](#端到端结果)
- [实采与回环](#实采与回环)
- [工程中的几个发现](#工程中的几个发现)
- [快速开始](#快速开始)
- [代码结构](#代码结构)
- [局限](#局限)

---

## 硬件：OmniCore SDR

本项目是 OmniCore SDR 的演示应用：插上板子，任何一台带 USB3 和 NVIDIA 显卡的电脑都能做全带宽的实时信号识别，不需要专用的 GPU 嵌入式平台或 RDMA 网卡。

| | |
|---|---|
| 射频 | AD9361，2T2R，70 MHz – 6 GHz，模拟带宽最高 56 MHz |
| 接口 | USB3（5 Gb/s），C ABI 驱动（OmniCore-SDR-Host） |
| 采样 | 2 路 61.44 MSPS（12 bit packed 或 8 bit）；12 bit 2R1T 时最高 58.5 MSPS |
| 时间 | 硬件时间戳，按样点计数，丢样可精确定位 |
| 可编程 | RX FIR 可在线加载（本项目用 64 tap ×2 抽取，阻带 −78 dB） |
| 传输层实测 | `och bench` 2 × 61.44 MSPS：368.8 MB/s，零丢样，零溢出 |

<!-- TODO: 购买 / 联系方式；OmniCore-SDR-Host 仓库链接 -->

cuRF-Stream 只通过 OmniCore 的 C ABI 依赖它（`-DCURF_OMNICORE=ON`）。没有硬件也可以用合成源或录制文件跑通全部流程。

---

## 系统架构

```
CPU（C++）                                GPU（CUDA）
──────────────────────                   ─────────────────────────────────────
OmniCore USB 异步收包
  │ och_recv 直接写入
HostRing：SPSC 无锁环形缓冲
  （2 MB 透明大页 + cudaHostRegister）
  │ H2D（copy stream）
  └────────────────────────────────────▶ ① 多相信道化：12 bit 解包融进 FIR fold + 批量 cuFFT
                                         ② 逐信道功率、噪声底（25% 分位）、噪声形状白化
                                         ③ 检测 + 邻道泄漏抑制 + warp ballot 压缩活跃信道
                                         ④ gather：转置 + 归一化，直接写 fp16
                                         ⑤ 推理：自研 INT8 引擎（6 个 kernel，CUDA Graph）
                                         ⑥ softmax / argmax
  ┌──────────────────────────────────── D2H：活跃数、识别结果、显示用功率
调度：GPU 决定 batch 大小（主机只读回一个计数），推测提交
背压 / 丢样计量，样点时间戳查缺口
浏览器界面（SSE）：瀑布图 + 信号跟踪 + 识别标注
```

数据源通过 `ISource` 接口可插拔：合成源（可超过 USB 速率，用来压测 GPU）、文件回放源（可复现）、OmniCore 实采源。

---

## 自研推理引擎

模型是一个小 ResNet-1D（13.6 万参数，每窗口 50.8 MFLOP，输入 `[B, 2, 1024]`）。引擎从 ONNX 开始全部自己实现：

- **离线转换器**（`model/convert.py`）：BN 折叠、conv + bias + 残差 + ReLU 融合、channels-last 布局、自定义二进制格式，同时导出逐层 golden 激活。
- **运行时**（`src/engine/`）：按张量生命周期做静态内存规划（arena 是不复用时的 1/2.5），热路径零 `cudaMalloc`；按 batch 档位分别捕获 CUDA Graph；每层多个 kernel 变体在线自动调优，结果可缓存。
- **kernel**（`src/kernels/`）：从 naive 到手写 PTX 的 `cp.async` + `ldmatrix` + `mma.sync` 隐式 GEMM，XOR swizzle 无 bank 冲突布局，fragment 寄存器双缓冲；整个 BasicBlock 融成一个 kernel；INT8 版复用同一套 tile 结构。
- **融合决策由实测决定**：每个 batch 档位、每个块，在整网里交替计时“融合 / 不融合 / 带 stem 融合 / 带 shortcut 融合”，取更快的方案。

### 优化阶梯

batch 512，CUDA Graph，冷机。每一级都有 Nsight Compute 数据支撑。

| 步骤 | 窗口/秒 | 有效 TFLOPS | 关键改动 |
|---|---|---|---|
| 1. naive | 1,277 | 0.07 | 每线程一个输出 |
| 2. shared memory 分块 | 16,503 | 0.84 | fp32 FMA |
| 3. WMMA Tensor Core | 79,939 | 4.06 | epilogue 分离 |
| 4. + epilogue 融合、fp16 累加 | 140,848 | 7.15 | bias + 残差 + ReLU 在寄存器内完成 |
| 5. `cp.async` + `ldmatrix` + `mma.sync` | 177,472 | 9.01 | 多级流水、im2col 地址外提 |
| 6–8. + GAP / stem / 整块融合 | 224,044 | 11.4 | 激活流量 840 → 260 KB / 窗口 |
| 8'. + swizzle、fragment 双缓冲 | **230,200** | **11.7** | 每层都追平或超过 CUTLASS |
| 9. INT8 + 融合块 | 300,000–315,000 | — | 对称量化，99.99% 分位校准 |
| 10. + stem / shortcut / head 融合 | **343,000** | — | 每次推理 6 个 kernel |

从 naive 到 FP16 最终版约 **180×**；INT8 再快 1.5×。batch 1 延迟 30.1 µs。

### 和库的对比

同一 ONNX、同一轮背靠背测量（GPU 已温热，绝对值偏低）。cuDNN 和 CUTLASS 作为 kernel 变体接进同一个引擎，直接在引擎的 buffer 上运行，没有额外拷贝或重排。

| 后端（FP16） | batch 512 窗口/秒 | 自研领先 | batch 1 | batch 8 |
|---|---|---|---|---|
| **自研，完整融合** | **204.4k** | — | 42.1 µs | 59.2 µs |
| CUTLASS 4.5 | 178.5k | 1.15× | 48.3 µs | 66.5 µs |
| 自研，逐层不融合 | 176.9k | 1.16× | — | — |
| TensorRT 11.4 | 154.5k | 1.32× | 42.0 µs | 81.8 µs |
| cuDNN 9.21 | 113.5k | 1.80× | 53.6 µs | 124.5 µs |

- **单层 GEMM**：与 CUTLASS 基本持平（不含 stem 的 10 层合计，自研快 3.8%）。
- **整体领先**：来自库做不到的部分，即整块融合、GAP 融合，以及按实测决定融合方案。
- **对 TensorRT**：多出的余量还来自头部（TensorRT 为 Myelin 子图做了两次 reformat）和省掉的输入重排。

---

## 端到端结果

### 负载与 CPU 基线

全部 1024 个信道都识别时，需要每秒 24 万个窗口。

| | GPU（本项目） | CPU（i5-11400H，AVX-512） |
|---|---|---|
| 推理 | 34.3 万窗口/秒（INT8 引擎） | 3,339 窗口/秒（ONNX Runtime，12 线程） |
| 信道化 | 19.2× 实时（每块 0.111 ms） | PyTorch 0.07× 实时 |
| 能跟上的信道数 | 全部 1024（吞吐 1.31×） | 约 14 |

- **信道化**：12 bit 解包融进 FIR fold，不落地 fp32 中间结果，比 PyTorch 实现快 9.3×。
- **profile**：全负载时推理引擎占 GPU 时间的约 89%。

### 延迟

“端到端”从一个窗口的最后一块数据进入环形缓冲算起，到识别结果回到主机为止。窗口本身 4.27 ms 的积累时间由算法决定，不计入。

| 场景 | 丢样 | p50 / p99 / max |
|---|---|---|
| 合成源，每路 64 个活跃信道 | 0 | 0.95 / 1.05 / 1.15 ms |
| 合成源，全负载 30 秒 | 0 | 3.8 / 4.1 / 4.2 ms |
| **实采 300 秒，相对门限检测（平均 318 个活跃信道）** | **0** | **1.74 / 2.47 / 8.4 ms** |

### 准确率

准确率不是本项目的重点，模型保持小而够用。数字只用来说明整条链路没有损失精度。

| | 准确率 |
|---|---|
| 模型本身（合成数据，SNR −4 至 24 dB） | 89.5%（SNR ≥ 10 dB 时 94.7%） |
| 合成宽带场景走完整流水线，FP16 / INT8 | 92.9% / 92.7%（同信号直接送模型 92.9%），检测召回 100% |
| **实采 TX→RX 线缆回环，SNR 24 dB** | **91.4%** 单窗口；按 128 ms 投票 **96.3%** |

主要错误来自 16QAM 和 64QAM 互相混淆：1024 点窗口里只有约 43 个符号，不足以可靠区分高阶 QAM。

---

## 实采与回环

- **300 秒实采**：2 × 61.44 MSPS，驱动层和流水线均零丢样，没有出现不连续。主机队列峰值只有 0.9 MiB。
- **线缆回环**：TX1 发射 104 个已知调制的信号（9 类，在频域按整周期构造，循环发射无接缝），TX 衰减从 89.75 dB 逐级降到 50 dB，测量识别率随 SNR 的变化。软件把 TX 衰减锁在 ≥ 40 dB，低于这个值时拒绝打开设备，保护 RX 端。
- **空中实测**：900 MHz 的 LTE/NR 下行、2.4 GHz 的 WiFi 突发、FM 电台都能被检测并识别。
- **前端噪声形状校准**：AD9361 的通带边缘有 5–9 dB 的噪声抬升，大小由增益决定。先在无信号时校准一次噪声形状，检测时按形状白化后用全局本底判断，误报减少一半到三分之二。
- **实时界面**：信道跟踪（指数衰减投票）、信号出现 / 消失的迟滞判断、相邻同类信道合并、区分突发和持续信号、拖动放大、信号列表。

---

## 工程中的几个发现

- **H2D 只有 PCIe 理论值的 45%**：根因是 IOMMU 的 IOTLB 缺失。`cudaHostAlloc` 按 4 KB 页映射，1 MB 就要 256 个表项。改用 2 MB 透明大页加 `cudaHostRegister` 后，H2D 从 7.0 提升到 12.4 GB/s（1.8×）。运行时还会从 `/proc/self/smaps` 检查大页覆盖率。
- **融合了反而更慢，ncu 每次都能定位原因**：常见的有三种，smem 超额导致 occupancy 下降、`cp.async.cg` 绕过 L1 后 k = 3 的重复读要回 L2、epilogue 写不合并。
- **融合与否不能靠单独计时来决定**：单独测时输入还留在 L2 里，而笔记本降频时 SM 时钟下降、显存时钟不变，两者都会让判断翻转。所以改为在整网里交替计时。
- **热态下的 autotune 不可靠**：GPU 热的时候调优，选出的 kernel 时好时坏（同样的负载跑出 0.89–1.09×）。应在冷机时调优一次并缓存结果（`--tune-cache`）。
- **有记录的负结果**：例如把 gather 并进 stem 融合块（访存跨度 4 KB，每个 sector 只用到 8 字节），以及在融合块上移植 swizzle 和双缓冲。都测过、写清原因后删除或保留为变体。
- **主机侧忙等也耗电**：`cudaEventSynchronize` 默认自旋，会占满一个 CPU 核；改用 blocking sync 后 CPU 占用从 90% 降到 30%。笔记本的 CPU 和 GPU 共用散热，CPU 少发热，GPU 就能维持更高的时钟。

**正确性**：每个 GPU kernel 都先有 CPU 参考实现或 golden 数据。解包、信道化线格式融合、fp16 输入等路径与参考逐位一致；推理的每个 kernel 变体都逐层对齐 golden。共 41 个测试（CUDA 25 个，CPU 16 个）。

---

## 快速开始

### 构建

CPU 部分（macOS / 任意平台，含参考实现、源、环形缓冲、模型加载等测试）：

```sh
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug
```

CUDA 部分（Linux，sm_86，CUDA 12.8）：

```sh
cmake --preset cuda-release
cmake --build --preset cuda-release
ctest --preset cuda-release
```

可选组件：

| CMake 选项 | 作用 |
|---|---|
| `-DCURF_OMNICORE=ON -DCMAKE_PREFIX_PATH=<omnicore 安装目录>` | OmniCore 实采源（需要 libusb-1.0 开发包） |
| `-DCURF_TENSORRT=ON`，环境变量 `TENSORRT_ROOT` | TensorRT 基线 |
| `CURF_CUDNN` / `CURF_CUTLASS`（`CURF_CUTLASS_DIR` 指向头文件） | cuDNN / CUTLASS 基线，找到时自动启用 |

### 模型

模型文件不进仓库，用自带的 GPU 合成器训练（RTX 3050 Laptop 上约 37 分钟），然后导出、转换：

```sh
pip install -r model/requirements.txt   # torch 需从 cu128 源安装，见文件内注释
python model/train.py   --out model/artifacts/run2 --steps 30000
python model/export.py  --ckpt model/artifacts/run2/best.pt --out model/artifacts/run2/export
python model/convert.py --onnx model/artifacts/run2/export/amc_fp32.onnx \
                        --ref  model/artifacts/run2/export/reference.npz \
                        --out  model/artifacts/run2/export
```

### 运行

```sh
# 合成源，每路 64 个活跃信道，按真实速率跑 10 秒
build/cuda-release/bench/curf_classify --int8 --tune-cache /tmp/i8.tune --active 64 --realtime --drop --seconds 10

# 全负载：1024 个信道全部识别
build/cuda-release/bench/curf_classify --int8 --tune-cache /tmp/i8.tune --detect all --seconds 30

# 带标注的宽带场景：生成 → 识别 → 打分
python tools/make_scenario.py --out /tmp/scn --windows 16 --spacing 4 --snr 5 20
build/cuda-release/bench/curf_classify --int8 --source file --file /tmp/scn/capture.raw --detect relative --dump /tmp/scn/results.csv
python tools/make_scenario.py --score /tmp/scn

# 实采 + 浏览器界面（http://<host>:8080/）
python3 tools/ui_server.py --port 8080 -- build/cuda-release/bench/curf_classify --int8 \
    --tune-cache /tmp/i8.tune --source omnicore --freq 2.44e9 --gain 40 \
    --noise-shape <校准文件> --threshold-db 4 --band-mhz 60 --seconds 300
```

前端噪声形状校准：天线断开（或输入端接负载）时，运行 `curf_classify --source omnicore --calibrate-noise <输出文件>`。换增益后需要重新校准。

### 基准

`bench/` 下每个程序都输出 JSON 行，便于复现和画图：

| 程序 | 测什么 |
|---|---|
| `curf_bench_unpack` / `curf_bench_pcie` | 解包 kernel 带宽；H2D / D2H 与大页的关系 |
| `curf_pipeline` | P1 流水线吞吐、stream 重叠、延迟 |
| `curf_bench_channelizer` | 多相信道化 |
| `curf_check_extraction` | 已知信号参数的 IQ 提取、连续 CPU 参考对齐与导出（见 [M2 验证说明](docs/m2-extraction.md)） |
| `tools/run_signal_checks.py` | M2 信号级回放、双路 300 秒实时验收（见 [链路与复现说明](docs/m2-pipeline.md)） |
| `tools/run_display_checks.py` | M3 频段策略、真实模型回放与 ADS-B 联动检查（见 [显示策略说明](docs/m3-display.md)） |
| `tools/run_room_checks.py` | M4 开发 / 验收种子分离的随机 RF 场景、负样本控制和标签错误率检查（见 [验收口径](docs/m4-evaluation.md)） |
| `model/make_room_scene.py` | 分块生成 61.44 MS/s 室内 packed12 场景，保留每个 RX 的包级真值与源文件哈希 |
| `curf_bench_engine` | 引擎优化阶梯、逐层耗时（`--ops`）、ncu 采集（`--profile`）、库基线 |
| `curf_classify` | 端到端：吞吐、延迟分位、丢样、准确率 |

辅助脚本：`tools/ncu_summary.py`（ncu 报告逐 kernel 汇总）、`tools/nsys_overlap.py`（拷贝与计算的重叠率）、`tools/cpu_baseline.py`、`tools/loopback.py`。

---

## 代码结构

```
include/curf/         公共头文件（格式、ISource、参考实现、CUDA / 引擎 / 运行时接口）
src/
  source/             合成源、文件回放源、OmniCore 实采源
  runtime/            HostRing、生产者、P1 流水线、端到端 classifier
  kernels/            解包、PSD、信道化、检测与 gather、conv（naive / WMMA / mma / INT8）、
                      融合块（FP16 / INT8）、cuDNN 与 CUTLASS 基线
  engine/             模型加载、内存规划、融合决策、自动调优、CUDA Graph
  reference/          CPU 参考实现
model/                合成器、ResNet-1D、训练 / 导出 / 转换
bench/                基准程序与 curf_classify
tests/                Catch2 测试（GPU 对 CPU 参考 / golden）
tools/                界面、场景生成、回环、FIR 设计、噪声形状、profile 汇总
```

---

## 局限

- **模型是在合成数据上训练的**。真实的 FM 广播（带立体声和 RDS）、GSM 与训练数据有域差异；硬件伪影（直流残余、杂散）也会被分到某一类。后续计划：用实采数据微调，并增加“未知”类别。
- **单个 240 kHz 信道里，宽带类噪声和 OFDM 本来就难以区分**，所以空中实测里 OFDM 的命中只能算定性一致。
- **全负载在实采时扛不住**：持续 1024 信道识别时，实采只有 0.97×（丢样 3%）。CPU 处理 368 MB/s USB 数据产生的热量会压低 GPU 时钟。要留出余量，需要减少计算（时间复用、模型瘦身），或者走零拷贝 USB → 锁页内存这条路。
- **只在一台机器上测过**（RTX 3050 Laptop，散热上限约 38–45 W）。
