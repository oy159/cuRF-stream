# M2：已知参数的信号提取验证闭环

2026-11-17。先固定中心频率和抽取倍率，验证未归一化的 IQ，再接检测、跟踪和第二级模型。

## 实现与数据约定

- `include/curf/extraction.hpp`：提取参数、有限 FIR 和连续 CPU 参考。
- `include/curf/cuda/extraction.hpp` / `src/kernels/extraction.cu`：CUDA overlap-save 基线。
- `bench/check_extraction.cpp`：合成已知信号、打包 wire、GPU 解包、提取和 IQ 导出。
- `tools/run_extraction_checks.py`：运行案例、核对误差，写 JSONL；`tools/check_extraction.py` 可单独检查一次导出，也能复现现有 SignalStage 的独立 FFT 裁剪算法。

CPU 参考先按原始样点时间混频，再直接做双精度连续 FIR 卷积和整数抽取。CUDA 则把 FIR 调制到原始中心频率，使用 FFT 卷积，保留历史、丢弃无效前缀，抽取后按绝对样点时间混频。两者不共用 FFT 或块处理逻辑，只共用 FIR 系数及参数定义。

设整数抽取倍率为 D，输出采样率为 fs/D。FIR 为 32D+1 tap 的 Blackman 窗低通，截止频率为输出采样率的 0.375 倍，DC 增益为 1；期望信号的整个带宽应不超过输出采样率的一半。测试核对通带边缘（0.25 倍输出采样率）的增益误差和 Nyquist（0.5 倍）的抑制度。

滤波器是因果的，群延迟为 **16D 个原始样点，即 16 个输出样点**。每个输出的名义样点时间为 `first_sample + j*D`，群延迟单独写入元数据；幅度、相位、启动瞬态不通过 RMS 归一化隐藏。采集开始前的历史定义为零；有限文件末尾不自动补尾，需要尾部响应时由调用方继续输入零。

每次调用的输入块大小固定，且必须整除 D。调用之间检查样点时间连续性；断流或调谐后必须显式 `reset()`，避免拼接不同时间或不同频率的数据。设备分配和 cuFFT plan 在构造时完成，热路径只提交到调用方 stream。测试连续提交多块后只在末尾同步，覆盖状态在异步队列中的复用；也覆盖 `2^60` 以上的样点时间。

## 复现

仅验证 CPU 部分可以在 Mac 上运行：

```sh
cmake --build --preset cpu-debug
ctest --preset cpu-debug
```

CUDA 主机上构建并运行（Python 需要 NumPy）：

```sh
cmake --preset cuda-release -DCURF_TENSORRT=OFF -DCURF_CUDNN=OFF -DCURF_CUTLASS=OFF
cmake --build build/cuda-release --target curf_check_extraction curf_tests_cuda -j 4
build/cuda-release/tests/curf_tests_cuda '[extraction]'
python tools/run_extraction_checks.py \
  --bin build/cuda-release/bench/curf_check_extraction \
  --out /tmp/curf-m2-iq --results docs/results/m2_extraction.jsonl
```

单独生成一个案例：

```sh
build/cuda-release/bench/curf_check_extraction \
  --out /tmp/curf-narrow --decimation 4096 --signal tone --first 0
python tools/check_extraction.py /tmp/curf-narrow --legacy
```

每个输出目录包含 `input.cf32`、`cuda.cf32`、`reference.cf32` 和 `meta.json`。IQ 是 little-endian complex64，未做归一化。`input.cf32` 是所选 RX 路已经量化再解包的数据；另一路放置不同信号，以检查去交织和选路。元数据记录中心、采样率、抽取倍率、原始样点时间、块大小、FIR 长度、群延迟和误差。

跨块突发案例同时包含噪声和同幅带外干扰。`--fc` 支持非 FFT 栅格中心；`--streams` / `--stream` 选择输入路；`--wire` 支持 packed12 / signed8。导出程序或 Python 检查在最大复数绝对误差超过 **3e-6** 时返回非零。

## 远端实测

RTX 3050 Laptop，CUDA 12.8，Release，原始采样率 61.44 MS/s。在独立目录 `~/work/cuRF-Stream-m2-check` 构建，模型文件只读复用原目录。此验证构建关闭 TensorRT、cuDNN 和 CUTLASS 基线；现有自研引擎和第二级测试仍参与回归。

10 组案例全部通过：

| 案例 | 输出采样率 | 最大逐点绝对误差 | 块边界最大绝对误差 |
|---|---:|---:|---:|
| 宽带单音 | 15.36 MS/s | 2.10e-7 | 6.66e-8 |
| 宽带跨块突发 | 15.36 MS/s | 2.48e-7 | 8.69e-8 |
| 信道单音 | 240 kS/s | 1.90e-7 | 1.50e-7 |
| 信道跨块突发 | 240 kS/s | 1.79e-7 | 1.47e-7 |
| 窄带单音 | 15 kS/s | 1.11e-7 | 7.45e-8 |
| 窄带跨块突发 | 15 kS/s | 1.10e-7 | 5.37e-8 |
| signed8 窄带突发 | 15 kS/s | 9.54e-8 | 6.32e-8 |
| 非二次幂抽取 D=3、非栅格中心 | 20.48 MS/s | 3.16e-7 | 1.20e-7 |
| 历史区是输入块的 4 倍 | 60 kS/s | 8.02e-8 | 8.02e-8 |
| 靠近输入 Nyquist 的信号 | 15 kS/s | 1.16e-7 | 7.78e-8 |

这里的误差以连续 CPU FIR 为参考，包含零历史启动和所有块边界。`docs/results/m2_extraction.jsonl` 保存完整记录，原始 IQ 留在远端 `/tmp/curf-m2-iq/`，不加入仓库。

另用 0.25 幅度单音复现原 SignalStage 的裁频点、余弦边缘、逐块 IFFT，去掉固定初始载波相位；新路径扣除已知群延迟后，与理想单音比较，排除 FIR 启动的前 32 个输出样点：

| 输出采样率 | 原路径块边界 RMS 误差 | overlap-save 块边界 RMS 误差 |
|---|---:|---:|
| 15.36 MS/s | 0.0696 | 8.87e-5 |
| 240 kS/s | 0.0709 | 2.57e-5 |
| 15 kS/s | 0.0604 | 1.09e-6 |

这组单音比较用于定位块边界失真。两条路径的滤波器不同；量化噪声和 FIR 通带误差也包含在理想单音误差中，不能把这张表当作所有调制或邻道条件下的质量结论。

CUDA 新测试通过；Compute Sanitizer memcheck 报告 0 errors。远端完整回归 **51/51 通过**。Mac CPU 回归 21 个通过，1 个因本地没有模型文件而跳过。完整回归时发现 CPU 模型加载测试仍写死 run1 的 10 类，而 run2 已加入 `wide`，现已兼容两版并在 golden 存在时核对类别数。

## M2 后续接入边界

此处记录的是第一步基线完成时的边界。后续共享 FFT 滤波器组、运行时接入和验收见 [M2 信号级链路](m2-pipeline.md)。

本次完成的是**已知参数 → wire / CUDA 提取 → IQ 导出 → 连续参考对齐**。现有 `SignalStage` 仍使用实验性的 FFT 裁剪路径，尚未用此基线替换；M2 的实时零丢样和信号级延迟验收尚未完成。

- 此基线每个信号各做一组 FFT 卷积，方便独立核对。多信号共享前向 FFT、IFFT 裁剪和 plan/workspace 复用应在保持参考结果的前提下继续实现；这里没有测量多信号长测性能。
- 此基线采用整数 D。现有 SignalStage 的 `{2^m, 1.5*2^m}` 输出尺寸通常对应有理数抽取，不能直接换成整数 D 后沿用同一训练分布。接入前需要明确采样率档位、相位、滤波器和群延迟契约。
- 下一步先用已知参数将正确提取路径接入第二级，比较同一模型对 CPU / CUDA 提取窗口的输出；再接入检测和跟踪。短突发检测、邻近信号切分、场景真值的时间匹配及完整信号级延迟统计仍需单独验证。
