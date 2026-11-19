# cuRF-Stream

USB3 SDR → GPU 实时流处理与信号识别。数据来自自研传输层 [OmniCore-SDR-Host](../OmniCore-SDR-Host)，GPU 侧负责解包、信道化、检测和自研推理引擎。

## 构建

CPU 部分（macOS / 任意平台）：

```sh
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug
```

CUDA 部分（Linux，sm_86；TensorRT 由 `TENSORRT_ROOT` 指定）：

```sh
source ~/opt/tensorrt-11.4/env.sh
cmake --preset cuda-release
cmake --build --preset cuda-release
ctest --preset cuda-release
```

## run1 演示回放

单路 packed12 录音可通过 `--freq` 指定采集中心频率，界面按绝对频率显示。

```sh
build/cuda-release/bench/curf_classify --source file --file capture.p12 \
  --streams 1 --freq 5.77e9 --seconds 0 --int8 --ui-out capture.feed
python3 tools/ui_server.py --bind 127.0.0.1 --replay capture.feed --loop
```

2.4 GHz 演示优先显示有 Barker‑11 证据的 DSSS。DSSS 色带限在对应信道的占用子区段，其余宽带按相邻信道的 OFDM 票判决。
同一片占用的 Barker 标签过期后显示「占用」。5 GHz 演示依据与占用频带匹配的循环前缀。
物理层证据检测支持 40–65 MS/s 的输入。用「原始分类」查看逐信道模型的结果。

## 重训版实时演示

使用七类 demo 模型的导出目录。以下命令以 RX2 只收模式采集 30 秒，并同时保存瀑布和逐突发标签。

```sh
build/cuda-release/bench/curf_classify --burst --model model/artifacts/demo/export \
  --int8 --streams 1 --rx 2 --source omnicore --freq 2441e6 \
  --gain 18 --bandwidth 56e6 --band-mhz 50 --ring 128 --seconds 30 \
  --exclude-hz 2457600000 --ui-out rx2.feed
python3 tools/ui_server.py --bind 127.0.0.1 --replay rx2.feed --loop
```

5.8 GHz 可用 `--freq 5770e6 --gain 40 --band-mhz 44`。实时运行时，把采集命令放在
`tools/ui_server.py --bind 127.0.0.1 --` 后并省略 `--ui-out`。信号列表显示模型置信度或逐包证据。

## 目录

```
include/curf/      公共头文件（格式、ISource、参考实现、CUDA 接口）
src/reference/     CPU 参考实现
src/source/        合成源、文件回放源
src/kernels/       CUDA kernel
tests/             Catch2 测试（GPU 结果与 CPU 参考逐比特比对）
bench/             基准程序，输出 JSON 行
tools/             远程构建脚本
```
