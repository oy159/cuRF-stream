# cuRF-Stream

USB3 SDR → GPU 实时流处理与信号识别。数据来自自研传输层 [OmniCore-SDR-Host](../OmniCore-SDR-Host)，GPU 侧负责解包、信道化、检测和自研推理引擎。计划与进度见 [PLAN.md](PLAN.md)。

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

在 Mac 上开发、在 CUDA 主机上编译运行：先把 `tools/remote.env.example` 复制为 `tools/remote.env` 并填好主机，然后：

```sh
tools/remote.sh test     # 同步 + 编译 + 测试
tools/remote.sh bench    # 同步 + 编译 + 基准
```

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
