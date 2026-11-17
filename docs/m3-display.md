# M3 信号显示策略

M3 的软件链路已接通：原始 `sig` → 可解释的显示判决 → 演示 / 工程视图。策略在
`tools/signal_policy.py` 中实现，实时 SSE 和离线评估共用同一份代码。模型、训练权重和原始
13 类概率没有被显示策略改写。真实 50 Ω 杂散校准仍待硬件条件具备后采集。

## Feed 和运行

新版 `curf_classify --stage2` 输出 version-2 元数据，声明信号类别顺序、是否运行逐信道分类、
RF 中心频率和 ADS-B 的接收路。文件 / 仿真源显式传入 `--freq` 时也保留该频率；未传时为
基带，不能自动假设 2.4 GHz。类别顺序对应冻结的 run4 模型。

每条 `sig` 保留原有字段，新增完整 `probs`、源时钟 `t` 和 `activity_fraction`。
`t0/t1` 仍是实际模型观察窗，而 `t` 是完成提取的原始 hop 末端时间，时间滞回使用后者。
`activity_fraction` 是提取 span 中最多 64 个候选窗的相对能量占比估计，门限为峰值的 20%；
它用于区分突发 / 持续的弱证据，**不是精确的发射占空比或包长度**。不足两个模型窗时为 -1。

服务器保留 `sig`，在同一 envelope 追加 `display` 和 `policy_stats`。工程视图同时提供
原始概率、允许集合内的后验、允许集合的原始概率总量、判决原因和时间参数。

```sh
# 在 CUDA 主机运行：文件保持源时钟速度回放
python tools/ui_server.py --port 8080 --tee /tmp/display.feed -- \
  build/cuda-release/bench/curf_classify --source file --file capture.p12 \
  --streams 1 --freq 2440e6 --realtime --seconds 0 \
  --stage2 model/artifacts/s2_run4/export

# 已录制的原始或带 display 的 feed
python tools/ui_server.py --replay /tmp/display.feed

# 离线应用完全相同的策略，并用发射真值独立评分
python tools/apply_signal_policy.py --feed scene_0.feed --out scene_0.display.feed \
  --truth truth.json --score display_score.json
```

`--tee` 保存原始记录及显示判决。重放带策略的 feed 默认保留录制判决；显式传入
`--policy overrides.json` 或 `--spurs spurs.json` 会重新计算。离线评分只接受一次完整有限运行，
拒绝缺少 `end` 或包含循环片段的输入，也拒绝原地覆盖输入文件。

## 先验、门限和时间状态

默认参数见 `tools/policy/default.json`。`--policy` 接受 JSON 覆盖；`bands` 覆盖整张频段表。
先验按计划覆盖 2.4 GHz、5 GHz、广播 FM、1090 MHz 和指定蜂窝频段；其他 / 无 RF 中心的
输入保留更高门限。蜂窝范围采用标准的 operating-band 边界，作为演示先验，不表示本地每个
频点都在发射或有某个运营商。参考 [LTE TS 36.101 表 5.5-1](https://www.etsi.org/deliver/etsi_ts/136100_136199/136101/10.32.00_60/ts_136101v103200p.pdf)
和 [NR TS 38.101-1 表 5.2-1](https://www.etsi.org/deliver/etsi_ts/138100_138199/13810101/16.24.00_60/ts_13810101v162400p.pdf)。

有先验时，进入已命名状态需要同时满足：

- 允许集合的原始概率总量 ≥ 0.85。
- 候选类别的原始概率 ≥ 0.80。
- 允许集合内归一化最高概率 ≥ 0.85，与第二名相差 ≥ 0.20。
- 同一频点连续 3 个独立 hop 的有效判断一致。重复记录同一 hop 不增加票数。

其他 / 基带输入的原始和归一化置信门限为 0.94。即使 FM 或 5 GHz 先验只允许一类，
归一化结果为 1 也不会绕过原始概率要求。旧 feed 只有前三概率时保持未识别，工程视图仍可查看。

确认后，至多 0.30 秒内可用较低的 0.70 原始概率 / 允许集合总量 / 后验和 0.10 差值保持标签。
新冲突或不足的证据立即显示未识别，连续两次不足会清除旧稳定状态。不能用旧标签掩盖一个
当前明显不合理的类别。短突发按 RX 和相近频点保留 3 秒投票，避免 C++ track ID 超时或复用
破坏 BLE 累积；中心与宽度不相容时建立新身份。状态最多 512 个载波，每次新运行 / 循环重置。

## 协议推测和解码

WiFi、蓝牙 / BLE、Zigbee、LTE / NR、GSM 的推测需要相应调制族、频段、带宽与活动证据。
缺少活动证据时只显示调制族；持续蜂窝推测还要求至少 0.3 秒的观测跨度。协议名称始终带
“推测”。B41 与 n41 的频段重叠，当前调制族模型无法可靠区分，显示 `LTE / NR（推测）`。
GSM 推测只用于 900 / 1800 MHz 的相应下行范围，不把任意蜂窝频段的 FSK 都叫作 GSM；频带依据
[GSM TS 45.005 §2](https://www.etsi.org/deliver/etsi_ts/145000_145099/145005/18.00.00_60/ts_145005v180000p.pdf)。

1090 MHz 附近的模型结果不成为演示标签。C++ 解码 tap 写入显式 `crc_ok`、绝对频率和
接收路；只有 CRC 通过的 DF17/18 进入飞机表和同一 RX 的 ADS-B 频带记录。单比特纠错后的
有效 CRC 同样保留 `fix` 标记。其他 Mode S / 未验证消息不当作 ADS-B 飞机显示。
选择 ADS-B 信号的频率会联动飞机选择与频段放大；飞机表和轨迹保留自己的源时间寿命。

## 杂散校准

没有内置的“实测杂散表”。已有 RX 接 TX 的线缆录音不当作 50 Ω 参考。

准备真实负载后，用固定手动增益记录至少 10 秒、零丢样 / 零时间戳缺口的硬件 feed：

```sh
python tools/build_spur_table.py --feed terminated.feed --terminated --out spurs.json
python tools/ui_server.py --spurs spurs.json --replay capture.feed
```

`--terminated` 声明这次录音确实接了 50 Ω 负载，软件不能检查物理连接。生成器只取至少
98% 功率行中持续高于本底 10 dB 的频带。表记录设备身份、源、RF 中心、采样率、RX 映射、
增益、AGC、模拟带宽、量化位数和噪声形状配置；设置不一致则不使用，并在界面说明。
只有覆盖候选自身带宽至少 80% 的已校准频带才匹配，不能用窄杂散掩盖旁边的宽信号。
`spur_action` 可设为 `mark`（默认，本机杂散）或 `hide`；工程视图都能检查。

## 界面与统计口径

演示使用信号级 `display`，不再在 JS 中把逐信道调制标签拼成发射机；原始功率瀑布图继续
显示全部能量。工程视图提供逐信道结果；第一级只检测时明确写“仅检测”，不伪造置信度。
类别筛选、持续信号筛选和频率按钮都可用键盘操作，原有拖动缩放有工具栏按钮替代。

晚加入的浏览器收到有界快照：最多 250 行功率、512 个最新信号和 128 架飞机，ADS-B 同一
事件不会被快照重复计数。每个订阅队列有上限，慢浏览器不会拖累采集。连接失败 / 超过三秒
无新样点时明确保留上次数据并提供重连；新运行重置投票、飞机和历史。

评分的主分母是**已命名调制族记录数**，错误包括匹配错误类别与未匹配到发射的记录。
未识别比例的分母为**已命名 + 未识别记录数**。杂散与 decoder-only 另计；完全没有已命名
标签时错误率为 null，不能宣称零错误。协议推测尚无独立协议真值评分。

## 验证与边界

```sh
python -m unittest discover -s tests -p 'test_signal_policy.py'
python -m unittest discover -s tests -p 'test_display_pipeline.py'
python tools/run_display_checks.py --bin build/cuda-release/bench/curf_classify \
  --model model/artifacts/s2_run4/export --src ~/dataset/onair \
  --emissions ~/dataset/ds_v5/emissions.json --out ~/dataset/m3_checks
```

实际 GPU + run4 的固定种子接入检查：

| 独立场景 | 已命名 / 未识别 | 已命名调制族错误 |
|---|---:|---:|
| 2440 MHz，WiFi OFDM + BLE，0.5 s | 36 / 4（10.0% 未识别） | 0 / 36 |
| 5800 MHz，WiFi OFDM，0.5 s | 18 / 2（10.0%） | 0 / 18 |
| 2135 MHz，LTE 5 MHz，0.5 s | 16 / 14（46.7%） | 0 / 16 |

另有双 RX、1080 MHz 调谐、RX2 的 +10 MHz ADS-B PPM 仿真：25 条 CRC 通过的 DF17，
24 条输出位置，正确联动 RX2 / 1090 MHz。没有通过射频发射信号。

2.4 GHz 同一场景额外循环实时回放 20.001 秒：1562 条二级结果，丢样 / 缺口 / busy / ring /
capacity 跳过均为 0，处理 p99 为 4.758 ms。此运行的低未识别比例受重复场景和确认启动开销影响，
不能外推为随机空中流量表现。

远端 C++ 回归 54/54；阶段二 Compute Sanitizer 为 3252 个断言、2 个用例通过、0 errors。
Python 策略 / feed 检查共 22 个，原有场景评分 4 个。浏览器检查覆盖演示 / 工程切换、原始
概率、逐信道详情、键盘选择器、空筛选、未识别、ADS-B 联动及 390 / 1280 px 布局；设计静态
审计无错误。记录保存在 `docs/results/m3_*`，完整源数据在远端 `~/dataset/m3_checks`。

这些是 **M3 软件接入检查**，不是 M4 的随机场景 <2% 验收。M4 还需要扩大随机频点、功率、
突发时序、噪声与错误类别注入，并同时检查已命名错误和未识别比例；M5 实采仍未进行。
