# ACWM Modality Forcing 数据输入改造说明

> 本文说明本次数据输入改造完成了什么、为什么需要修改，以及这些修改可能带来的效果。  
> 当前范围：OmniViTac/现有已处理数据。  
> 暂不包含：DROID 数据接入、模型侧 task mask 接入、正式训练与实验结果。

## 1. 改造目标

现有 ACWM（Action-Conditioned World Model）使用机器人动作作为条件，同时生成三种模态：

- video：RGB 视频。
- contact：左右手指的二维接触强度图，共 2 通道。
- force：左右手指的 xyz 形变/力代理场，共 6 通道。

原训练只有一种任务：

```text
T0：action + 第一张 RGB observation
    → future video + contact + force
```

本次目标是为 modality forcing 准备四种数据任务：

| 任务 | 条件 | 生成目标 | 训练比例 |
|---|---|---|---:|
| T0 | action + 第一张 RGB | 后续 video + contact + force | 40% |
| T1 | action + 完整 video | contact + force | 20% |
| T2 | action + 完整 video + 完整 contact | force | 20% |
| T3 | action + 完整 contact | 完整 video + force | 20% |

T3 不包含 RGB observation。它需要从第 0 帧开始生成完整 video，这一点与 T0 不同。

## 2. 本次交付范围

本次新增了一套独立的数据构建工具，放在 `modality_forcing_data_spec/`，没有直接修改原来的 `physical_wm` 和 `DiffSynth-Studio` 代码。

已完成的数据链路：

```text
已处理 episode
→ VAE clip index
→ episode 级 train/test split
→ train-only physical statistics
→ train/test multimodal pipeline data
→ T0/T1/T2/T3 mixture
→ 端到端 validation
```

本次生成的是索引、统计和检查报告，不复制原始 RGB/`.npy`，也不重新运行 SAM。

## 3. 修改总览

| 修改 | 为什么修改 | 可能达成的效果 |
|---|---|---|
| 新建独立数据工具目录 | 避免直接破坏旧数据脚本，便于对比和回退 | 降低对现有训练流程的影响，方便 code review |
| 统一稳定 `base_sample_id` | 各阶段原来主要依赖列表顺序和路径 | 能追踪同一 clip 在 VAE、split、pipeline、mixture 中的变化 |
| VAE clip 同时检查 RGB/contact/force | 旧代码构建 clip 时只显式检查 contact | 减少训练时才发现 force/RGB 缺失的问题 |
| 严格检查相机、帧号、通道和 NaN/Inf | 错误数据可能静默进入索引 | 更早暴露数据损坏和对齐错误 |
| train/test 改为严格 episode 级划分 | clip 级随机划分会泄漏相邻画面 | 测试指标更能反映真实泛化能力 |
| 缺文件不再被误判为 inactive | 旧 activity 检查可能跳过缺失文件 | 避免坏数据进入 inactive 样本池 |
| physical statistics 只使用 train clip | 全量统计会包含 test 分布 | 避免归一化层面的 test leakage |
| 统计 JSON 对齐 Dataset 实际字段 | 旧统计生产格式与 Dataset 读取字段不完全一致 | 减少训练启动时 key/schema 不匹配 |
| pipeline 显式保存 contact/force 路径 | 旧 Dataset 从 RGB 文件名推导物理路径 | 数据接口不再依赖固定命名规则，便于多数据源扩展 |
| action 使用相对 SO(3) 旋转 | Euler 角直接相减不能正确表达一般旋转 | 动作条件更符合机器人真实相对运动 |
| 训练 mixture 精确为 40/20/20/20 | 单纯独立随机抽任务会产生比例波动 | 实验可复现，实际任务数量可审计 |
| test 每个 clip 展开为四种任务 | 随机测试任务无法逐样本公平对比 | 可以比较同一轨迹在 T0–T3 下的表现差异 |
| 新增 condition/noise/loss mask | 旧 pipeline 不知道哪些模态是条件 | 为模型侧 modality forcing 提供明确接口 |
| T3 关闭 RGB observation 条件 | 先前错误假设四任务都有第一张 RGB | 保证 T3 真正测试 `action + contact → video + force` |
| 新增端到端严格验证 | 多阶段 JSON 容易出现数量、路径或顺序错位 | 正式训练前自动阻断错误数据 |
| 新增一键运行入口和配置 | 手工执行多条命令容易传错路径 | 提高复现性并减少操作错误 |
| 新增合成数据端到端测试 | 真实数据体积大，不适合每次开发验证 | 可以快速验证完整控制流和 T3 语义 |

## 4. VAE clip index：改了什么

新增文件：[build_vae_index.py](./build_vae_index.py)

### 4.1 原来的行为

原始 [`physical_wm/build_vae_data.py`](../physical_wm/build_vae_data.py) 会：

1. 找到具备 metadata、mask、contact、force 和审核标志的 episode。
2. 删除静止头尾。
3. 从运动区间选取固定长度 clip。
4. 写出 `clips.json`。

项目当前使用的主要 clip 参数为：

```text
n_frames = 17
frame_stride = 3
clips_per_episode = 25
trans_thresh = 1.0 mm
rot_thresh = 0.01 rad
camera = camera2
```

### 4.2 本次修改

- RGB、contact、force 三种文件必须全部存在。
- contact 必须为 `(2,H,W)`。
- force 必须为 `(6,H,W)`。
- RGB、contact、force 空间尺寸必须一致。
- 数组不能包含 NaN 或 Inf。
- 指定相机不存在时，严格模式直接拒绝 episode，不回退到混合相机 metadata。
- 每个被拒绝的 episode/clip 都写入 `rejected_vae_index.json`。
- JSON 使用原子写入，避免程序中断产生半文件。

### 4.3 修改原因

旧代码在生成 clip 时主要检查 contact 是否存在，没有同时严格确认 force 和 RGB。这样可能出现“索引成功生成，但训练加载到某一帧才失败”。

相机过滤的回退逻辑也可能在 camera2 缺失时混入其他相机记录，造成重复 frame_idx 或时序混乱。

### 4.4 预期效果

- 训练前发现缺文件、坏数组和多相机问题。
- VAE、pipeline 使用同一组可靠的 17 帧。
- 出错时能够通过 rejection report 定位 episode 和 observation 帧。

这些检查不能直接提高模型精度，但能够降低由于错误数据导致的训练崩溃、loss 异常和错误监督。

## 5. Train/test split：改了什么

新增文件：[split_train_test.py](./split_train_test.py)

### 5.1 本次修改

- 先按 `setting = episode 路径第一段` 分组。
- 每个 setting 内按完整 episode 分配 train/test。
- 同一 episode 的全部 clip 只能进入一个 split。
- 每个 clip 根据 contact 或 force 判断 active。
- 保留全部 active clip，再抽样 inactive，使 active 约占配置比例。
- activity 检查遇到缺文件、坏 shape 或 NaN/Inf 时，在严格模式下停止。
- 输出 train/test 实际数量、episode 数、active 比例和 setting 覆盖报告。

### 5.2 修改原因

同一 episode 的相邻 clip 高度相似。如果按 clip 随机切分，train 和 test 可能包含同一轨迹的相邻片段，造成测试指标虚高。

物理数据又非常稀疏。如果完全随机保留 inactive clip，大量“没有接触”的样本会淹没有效 contact/force 信号；但如果只保留 active，模型又学不到真正的无接触状态。因此采用“全部 active + 部分 inactive”。

### 5.3 预期效果

- 减少 episode 泄漏，测试结果更可信。
- 保留全部有物理信号的 clip。
- 控制零接触样本比例，降低稀疏监督被淹没的风险。
- 可以明确知道哪些 setting 没有进入 test。

当前示例配置沿用已有划分参数：

```text
train_weight:test_weight = 1000:200
active_frac = 0.85
activity_modality = contact
seed = 0
```

这些是当前参考配置，不是本次新确定的研究结论。

## 6. Physical statistics：改了什么

新增文件：[compute_physical_statistics.py](./compute_physical_statistics.py)

### 6.1 发现的问题

原 `build_vae_data.py --with_stats`：

- 在 split 之前读取全量 clip，可能包含 test episode。
- contact 使用嵌套的 `contact.scale_p99` 等字段。

但当前 `PhysicalClipDataset` 和 `ACWMDataset` 实际读取：

```text
contact_ch_max
force_ch_active_std
```

生产者和消费者的统计 schema 不完全一致。

### 6.2 本次修改

- statistics 必须读取 `clips_train.json`，不扫描全部 source_root。
- 默认对重复出现的相同物理帧去重。
- contact 计算每通道原始最大值 `contact_ch_max`。
- force 只在非零像素上计算每通道 mean/std。
- 输出 train episode 列表、clip 数、唯一帧数和抽样 seed。
- test、validation 和 inference 必须复用同一份 train statistics。

### 6.3 修改原因

使用 test 数据计算归一化参数属于信息泄漏。虽然没有直接使用 test label 训练梯度，但模型预处理已经提前知道 test 分布。

force 大约有大量零背景。如果用全部像素计算标准差，std 会被零值稀释得很小，真实力值除以很小的 std 后会被异常放大。

### 6.4 预期效果

- 避免 normalization leakage。
- 保持 contact/force 的零背景语义。
- 降低 force 归一化后极端放大和训练不稳定风险。
- 统计文件可以直接被现有 Dataset 接口读取。

## 7. Pipeline data：改了什么

新增文件：[build_pipeline_data.py](./build_pipeline_data.py)

### 7.1 本次修改

每条 pipeline sample 现在显式包含：

```text
1 个 observation RGB 路径
16 个 future RGB 路径
1 + 16 个 contact 路径
1 + 16 个 force 路径
16 × 7 action
17 个 frame_indices
稳定 base_sample_id
```

action 定义保持为：

```text
[dx, dy, dz, drx, dry, drz, gripper_abs]
```

其中：

```text
translation = t_to - t_from
rotation = EulerXYZ(R_from^-1 × R_to)
gripper = to frame 的绝对 gripper 值
```

训练 pipeline 生成 action statistics；测试 pipeline 通过 `--skip_stats` 禁止重新统计。

### 7.2 修改原因

旧 [`acwm_dataset.py`](../DiffSynth-Studio/examples/wanvideo/model_training/acwm_dataset.py) 根据 RGB 文件名猜测：

```text
images/frame_000206.png
→ modalities/contact/000206.npy
→ modalities/force/000206.npy
```

这种方式依赖目录和命名规则，不利于后续支持其他数据源，也无法明确审计物理路径是否真的与 RGB 对齐。

### 7.3 预期效果

- RGB/contact/force/action 的对应关系可直接检查。
- Dataset 不再依赖隐式文件命名。
- 后续接入其他数据集时，只需要 adapter 输出统一 schema。
- action 统计与测试数据分离。

## 8. T0–T3 mixture：改了什么

新增文件：[build_modality_mixture.py](./build_modality_mixture.py)

### 8.1 训练集修改

训练集保持“一条基础 clip 对应一条任务样本”，不会把每条训练 clip 复制四份。

任务数量先通过最大余数法计算，再按 setting 分层分配：

```text
T0 = 40%
T1 = 20%
T2 = 20%
T3 = 20%
```

固定 seed 后，相同输入会得到相同任务分配。

### 8.2 测试集修改

每个 test base clip 展开为四条：

```text
clip_A:T0
clip_A:T1
clip_A:T2
clip_A:T3
```

四个测试文件的 `base_sample_id` 集合和顺序完全相同。

### 8.3 新增字段

每条样本新增：

- `task_type`
- `action_is_condition`
- `observation_rgb_is_condition`
- `condition_modalities`
- `target_modalities`
- `condition_mask`
- `noise_mask`
- `loss_mask`
- `video_noise_frame_mask`
- `video_loss_frame_mask`

四种显式条件列表：

```text
T0: [action, observation_rgb]
T1: [action, video]
T2: [action, video, contact]
T3: [action, contact]
```

### 8.4 修改原因

只写 `task_type=T3` 不足以让训练代码正确工作。Dataset 和 loss 还必须知道：

- 哪个模态是干净条件。
- 哪个模态要加噪。
- 哪个模态要计算 loss。
- video 的第 0 帧是不是条件。

T0 与 T3 都需要生成 video，但 T0 不生成第一帧，T3 必须生成第一帧。因此需要额外的 video 帧级 mask。

### 8.5 预期效果

- 一个模型可以用统一输出头训练四种条件组合。
- 训练比例可复现、可审计。
- 可以公平比较同一测试 clip 在四种条件下的结果。
- 为后续独立 timestep 和 modality forcing loss 提供明确数据接口。

这些效果只有在模型侧真正读取 mask 后才能生效；仅生成 JSON 不会自动改变模型行为。

## 9. T3 语义修正

最初设计曾假设第一张 RGB observation 是四种任务共有条件。该假设已经删除。

当前 T3 定义为：

```text
输入：action + full contact
输出：full video + force
```

因此：

```text
observation_rgb_is_condition = false
condition_modalities = [action, contact]
video_noise_frame_mask = [1,1,...,1]  # 17个1
video_loss_frame_mask  = [1,1,...,1]  # 17个1
```

经过 Wan 因果 VAE 后，T3 的 video latent mask 应为：

```text
[1,1,1,1,1]
```

现有 loss 会无条件使用 observation 覆盖第一个 video latent。模型侧接入 T3 时必须关闭这一行为。

可能效果：T3 将真正衡量“仅凭动作与接触信息恢复/预测视觉序列”的能力，而不是偷偷使用第一张 RGB。

## 10. 数据验证：改了什么

新增文件：[check_data_pipeline.py](./check_data_pipeline.py)

验证内容包括：

- VAE clip 是否都是 17 帧。
- RGB/contact/force 文件是否存在。
- 帧号、通道数、空间尺寸是否一致。
- 是否存在 NaN/Inf。
- train/test clip 和 episode 是否重叠。
- split clip 是否在 pipeline 中一一出现。
- physical/action statistics 是否只来自 train episode。
- action 是否为有限的 `16×7`。
- train mixture 是否一条 base clip 只出现一次。
- 40/20/20/20 是否满足整数目标数量。
- 四种 task mask 是否正确。
- 四套 test 是否使用完全相同的 base clip 和顺序。

存在 error 时：

```text
validation_report.json: status = fail
脚本返回非零退出码
一键流程立即停止
```

修改原因：旧流程的很多问题只会在 GPU 训练中途暴露，代价高且难定位。

预期效果：把大部分格式、路径、对齐和泄漏问题提前到 CPU 数据检查阶段发现。

## 11. 一键执行和配置：改了什么

新增：

- [run_data_pipeline.py](./run_data_pipeline.py)
- [示例配置](./configs/omnivitac_data_build_v1.example.json)
- [QUICKSTART.md](./QUICKSTART.md)

一键入口严格按以下顺序执行：

```text
1. build VAE clip index
2. split train/test
3. compute train-only physical statistics
4. build train pipeline + action statistics
5. build test pipeline without statistics
6. build modality mixture
7. validate everything
```

入口会拒绝：

- `with_stats_before_split=true`
- 非 40/20/20/20 的任务比例
- 任一子步骤非零退出

`--dry_run` 只显示将执行的命令，不写数据。

可能效果：减少人工执行顺序错误、路径传错和实验不可复现问题。

## 12. 测试：改了什么

新增文件：[tests/test_end_to_end.py](./tests/test_end_to_end.py)

测试动态构造 10 个合成 episode，并完整运行：

```text
episode discovery
→ VAE clip
→ split
→ statistics
→ pipeline
→ mixture
→ validation
```

已验证结果：

- 测试通过。
- train mixture 共 10 条，得到 T0/T1/T2/T3 = 4/2/2/2。
- train/test episode 无重叠。
- physical statistics 不包含 test episode。
- T3 的 `observation_rgb_is_condition=false`。
- T3 条件列表为 `[action, contact]`。
- T3 的 17 帧 video noise/loss mask 全部为 1。

测试只能证明程序逻辑在受控输入上正确，不能代替真实数据上的规模、性能和分布检查。

## 13. 文件清单

| 文件 | 状态 | 用途 |
|---|---|---|
| [data_common.py](./data_common.py) | 已完成 | 公共 JSON、路径、metadata、ID、mask 工具 |
| [build_vae_index.py](./build_vae_index.py) | 已完成 | 生成并检查 VAE clip index |
| [split_train_test.py](./split_train_test.py) | 已完成 | episode 级 split 与 active/inactive 采样 |
| [compute_physical_statistics.py](./compute_physical_statistics.py) | 已完成 | train-only contact/force 统计 |
| [build_pipeline_data.py](./build_pipeline_data.py) | 已完成 | 显式多模态 pipeline 和 action |
| [build_modality_mixture.py](./build_modality_mixture.py) | 已完成 | 40/20/20/20 mixture 和四套 test |
| [check_data_pipeline.py](./check_data_pipeline.py) | 已完成 | 端到端严格检查 |
| [run_data_pipeline.py](./run_data_pipeline.py) | 已完成 | 配置驱动的一键执行 |
| [tests/test_end_to_end.py](./tests/test_end_to_end.py) | 已完成 | 合成数据端到端回归测试 |
| [MODALITY_FORCING_DATA_SPEC.md](./MODALITY_FORCING_DATA_SPEC.md) | 已完成 | 数据 schema 和实现规格 |

## 14. 预期生成结果

```text
outputs/
├── vae_index/
│   ├── clips.json
│   ├── physical_statistics.json
│   └── rejected_vae_index.json
├── split/
│   ├── clips_train.json
│   └── clips_test.json
├── pipeline/
│   ├── train/
│   │   ├── data.json
│   │   ├── statistics.json
│   │   └── rejected_pipeline.json
│   └── test/
│       ├── data.json
│       └── rejected_pipeline.json
├── mixture/
│   ├── train_mixture.json
│   ├── test_t0.json
│   ├── test_t1.json
│   ├── test_t2.json
│   ├── test_t3.json
│   └── mixture_statistics.json
└── reports/
    ├── split_report.json
    └── validation_report.json
```

## 15. 如何运行

复制配置：

```bash
cp modality_forcing_data_spec/configs/omnivitac_data_build_v1.example.json \
   modality_forcing_data_spec/configs/omnivitac_data_build_v1.local.json
```

把 `source_root` 改为真实数据根目录。

预览命令：

```bash
python modality_forcing_data_spec/run_data_pipeline.py \
  --config modality_forcing_data_spec/configs/omnivitac_data_build_v1.local.json \
  --dry_run
```

正式执行：

```bash
python modality_forcing_data_spec/run_data_pipeline.py \
  --config modality_forcing_data_spec/configs/omnivitac_data_build_v1.local.json
```

最终成功标准：

```text
outputs/reports/validation_report.json
status == "pass"
```

## 16. 当前尚未修改的模型代码

本次只完成数据侧代码。以下内容仍是下一阶段工作：

### 16.1 `ACWMDataset`

当前 [`acwm_dataset.py`](../DiffSynth-Studio/examples/wanvideo/model_training/acwm_dataset.py) 仍然：

- 从 RGB 文件名推导 contact/force 路径。
- 无条件把 observation RGB 当成输入。
- 不返回 task/noise/loss mask。
- 对包括 gripper 在内的全部 7D action 做 z-score。

后续需要改成读取本次新增的显式路径和 mask，并根据 T3 关闭 observation 条件。

### 16.2 Flow Matching loss

当前 [`flow_match_acwm.py`](../DiffSynth-Studio/diffsynth/diffusion/flow_match_acwm.py) 仍然：

- 三个模态共享一个 timestep。
- 三个模态全部加噪。
- 三个模态全部计算 loss。
- 强制要求 video first-frame condition。

后续需要：

- 支持 `t_video/t_contact/t_force`。
- 条件模态保持 clean。
- 只对目标模态计算 loss。
- video 第 0 帧根据 T0/T3 使用不同 mask。

### 16.3 DiT/MoT

当前 [`wan_video_dit_acwm_mot.py`](../DiffSynth-Studio/diffsynth/models/wan_video_dit_acwm_mot.py) 的 forward 仍只接收一个 timestep。后续需要为三个模态分别构造 time embedding，并保证联合 attention 仍能跨模态交换信息。

### 16.4 推理与评估

后续需要让推理入口分别准备四种条件组合，并在同一批 test base clip 上报告：

- video 指标。
- contact 指标。
- force 指标。
- T0–T3 间的对比。

## 17. 可能达到的总体效果

### 17.1 数据可靠性

更早发现路径、形状、帧号、相机和 NaN/Inf 问题，减少 GPU 训练中途失败。

### 17.2 实验可信度

episode 级 split 和 train-only statistics 降低数据泄漏风险，使 test 指标更可信。

### 17.3 多任务能力

模型侧完成接入后，同一个 ACWM 有可能在不同可用传感器条件下补全缺失模态，而不只支持 observation+action 的单一生成任务。

### 17.4 跨模态学习

T1/T2/T3 可能促使模型学习 video、contact、force 之间更明确的对应关系，例如：

- 从视觉运动推断接触和力。
- 从视觉+接触推断力。
- 从接触变化辅助生成视觉和力。

### 17.5 推理灵活性

如果部分真实传感器数据在 inference 时可用，可以将它作为 clean condition，而不是仍然要求模型从零生成全部模态。

### 17.6 不能提前保证的效果

当前不能保证：

- loss 一定下降。
- video/contact/force 指标一定提升。
- 多任务一定优于原 T0 单任务。
- 模型一定能泛化到新物体或新数据集。

这些结论必须在完成模型侧接入、正式训练和消融实验后验证。数据改造的直接成果是建立正确、可复现、可审计的实验输入，而不是提前证明模型效果。

## 18. 建议实验与验收

建议至少比较：

```text
Baseline：只训练原始 T0
Experiment A：T0/T1/T2/T3 = 40/20/20/20
Experiment B：四任务等比例 25/25/25/25（消融）
```

测试时对同一 base clip 分别运行四个任务，避免不同测试样本造成比较偏差。

正式验收条件：

- 数据 `validation_report.status == pass`。
- Dataset 能正确返回四类条件和 mask。
- T3 forward 中没有输入/覆盖 RGB observation。
- 条件模态确实不加噪且 loss 为 0。
- 目标模态正常加噪并产生梯度。
- 小规模 overfit test 能跑通。
- 完整训练没有明显 NaN、OOM 或模态 loss 异常。
- 四任务分别有独立评估结果。

## 19. 已知风险与待确认项

1. action 第 7 维 gripper 的归一化：数据脚本认为它不应盲目与前六维统一 z-score，但旧 Dataset 当前会这样做，需要统一。
2. 三个模态是否需要三个完全独立随机 timestep，还是只需独立 clean/noisy 状态，需要在模型实现前确认。
3. T0 中 observation 时刻的 contact/force 是否全部作为预测目标，当前按旧训练逻辑保留。
4. 多任务比例 40/20/20/20 已固定，但 train:test 权重和 active_frac 仍属于可调整数据配置。
5. 大规模开启完整数组检查会增加预处理时间，需要在服务器上评估耗时，但正式产数不建议关闭。

## 20. DROID 状态

DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset 当前不纳入本次实现。

后续需要单独确认：

- 相机视角和频率。
- action 维度、坐标系、单位和绝对/相对语义。
- gripper 表示。
- episode 边界。
- 是否存在 contact/force ground truth。

如果 DROID 缺少 contact/force，不能用全零伪造监督，因为全零在当前项目中本身表示“真实无接触”。应通过 `available_modalities` 和 loss mask 明确标记缺失监督。

