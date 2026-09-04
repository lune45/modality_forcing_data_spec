# Modality Forcing 数据构建技术规格

> 文档状态：v0.3（完整数据链路，已配套实现）  
> 当前数据源：现有 OmniViTac/本地已处理数据  
> 暂不纳入：DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset  
> 训练任务比例：T0 40%，T1 20%，T2 20%，T3 20%

## 1. 文档目的

本文档用于指导下一步代码实现。起点是已经完成 RGB、SAM、contact、force 预处理的 episode；在此基础上生成 VAE clip index、划分 train/test、生成 pipeline data，最后生成四类 modality forcing 训练样本及对应测试清单。目标不是重新运行 SAM 或重新生成底层 RGB/contact/force 文件。

当前阶段的核心产物是数据索引 JSON，而不是复制图片或 `.npy` 文件。一个新样本只需要引用现有文件，并额外记录它属于哪种任务、哪些模态是条件、哪些模态需要加噪和计算 loss。

## 2. 项目目标（一句话）

给模型不同组合的已知信息，让同一个模型学会补全未知的 video、contact 和 force。

- action：机器人接下来执行什么动作；始终是条件。
- observation RGB：当前时刻的第一张 RGB 图；它只在任务明确提供 RGB/video 时才是条件，不是四个任务的共有输入。
- video：完整序列为当前 1 帧加未来 16 帧；不同任务需要生成的范围不同。
- contact：左右手指的二维接触强度图。
- force：左右手指的三维形变/力代理场。

这里的“三个模态”是 `video`、`contact`、`force`。`action` 是控制条件，不计入这三个输出模态。

## 3. 当前阶段范围

### 3.1 本阶段要做

1. 从已完成 SAM 和 contact/force 预处理的 episode 生成 VAE clip index。
2. 按 setting 和 episode 划分 train/test，保证 episode 零泄漏。
3. 分别从 train/test clip index 生成 pipeline data，并附加 RGB、action、contact、force 的显式路径。
4. 给训练 pipeline clip 分配 T0/T1/T2/T3 标签，比例严格为 40/20/20/20。
5. 为每条样本写入条件模态、目标模态、noise mask 和 loss mask。
6. 将每条测试 clip 展开为四条任务样本，分别构建 T0/T1/T2/T3 测试集。
7. 生成统计报告并执行完整性、比例、路径、形状和数据泄漏检查。
8. 为后续修改 `ACWMDataset` 和训练 loss 提供稳定的数据接口。

### 3.2 本阶段不做

1. 不重新运行 SAM。
2. 不重新生成现有 contact/force 文件。
3. 不修改 VAE、DiT、MoT 或推理代码。
4. 不开始模型训练。
5. 不接入 DROID；等论文和数据格式确认后再单独设计 adapter。
6. 不把缺失模态伪造为全零 ground truth。

## 4. 现有数据基础

现有 pipeline 样本已经包含以下信息：

- `episode`：样本所属 episode。
- `obs_frame_idx`：观察帧编号。
- `observation_frame`：当前 RGB 观察帧。
- `frames`：未来 16 张 RGB 图。
- `actions`：16 个动作，每个动作 7 维。
- `observation_contact_path`：观察时刻 contact。
- `contact_path`：未来 16 个 contact。
- `observation_force_path`：观察时刻 force。
- `force_path`：未来 16 个 force。

因此每条基础 clip 可还原为：

```text
RGB:      1 张 observation + 16 张 future = 17 帧
contact:  1 个 observation + 16 个 future = 17 帧
force:    1 个 observation + 16 个 future = 17 帧
action:   16 个，每个 7 维，对应 16 次状态转移
```

现有样例索引见：

```text
physical_wm/pipeline_200.json
```

注意：新代码必须直接读取显式的 contact/force 路径字段，不能只依赖从 RGB 文件名猜测物理模态路径。

### 4.1 三项上游任务和完整执行顺序

学长所说的三项工作对应当前代码中的三个阶段：

```text
已处理 episode
    │
    ├── metadata.json、images/、modalities/contact/、modalities/force/
    │
    ▼
1. generate VAE data
   modality_forcing_data_spec/build_vae_index.py
   输出全量 clips.json
    │
    ▼
2. split train / test set
   modality_forcing_data_spec/split_train_test.py
   输出 clips_train.json、clips_test.json
    │
    ▼
2.5 用 clips_train.json 计算 physical_statistics.json
    contact/force 归一化统计只能来自 train
    │
    ▼
3. generate pipeline data
   对 train/test 分别运行 modality_forcing_data_spec/build_pipeline_data.py
   输出 train/data.json、test/data.json，以及训练 action statistics.json
    │
    ▼
4. generate modality-forcing mixture
   为 train 分配 T0/T1/T2/T3，为 test 建立四个任务视图
```

虽然会议中列举任务的语序可能不同，工程上必须先完成 episode 级 split，再分别生成 train/test pipeline，防止同一 episode 的相邻 clip 泄漏。

### 4.2 上游输入 episode 的最低要求

`source_root` 下每个可用 episode 至少应具备：

```text
<source_root>/<setting>/<task>/<episode>/
├── metadata.json
├── masks.json
├── double_checked.flag
├── images/
│   └── frame_XXXXXX.png
└── modalities/
    ├── contact/
    │   └── XXXXXX.npy
    └── force/
        └── XXXXXX.npy
```

其中：

- `metadata.json` 提供 `frame_idx`、相机、末端位姿和 gripper。
- `masks.json` 与 `double_checked.flag` 是当前 `build_vae_data.py` 判断 episode 已完成审核的门槛。
- `images` 是 pipeline data 的 RGB 来源。
- `contact` 单帧预期为 `(2,H,W)`。
- `force` 单帧预期为 `(6,H,W)`。
- 当前相机默认使用 `camera2`。

缺少上述任一关键内容的 episode 不应悄悄混入训练。构建脚本应统计被跳过的 episode/clip 数并给出原因。

当前 metadata 加载逻辑在找不到 `camera2` 时会退回使用未过滤的全部 metadata。若文件含多相机记录，这可能造成重复帧号或时序混合。正式实现建议使用严格模式：指定相机不存在时拒绝该 episode，并写入 rejection report。

### 4.3 第一阶段：generate VAE data

#### 4.3.1 “VAE data”的准确含义

新实现 [`build_vae_index.py`](./build_vae_index.py) 参考了原有 [`physical_wm/build_vae_data.py`](../physical_wm/build_vae_data.py)。它不负责重新运行 SAM，也不负责生成 contact/force `.npy`，而是负责：

1. 找到审核完成的 episode。
2. 从每个 episode 中选择 17 帧 clip。
3. 为每个 clip 记录 contact/force 文件路径。
4. 输出供 ContactVAE/ForceVAE 使用的 `clips.json`。
5. 当前脚本可选读取 `.npy` 计算物理模态归一化统计；正式数据构建时必须改为只从 split 后的 train clip 计算。

因此本阶段的输入是“已经生成好 RGB/contact/force 的 episode”，输出是“VAE 训练可以读取的 clip 索引”。

#### 4.3.2 静止头尾裁剪

脚本先按 metadata 顺序计算相邻原始帧的运动量：

```text
translation_delta = ||t[i] - t[i-1]||
rotation_delta    = wrap_to_pi(rpy[i] - rpy[i-1]) 的 L2 范数
moving            = translation_delta > 1.0 mm
                    或 rotation_delta > 0.01 rad
```

默认阈值：

- `trans_thresh = 1.0 mm`
- `rot_thresh = 0.01 rad`

只删除第一个运动帧之前和最后一个运动帧之后的静止部分。运动区间中间即使暂时静止也保留，因为这些停顿可能包含接触、抓稳或释放信息。

若整个 episode 没有超过阈值的运动，或有效区间放不下一个完整 clip，则跳过该 episode。

#### 4.3.3 clip 结构

当前项目采用：

```text
n_frames = 17
frame_stride = 3
clips_per_episode = 25（当前本地正式索引采用值）
```

一个 clip 的 metadata 位置为：

```text
s, s+3, s+6, ..., s+48
```

对应：

- 第 0 帧：观察/序列起点。
- 第 1–16 帧：16 个后续时间点。
- 相邻采样时间点之间跨 3 个原始帧。
- 16 个 action 对应这 17 个时间点之间的 16 次转移。

脚本不会做密集滑窗，而是贪心选择起点：优先选择能覆盖更多未被其他 clip 精确使用的帧；相同覆盖增益时，选择离已有起点更远的位置。每个 episode 最多选 `clips_per_episode` 条。

#### 4.3.4 `clips.json` 格式

```json
{
  "config": {
    "frame_stride": 3,
    "n_frames": 17,
    "clips_per_episode": 25,
    "trans_thresh": 1.0,
    "rot_thresh": 0.01,
    "camera": "camera2",
    "n_episodes": 237,
    "n_clips": 5925
  },
  "clips": [
    {
      "episode": "setting/task/0000",
      "frame_indices": [19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67],
      "contact_paths": [
        "modalities/contact/000019.npy"
      ],
      "force_paths": [
        "modalities/force/000019.npy"
      ]
    }
  ]
}
```

示例路径数组省略了其余元素；真实 `contact_paths` 和 `force_paths` 都必须有 17 项，并与 `frame_indices` 逐项对应。此阶段的物理路径相对于 episode 目录。

完成下一阶段 split 后：

- ContactVAE 的训练/测试分别读取 `clips_train.json`/`clips_test.json` 中的 `contact_paths`。
- ForceVAE 的训练/测试分别读取相同两个索引中的 `force_paths`。
- 对 VAE 来说 17 帧是一个完整重建序列，不使用 T0–T3 标签；T0–T3 属于后面的 world-model pipeline 训练任务。

#### 4.3.5 VAE 归一化统计

这里存在两个必须修复的问题：统计数据来源不能包含 test，且当前生产者与消费者的 JSON schema 不一致。

当前 `build_vae_data.py --with_stats` 会在尚未 split 的全量 `clips.json` 上输出 `statistics.json`：

- contact：对每个手指通道收集非零值，先做平方根，再计算 `p99` 和最大值。
- force：对 6 个通道分别在非零像素上计算 mean/std。
- `--n_sample_frames 3000`：默认最多抽样 3000 个 clip-frame 条目。
- `--stats_full`：使用全部条目。
- `--seed`：只影响统计抽样顺序，不影响 clip 起点选择。

但当前 `PhysicalClipDataset`/`ACWMDataset` 实际读取的是另一种顶层字段：

```json
{
  "contact_ch_max": ["2 个通道的原始最大值"],
  "force_ch_active_mean": ["6 个通道的非零均值"],
  "force_ch_active_std": ["6 个通道的非零标准差"],
  "source_clip_index": "clips_train.json",
  "num_episodes": 0,
  "num_clips": 0,
  "num_frames": 0
}
```

因此在不同时修改 Dataset 归一化公式的前提下，不能直接把 `build_vae_data.py --with_stats` 产生的嵌套 `contact.scale_p99` schema 当作正式训练统计。最终文档以实际 Dataset 消费接口为准：

- contact：对 train clip 覆盖到的原始 contact 值计算每通道最大值 `contact_ch_max`；归一化时只做除法，不减均值，保持零背景。
- force：对 train clip 覆盖到的非零像素计算每通道 `force_ch_active_mean/std`；当前 Dataset 实际按 `force_ch_active_std` 缩放并保持零背景。
- 所有分母用 epsilon 防止除零。
- 统计中必须拒绝 NaN/Inf 和错误通道数。

正式训练也不能使用“全量 clips（含未来 test episode）”算出的归一化参数，否则属于统计信息泄漏。正确做法是：

1. 本阶段先只生成全量 `clips.json`。
2. 完成第 4.4 节的 train/test split。
3. 只遍历 `clips_train.json` 中列出的 contact/force 路径。
4. 按上述 Dataset 所需 schema 生成最终 `vae_index/physical_statistics.json`。
5. train/test/validation/inference 全部复用这一份 train statistics。

配套实现已采用独立的 `compute_physical_statistics.py --clips <clips_train.json>`，严格按 train clip index 计算并输出上述 schema。原有 `compute_physical_stats.py` 虽然输出键与 Dataset 更接近，但它会扫描 root 中的 episode，不能直接保证只用 train split，因此没有作为本流程入口。

#### 4.3.6 建议命令

```bash
python modality_forcing_data_spec/build_vae_index.py \
  --source_root /path/to/processed_root \
  --out_dir modality_forcing_data_spec/outputs/vae_index \
  --camera camera2 \
  --frame_stride 3 \
  --n_frames 17 \
  --clips_per_episode 25 \
  --trans_thresh 1.0 \
  --rot_thresh 0.01 \
  --seed 0
```

上面故意不传 `--with_stats`：先生成全量 clip index，split 后再用 train clip 计算最终 physical statistics。

#### 4.3.7 必须补强的检查

当前代码在构建 clip 时显式检查了 contact 路径，但没有同时显式检查每个 force 路径。下一步代码应修正为：

```text
17 个 RGB + 17 个 contact + 17 个 force 全部存在，clip 才能保留。
```

还必须检查数组形状、NaN/Inf、帧号一致性，并输出丢弃原因统计。

### 4.4 第二阶段：split train / test set

#### 4.4.1 输入和输出

新实现：[`split_train_test.py`](./split_train_test.py)，算法参考原有 [`physical_wm/split_train_test_v2.py`](../physical_wm/split_train_test_v2.py)。

输入：

- 第一阶段的全量 `clips.json`。
- `source_root`，用于读取 contact 或 force 文件判断 clip 是否 active。

输出：

- `clips_train.json`
- `clips_test.json`

二者保持与 `clips.json` 相同的 `{config, clips}` 结构，并在 `config.split` 中记录划分参数。

#### 4.4.2 active clip 定义

通过 `--modality contact` 或 `--modality force` 选择用于判断 active 的物理模态。

只要一个 clip 的任意一帧满足：

```text
max(abs(physical_array)) > 1e-6
```

该 clip 就视为 active。当前已有 `clips_train_0730.json`/`clips_test_0730.json` 使用的是 `contact`。

当前脚本遇到不存在的物理文件时会跳过该帧，严重时可能把“文件缺失”误判成 inactive。正式运行 split 前必须先完成路径检查；缺文件的 clip 应进入 rejection report，而不能作为 inactive 样本参与抽样。

#### 4.4.3 episode 级划分原则

1. `setting = episode` 路径的第一段。
2. 每个 setting 内单独划分 episode。
3. 同一个 episode 的所有 clip 必须全部进入 train 或全部进入 test。
4. 某 setting 有至少 2 个 episode 时，尽量保证 train/test 都包含该 setting。
5. 某 setting 只有 1 个 episode 时，当前代码把它放入 train，并警告 test 缺少该 setting。
6. episode 大小不同，所以 train:test 是近似比例，不是精确 clip 数量。

`--n_train` 和 `--n_test` 是权重。例如 `1000:200` 表示期望约 `83.33%:16.67%`，并不表示恰好输出 1000 和 200 条 clip。

脚本主要按 active clip 数量寻找接近目标比例的 episode 子集；如果该 setting 没有 active clip，则退化为按总 clip 数/episode 数进行平衡。

#### 4.4.4 active/inactive 采样

在 train 和 test 各自的 episode 集合内：

1. 保留全部 active clip。
2. 随机抽取一部分 inactive clip。
3. 使 active clip 最终约占 `active_frac`。

若 active 数量为 `A`、目标 active 比例为 `r`：

```text
inactive_target = round(A × (1-r) / r)
```

当前已有划分使用 `active_frac=0.85`。若可用 inactive 不足，实际比例会偏高，脚本必须打印 warning 和实际统计。

#### 4.4.5 建议命令

下面的 `1000:200`、`active_frac=0.85`、`seed=0` 与当前本地 `0730` 划分配置一致；若学长另有要求，应通过配置修改而不是写死：

```bash
python modality_forcing_data_spec/split_train_test.py \
  --clips modality_forcing_data_spec/outputs/vae_index/clips.json \
  --source_root /path/to/processed_root \
  --modality contact \
  --n_train 1000 \
  --n_test 200 \
  --active_frac 0.85 \
  --out_train modality_forcing_data_spec/outputs/split/clips_train.json \
  --out_test modality_forcing_data_spec/outputs/split/clips_test.json \
  --report modality_forcing_data_spec/outputs/reports/split_report.json \
  --seed 0
```

#### 4.4.6 split 验收

必须检查：

```text
train_episode_set ∩ test_episode_set = 空集
```

同时记录：

- train/test clip 数和 episode 数。
- 每个 setting 的 train/test episode 数。
- 每个 split 的 active/inactive 数量和实际 active 比例。
- 缺失于 train 或 test 的 setting。
- 请求比例与实际比例。

任何 episode overlap 都必须直接报错并停止，不能降级为 warning。

#### 4.4.7 split 后生成 train-only physical statistics

完成 split 后，使用 `clips_train.json` 中的 17 帧 contact/force 路径计算 VAE/物理模态归一化统计：

```text
输入：clips_train.json + source_root
输出：outputs/vae_index/physical_statistics.json
禁止读取：clips_test.json 中独占 episode 的任何数组
```

计算公式和输出 schema 沿用第 4.3.5 节。统计输出还应记录 `source_clip_index`、train episode 数、clip 数、实际抽样帧数和 seed，以便确认它确实来自 train split。

```bash
python modality_forcing_data_spec/compute_physical_statistics.py \
  --clips modality_forcing_data_spec/outputs/split/clips_train.json \
  --source_root /path/to/processed_root \
  --out modality_forcing_data_spec/outputs/vae_index/physical_statistics.json \
  --n_sample_frames 3000 \
  --seed 0
```

### 4.5 第三阶段：generate pipeline data

#### 4.5.1 输入和运行方式

新实现：[`build_pipeline_data.py`](./build_pipeline_data.py)，action 计算参考原有 [`physical_wm/build_pipeline_data_v2.py`](../physical_wm/build_pipeline_data_v2.py)。

它不重新选择 clip，而是读取第二阶段产生的 clip index，为每条 clip：

1. 从 metadata 找到 17 个 `frame_idx` 对应的 entry。
2. 生成 1 个 observation RGB 路径和 16 个 future RGB 路径。
3. 计算相邻采样时间点之间的 16 个 7D action。
4. 生成 action normalization statistics。

必须分别对 `clips_train.json` 和 `clips_test.json` 运行，输出到不同目录。

#### 4.5.2 action 定义

每个 action：

```text
[dx, dy, dz, drx, dry, drz, gripper_abs]
```

对于 `frame_indices[i] → frame_indices[i+1]`：

```text
translation = t_to - t_from                         # mm，base frame
rotation    = EulerXYZ(R_from^-1 × R_to)            # rad，相对 SO(3) 旋转
gripper     = destination frame 的绝对 gripper 值
```

不能直接用两个 Euler 角逐元素相减替代相对旋转矩阵计算，因为旋转存在周期和组合顺序问题。

`action_stride` 默认继承 clip index 的 `config.frame_stride`。若二者不一致，动作尺度和帧序列会错位，必须报错或至少在严格模式下拒绝继续。

#### 4.5.3 action statistics

统计所用 episode 只来自当前 clip index，不扫描 source_root 下的全部目录。每个入选 episode 先裁掉静止头尾，再在整个运动区间上计算相隔 `action_stride` 的动作对。

推荐：

- 只在 train clip index 上生成 action `statistics.json`。
- test 构建使用 `--skip_stats`。
- 训练、验证和推理统一使用 train statistics，不能分别用 test 数据计算归一化参数。
- 前 6 维如何标准化沿用训练配置。
- 第 7 维 `gripper_abs` 是双峰/离散控制语义，当前脚本注明不应与前 6 维直接统一 z-score；在开始训练前需要统一 Dataset 中的实际实现。

#### 4.5.4 当前输出和 modality forcing 所需扩展

当前 `build_pipeline_data_v2.py` 的 `data.json` 只明确输出：

```json
{
  "episode": "setting/task/episode",
  "obs_frame_idx": 203,
  "observation_frame": "setting/task/episode/images/frame_000203.png",
  "frames": ["16 个 future RGB 路径"],
  "actions": [["16 × 7"]]
}
```

这对旧 Dataset 依靠 RGB 文件名推导物理路径的方式勉强够用，但不足以可靠支持新的 modality forcing。下一步代码必须把 clip index 中已有的物理路径一并带入：

```json
{
  "observation_contact_path": "setting/task/episode/modalities/contact/000203.npy",
  "contact_path": ["16 个 future contact 路径"],
  "observation_force_path": "setting/task/episode/modalities/force/000203.npy",
  "force_path": ["16 个 future force 路径"]
}
```

路径转换规则：

```text
clip index:  modalities/contact/000203.npy  （相对 episode）
pipeline:    setting/task/episode/modalities/contact/000203.npy
             （相对统一 source_root）
```

如果输入路径已经是绝对路径或已经带 episode 前缀，不得重复拼接。建议用统一的路径规范化函数处理。

#### 4.5.5 pipeline 顶层格式

```json
{
  "source": "/path/to/clips_train.json",
  "n_samples": 3562,
  "samples": []
}
```

`n_samples` 必须等于 `len(samples)`。若 metadata 缺失或 frame_idx 找不到，当前脚本会丢弃 clip；新代码必须记录丢弃的 `base_sample_id` 和原因，并确保不会让 train/test 对齐检查失真。

#### 4.5.6 建议命令

训练 pipeline：

```bash
python modality_forcing_data_spec/build_pipeline_data.py \
  --source_root /path/to/processed_root \
  --clip_index modality_forcing_data_spec/outputs/split/clips_train.json \
  --out_dir modality_forcing_data_spec/outputs/pipeline/train \
  --camera camera2 \
  --split train \
  --compute_action_stats
```

测试 pipeline：

```bash
python modality_forcing_data_spec/build_pipeline_data.py \
  --source_root /path/to/processed_root \
  --clip_index modality_forcing_data_spec/outputs/split/clips_test.json \
  --out_dir modality_forcing_data_spec/outputs/pipeline/test \
  --camera camera2 \
  --split test \
  --skip_stats
```

新实现已经按第 4.5.4 节输出显式 contact/force 路径，不再依靠 RGB 文件名猜测。

### 4.6 三个上游阶段的交接约束

每一级输出都必须能通过稳定 ID 与上一级一一对应：

```text
base_sample_id = dataset_source + ":" + episode + ":" + obs_frame_idx
```

必须保持：

- VAE clip 的 17 个 `frame_indices` 与 pipeline 的 RGB/contact/force 17 个路径完全对应。
- split 只筛选/分组 clip，不修改帧序列。
- pipeline 只补充 RGB/action 和规范化路径，不重新采样帧。
- modality mixture 只添加任务语义和 mask，不修改底层媒体、action 或 split。
- 任一阶段若丢弃样本，必须有机器可读的 rejection report。

## 5. 四种任务的正式定义

### 5.1 约定

- `observation_rgb`：第一张 RGB 图。T0 将它作为条件；T1/T2 的 `full_video` 自然包含它；T3 不把它作为输入条件。
- `full_video`：`observation_frame + frames`，共 17 帧。
- `full_contact`：`observation_contact_path + contact_path`，共 17 帧。
- `full_force`：`observation_force_path + force_path`，共 17 帧。
- 条件模态：保持干净，不加噪，不计算该模态的生成 loss。
- 目标模态：按 scheduler 加噪，模型负责去噪，并计算该模态 loss。
- 对 video 而言，是否保留第一张 observation RGB 取决于任务：T0 保留首帧并只生成未来 16 帧；T1/T2 已知完整 video；T3 没有 RGB observation，需要生成包括第 0 帧在内的完整 17 帧 video。

### 5.2 任务表

| 任务 | 输入条件 | 模型需要生成 | 训练占比 | video | contact | force |
|---|---|---|---:|---|---|---|
| T0 | observation RGB + action | future video + contact + force | 40% | 目标（首帧除外） | 目标 | 目标 |
| T1 | full video + action | contact + force | 20% | 条件 | 目标 | 目标 |
| T2 | full video + full contact + action | force | 20% | 条件 | 条件 | 目标 |
| T3 | full contact + action | full video + force | 20% | 目标（含首帧） | 条件 | 目标 |

### 5.3 模态 mask

mask 使用对象而不是位置数组，避免把 `[video, contact, force]` 的顺序写错。

| 任务 | `condition_mask` | `noise_mask` | `loss_mask` |
|---|---|---|---|
| T0 | `{video:0, contact:0, force:0}` | `{video:1, contact:1, force:1}` | `{video:1, contact:1, force:1}` |
| T1 | `{video:1, contact:0, force:0}` | `{video:0, contact:1, force:1}` | `{video:0, contact:1, force:1}` |
| T2 | `{video:1, contact:1, force:0}` | `{video:0, contact:0, force:1}` | `{video:0, contact:0, force:1}` |
| T3 | `{video:0, contact:1, force:0}` | `{video:1, contact:0, force:1}` | `{video:1, contact:0, force:1}` |

`condition_mask.video` 表示“完整 video 是否作为条件”，不能单独表达 T0 只提供第一帧的情况。因此还必须读取 `observation_rgb_is_condition`：T0 为 `true`，T1/T2 因完整 video 已知也为 `true`，T3 为 `false`。

video 还需要帧级 mask，顺序对应 `[observation_frame] + frames` 的 17 帧：

| 任务 | `observation_rgb_is_condition` | `video_noise_frame_mask` | `video_loss_frame_mask` |
|---|---:|---|---|
| T0 | true | `[0, 1, 1, ..., 1]` | `[0, 1, 1, ..., 1]` |
| T1 | true | `[0, 0, 0, ..., 0]` | `[0, 0, 0, ..., 0]` |
| T2 | true | `[0, 0, 0, ..., 0]` | `[0, 0, 0, ..., 0]` |
| T3 | false | `[1, 1, 1, ..., 1]` | `[1, 1, 1, ..., 1]` |

每个数组真实长度必须为 17。表里的省略号只用于展示。

Wan 的因果 VAE 会把 17 个时间帧压缩为 5 个 latent 时间位置，因此训练代码还要把帧级语义转换为 latent 级 mask：

```text
T0 video latent mask: [0, 1, 1, 1, 1]
T1 video latent mask: [0, 0, 0, 0, 0]
T2 video latent mask: [0, 0, 0, 0, 0]
T3 video latent mask: [1, 1, 1, 1, 1]
```

这里最关键的区别是：T0 可以沿用“第一个 video latent 用 observation 覆盖”的做法；T3 必须关闭该覆盖，让第一个 video latent 也从噪声中生成。

## 6. 推荐输出目录和文件

下一步代码应在本目录中生成或维护以下文件：

```text
modality_forcing_data_spec/
├── README.md                              # 项目全链路与代码逐行导读
├── QUICKSTART.md                          # 快速运行说明
├── MODALITY_FORCING_DATA_SPEC.md          # 技术规格
├── data_common.py                         # 公共 schema、路径、metadata、mask 工具
├── build_vae_index.py                     # 生成 VAE clip index
├── split_train_test.py                    # episode 级 train/test 划分
├── compute_physical_statistics.py         # train-only contact/force 统计
├── build_pipeline_data.py                 # 显式四信号 pipeline 数据
├── build_modality_mixture.py              # 生成训练/测试任务清单
├── check_data_pipeline.py                 # 端到端数据检查
├── run_data_pipeline.py                   # 按配置一键运行
├── configs/
│   └── omnivitac_data_build_v1.example.json
├── tests/
│   └── test_end_to_end.py                 # 合成数据端到端测试
└── outputs/                                # 待生成，不复制原始媒体
    ├── vae_index/
    │   ├── clips.json
    │   └── physical_statistics.json
    ├── split/
    │   ├── clips_train.json
    │   └── clips_test.json
    ├── pipeline/
    │   ├── train/
    │   │   ├── data.json
    │   │   └── statistics.json
    │   └── test/
    │       └── data.json
    ├── mixture/
    │   ├── train_mixture.json
    │   ├── test_t0.json
    │   ├── test_t1.json
    │   ├── test_t2.json
    │   ├── test_t3.json
    │   └── mixture_statistics.json
    └── reports/
        ├── rejected_samples.json
        └── validation_report.json
```

如果后续代码位置需要遵循学长仓库结构，可以移动脚本，但 JSON 格式和验证规则应保持一致。

## 7. 输出 JSON 总体结构

推荐使用带元信息的顶层对象，而不是裸列表：

```json
{
  "schema_version": "modality_forcing_v1",
  "dataset_source": "omnivitac",
  "split": "train",
  "random_seed": 42,
  "task_ratios": {
    "T0": 0.4,
    "T1": 0.2,
    "T2": 0.2,
    "T3": 0.2
  },
  "n_samples": 1000,
  "samples": []
}
```

## 8. 单条样本格式

### 8.1 推荐完整格式

```json
{
  "sample_id": "omnivitac:switch_box/task_name/0056:203:T0",
  "base_sample_id": "omnivitac:switch_box/task_name/0056:203",
  "dataset_source": "omnivitac",
  "split": "train",
  "task_type": "T0",

  "episode": "switch_box/task_name/0056",
  "obs_frame_idx": 203,
  "matched_dist": 4,

  "observation_frame": "switch_box/task_name/0056/images/frame_000203.png",
  "frames": [
    "switch_box/task_name/0056/images/frame_000206.png"
  ],

  "observation_contact_path": "switch_box/task_name/0056/modalities/contact/000203.npy",
  "contact_path": [
    "switch_box/task_name/0056/modalities/contact/000206.npy"
  ],

  "observation_force_path": "switch_box/task_name/0056/modalities/force/000203.npy",
  "force_path": [
    "switch_box/task_name/0056/modalities/force/000206.npy"
  ],

  "actions": [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  ],

  "available_modalities": {
    "video": 1,
    "contact": 1,
    "force": 1,
    "action": 1
  },
  "action_is_condition": true,
  "observation_rgb_is_condition": true,
  "condition_modalities": ["action", "observation_rgb"],
  "target_modalities": ["video", "contact", "force"],
  "video_noise_frame_mask": [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  "video_loss_frame_mask": [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  "condition_mask": {
    "video": 0,
    "contact": 0,
    "force": 0
  },
  "noise_mask": {
    "video": 1,
    "contact": 1,
    "force": 1
  },
  "loss_mask": {
    "video": 1,
    "contact": 1,
    "force": 1
  }
}
```

示例中各路径数组只展示了一个元素；真实样本必须是 16 个 future 元素。

### 8.2 字段说明

| 字段 | 类型 | 含义 |
|---|---|---|
| `sample_id` | string | 当前任务样本的唯一 ID，包含任务类型 |
| `base_sample_id` | string | 不含任务类型的基础 clip ID，用于防止泄漏和测试对齐 |
| `dataset_source` | string | 当前固定为 `omnivitac` |
| `split` | string | `train` 或 `test` |
| `task_type` | string | `T0`、`T1`、`T2`、`T3` |
| `episode` | string | 原 episode 相对路径/标识 |
| `obs_frame_idx` | int | 第一张观察帧编号 |
| `observation_frame` | string | 第一张 RGB 路径 |
| `frames` | list[string] | 16 张未来 RGB 路径 |
| `actions` | list[list[number]] | `16 × 7` 动作 |
| `observation_contact_path` | string | 第 0 时刻 contact |
| `contact_path` | list[string] | 16 个未来 contact 路径 |
| `observation_force_path` | string | 第 0 时刻 force |
| `force_path` | list[string] | 16 个未来 force 路径 |
| `available_modalities` | object | 原始样本真实拥有的模态 |
| `action_is_condition` | bool | action 是否作为输入条件；当前四种任务必须全部为 true |
| `observation_rgb_is_condition` | bool | 第一张 RGB 是否属于输入条件；T0/T1/T2 为 true，T3 为 false |
| `condition_modalities` | list[string] | 显式输入列表：T0=`action,observation_rgb`；T1=`action,video`；T2=`action,video,contact`；T3=`action,contact` |
| `target_modalities` | list[string] | 需要生成并计算 loss 的完整模态名称 |
| `video_noise_frame_mask` | list[int] | 长度 17；逐帧表示 video 是否加噪 |
| `video_loss_frame_mask` | list[int] | 长度 17；逐帧表示 video 是否计算生成 loss |
| `condition_mask` | object | 哪些完整模态是已知条件 |
| `noise_mask` | object | 哪些模态需要加噪 |
| `loss_mask` | object | 哪些模态需要计算 loss |

## 9. 训练集 mixture 的生成算法

### 9.1 推荐策略

训练集保持“一条基础 clip 对应一条任务样本”，不复制媒体文件，也不把每条 clip 扩成四份。这样总体样本数不膨胀，并能精确控制四类任务比例。

步骤：

1. 读取已经划分好的 train 基础 clip。
2. 检查每条 clip 的完整性。
3. 按 `episode` 或上层 setting/task 分组。
4. 使用固定随机种子 `42`，在组内打乱。
5. 按 40/20/20/20 分配 T0/T1/T2/T3。
6. 根据任务表写入四类 mask。
7. 保存 `train_mixture.json`。
8. 再读回输出文件，做一次独立验证。

### 9.2 为什么优先分层分配

若直接对全部 clip 打乱，小任务或小场景可能恰好只落到某一类任务中。推荐尽可能按 setting/task 分层，让每个场景内部也接近 40/20/20/20。

当一个组样本太少，无法独立满足四种任务时：

1. 先在组内按最大余数法分配整数数量。
2. 再从全局层面修正剩余名额。
3. 最终全局数量必须严格满足目标计数。

### 9.3 整数计数规则

设训练样本总数为 `N`：

```text
n_T0 = floor(N × 0.40)
n_T1 = floor(N × 0.20)
n_T2 = floor(N × 0.20)
n_T3 = N - n_T0 - n_T1 - n_T2
```

最后一个任务接收舍入余数。也可使用最大余数法，但必须在统计文件中记录实际数量和实际比例。

### 9.4 可复现要求

- 固定 `random_seed=42`。
- 相同输入文件、相同配置、相同代码版本必须生成完全相同的任务分配。
- 输入样本先按 `base_sample_id` 排序，再执行带 seed 的打乱，避免文件原始顺序变化导致结果漂移。

## 10. 测试集生成算法

测试阶段的目的不是维持 40/20/20/20，而是公平比较同一个基础 clip 在四种任务下的表现。

因此每条 test 基础 clip 应展开四份：

```text
base clip A → A:T0, A:T1, A:T2, A:T3
base clip B → B:T0, B:T1, B:T2, B:T3
```

分别输出：

- `test_t0.json`
- `test_t1.json`
- `test_t2.json`
- `test_t3.json`

四个文件中的 `base_sample_id` 集合和顺序必须完全相同。这样才能对同一条轨迹做逐样本横向比较。

## 11. 数据泄漏规则

必须先按 episode 完成 train/test 划分，再生成任务 mixture。禁止先把 clip 扩展或分配任务，再随机拆 train/test。

必须满足：

```text
train episode 集合 ∩ test episode 集合 = 空集
train base_sample_id 集合 ∩ test base_sample_id 集合 = 空集
```

如果同一 episode 的相邻 clip 分别进入 train 和 test，即使 `obs_frame_idx` 不同，也属于数据泄漏。

## 12. 路径与数据形状检查

每条样本必须通过以下检查：

### 12.1 基础字段

- 必须包含所有必填 key。
- `task_type` 只能是 T0/T1/T2/T3。
- `sample_id` 全局唯一。
- `base_sample_id` 的构造稳定且不依赖绝对路径。
- `dataset_source == "omnivitac"`。

### 12.2 序列长度

- `len(frames) == 16`
- `len(contact_path) == 16`
- `len(force_path) == 16`
- `len(actions) == 16`
- 每个 action 的长度为 7。

### 12.3 文件存在性

- observation RGB 存在。
- 16 张 future RGB 全部存在。
- observation contact 和 16 个 future contact 全部存在。
- observation force 和 16 个 future force 全部存在。
- 路径应相对于统一的 `dataset_root` 解析，JSON 中不硬编码某台服务器的绝对路径。

### 12.4 内容形状

- 所有 RGB 帧分辨率一致；当前预期为 480×640，实际检查时以配置为准。
- contact 单帧预期形状为 `(2, H, W)`。
- force 单帧预期形状为 `(6, H, W)`。
- 数值不能含 NaN 或 Inf。
- contact/force 全零可以是“真实无接触”，不能仅凭全零判定文件错误。

### 12.5 时序一致性

- RGB、contact、force 的帧编号必须逐项对应。
- `obs_frame_idx` 必须对应三个 observation 路径的编号。
- action 第 `i` 项对应时间 `i → i+1` 的状态转移。

### 12.6 mask 一致性

- `condition_mask[m] + noise_mask[m] == 1`，这里是完整模态层级；video 的首帧例外由帧级 mask 进一步描述。
- `noise_mask == loss_mask`，适用于当前四种任务设计。
- `len(video_noise_frame_mask) == 17`。
- `len(video_loss_frame_mask) == 17`。
- T0 的 video 帧级 mask 必须为首帧 0、后 16 帧 1。
- T1/T2 的 video 帧级 mask 必须全为 0。
- T3 的 video 帧级 mask 必须全部为 1，且 `observation_rgb_is_condition == false`。
- 四种任务都必须满足 `action_is_condition == true`，且显式 `condition_modalities` 与任务表一致。
- `available_modalities[m] == 1` 才允许该模态作为条件或监督目标。
- mask 必须严格匹配第 5 节任务表。

## 13. 统计文件要求

`mixture_statistics.json` 至少包含：

```json
{
  "schema_version": "modality_forcing_v1",
  "random_seed": 42,
  "input_file": "...",
  "output_file": "...",
  "total_samples": 1000,
  "task_counts": {
    "T0": 400,
    "T1": 200,
    "T2": 200,
    "T3": 200
  },
  "task_ratios": {
    "T0": 0.4,
    "T1": 0.2,
    "T2": 0.2,
    "T3": 0.2
  },
  "episode_count": 0,
  "missing_file_count": 0,
  "invalid_shape_count": 0,
  "nan_or_inf_count": 0,
  "train_test_episode_overlap": 0
}
```

还应输出按 setting/task 的交叉统计，便于确认分层后没有明显偏斜。

`reports/validation_report.json` 用于汇总整条数据链路，而不只统计 mixture：

```json
{
  "status": "pass",
  "vae_index": {
    "clips": 0,
    "episodes": 0,
    "invalid_clips": 0
  },
  "split": {
    "train_clips": 0,
    "test_clips": 0,
    "episode_overlap": 0,
    "train_active_frac": 0.0,
    "test_active_frac": 0.0
  },
  "pipeline": {
    "train_samples": 0,
    "test_samples": 0,
    "missing_from_pipeline": 0,
    "misaligned_samples": 0
  },
  "mixture": {
    "train_samples": 0,
    "task_counts": {"T0": 0, "T1": 0, "T2": 0, "T3": 0},
    "invalid_masks": 0
  }
}
```

`reports/rejected_samples.json` 应为每个被丢弃项记录 `stage`、`episode`、`obs_frame_idx`（若已有）、`reason_code` 和可读说明。只要有未被允许的严重错误，顶层 `status` 必须为 `fail`，检查脚本返回非零状态。

## 14. 配置文件建议

`configs/omnivitac_data_build_v1.json` 推荐统一管理三个上游阶段和 mixture：

```json
{
  "schema_version": "modality_forcing_v1",
  "dataset_source": "omnivitac",
  "source_root": "/path/to/processed_dataset",
  "output_dir": "./outputs",
  "vae_index": {
    "camera": "camera2",
    "frame_stride": 3,
    "n_frames": 17,
    "clips_per_episode": 25,
    "trans_thresh_mm": 1.0,
    "rot_thresh_rad": 0.01,
    "with_stats_before_split": false,
    "seed": 0
  },
  "split": {
    "activity_modality": "contact",
    "train_weight": 1000,
    "test_weight": 200,
    "active_frac": 0.85,
    "seed": 0
  },
  "physical_statistics": {
    "source": "train_clip_index_only",
    "n_sample_frames": 3000,
    "seed": 0
  },
  "pipeline": {
    "camera": "camera2",
    "action_stride": 3,
    "train_statistics_only": true,
    "require_explicit_physical_paths": true
  },
  "mixture": {
    "random_seed": 42,
    "task_ratios": {
      "T0": 0.4,
      "T1": 0.2,
      "T2": 0.2,
      "T3": 0.2
    }
  },
  "expected_future_frames": 16,
  "expected_action_dim": 7,
  "check_file_exists": true,
  "check_array_content": true
}
```

这里 `1000:200` 和 `active_frac=0.85` 是当前本地已有划分的配置记录，不是本次新确定的不可修改研究结论；训练任务比例 `40/20/20/20` 才是本次明确固定的要求。

真实绝对路径只写入本地配置，不写死在代码中。若配置需要提交到仓库，应另做一个不含个人路径的 example 文件。所有输出路径都从 `output_dir` 派生，避免命令之间手工传错文件。

## 15. 配套脚本职责

### 15.1 `build_vae_index.py`

负责从已处理 episode 生成全量 VAE clip index；已经实现 RGB/contact/force 存在性、数组形状、NaN/Inf、严格相机选择和 rejection report 检查。物理统计由 split 后的 `compute_physical_statistics.py` 单独负责。

### 15.2 `split_train_test.py`

负责在 setting 内按 episode 划分 train/test，并根据选定物理模态保留全部 active clip、抽样 inactive clip。它只操作 clip index，不生成 RGB/action，不分配 T0–T3。

### 15.3 `build_pipeline_data.py`

负责分别把 train/test clip index 转换为 pipeline data。实现已经将 `contact_paths` 和 `force_paths` 显式复制并规范化为相对 `source_root` 的 pipeline 路径；训练侧生成 action statistics，测试侧不重新计算。

### 15.4 `build_modality_mixture.py`

只负责：

1. 读取配置和基础 JSON。
2. 统一解析顶层是裸列表还是 `{samples: [...]}` 的输入。
3. 为基础 clip 生成稳定 ID。
4. 分层、打乱并精确分配训练任务。
5. 生成四个任务测试视图。
6. 写入新 schema 和统计信息。

不负责：

- 图像预处理。
- SAM 分割。
- contact/force 生成。
- 张量归一化。
- 模型训练。

### 15.5 `check_data_pipeline.py`

负责对三个上游阶段和最终 mixture 做端到端检查并报告：

- VAE clip index 的帧数、路径和物理数组是否正确。
- split 是否存在 episode 泄漏，active 比例和 setting 覆盖是否合理。
- pipeline 是否与 split clip 一一对应，RGB/contact/force/action 是否严格对齐。
- schema/key 是否正确。
- 数量和比例是否正确。
- 所有路径是否存在。
- 序列长度和数据形状是否正确。
- mask 是否与任务匹配。
- train/test 是否泄漏。
- 四个测试文件是否具有完全相同的基础样本。

检查失败时脚本必须返回非零退出状态，不能只打印 warning 后继续。

## 16. 与 Dataset/训练代码的接口要求

数据 JSON 生成完成不等于功能完成。后续还要修改模型侧代码，使这些字段真正生效。

### 16.1 Dataset 侧

`ACWMDataset` 后续需要：

1. 读取 `task_type`。
2. 读取显式的 contact/force 路径。
3. 返回 `condition_mask`、`noise_mask`、`loss_mask` 和 video 帧级 mask。
4. 返回三种完整序列；`observation_frame` 在 T3 中是 video ground truth 的第 0 帧，不是输入条件。
5. 保持 action 为 `16 × 7`。
6. batch collate 后 mask 的形状应为 `[B, 3]` 或三个命名 tensor；优先避免依赖隐含顺序。

### 16.2 训练 loss 侧

当前代码会给 video/contact/force 全部加噪，并使用同一个 timestep。modality forcing 后需要：

1. 分别生成 `t_video`、`t_contact`、`t_force`。
2. 条件模态位于 scheduler 的 clean endpoint，不添加随机噪声。
3. 目标模态采样训练 timestep 并加噪。
4. 只对 `loss_mask=1` 的模态计算 loss。
5. video 第 0 帧是否加噪、是否计算 loss 必须由帧级 mask 决定：T0 不生成第 0 帧，T3 必须生成第 0 帧。
6. 三个输出分支仍然全部存在；“条件模态 loss 为 0”不等于删除输出头。

当前 flow matching 若采用：

```text
x_t = (1 - t) × clean + t × noise
```

则 `t=0` 是 clean endpoint。实现时仍应以项目 scheduler 的实际公式为准，不能只靠名称猜测。

### 16.3 模型侧

现有 DiT 接口目前偏向单一共享 timestep。后续需确认：

- forward 是否接受三个 timestep。
- 每个模态的 time embedding 是否独立进入对应分支。
- 条件模态和目标模态拼接后是否仍可进行跨模态 attention。
- 推理代码是否能按 T0/T1/T2/T3 分别准备 clean/noisy latent。
- 现有“强制用 observation 覆盖 video 第一个 latent”的逻辑必须改成按任务控制；T3 不允许执行该覆盖。

这些是模型代码任务，不属于本阶段数据索引脚本，但数据 schema 必须提前支持。

## 17. 归一化边界

mixture 构建脚本只组合路径和任务标签，不在 JSON 中写入已归一化的大数组。

现有归一化逻辑仍由 Dataset/训练代码执行：

- action：需要确认第 7 维 gripper 是否应与前 6 维统一 z-score；现有代码与数据处理注释存在口径不一致，接入新训练前必须确认。
- contact：保持零背景语义，不能随意减均值破坏“无接触=0”。
- force：保持零背景语义，并沿用现有有效区域缩放/截断规则。
- video：沿用 Wan VAE 现有图像预处理规则。

## 18. DROID 的后续位置

DROID 当前明确不接入本版 mixture，原因是其 action、相机、机器人本体、频率及是否具备 contact/force 监督，都需要在读完论文和检查原始 schema 后再确定。

后续接入时应新建独立的数据源适配器，先统一到本文件定义的 canonical sample schema，再参与训练 mixture。需要单独确认：

- RGB 相机数量、视角和时间戳。
- action 维度、坐标系、单位、绝对/相对形式。
- 机器人状态和 gripper 语义。
- episode 边界和采样频率。
- 是否存在可直接使用的 contact/force ground truth。
- 若缺少物理模态，哪些任务仍合法，loss mask 应如何设置。

禁止把 DROID 缺失的 contact/force 直接填成全零并当作真实监督，因为全零在当前项目中本身代表“没有接触”，会造成错误标签。

## 19. 待向学长确认的接口问题

以下问题不妨碍先实现通用脚本，但在真正修改训练代码前应确认：

1. 条件模态是否严格设置为 clean、无噪声、loss 为 0？本文暂按“是”设计。
2. 三个模态是否要求三个完全独立采样的 timestep，还是只要求可独立设置为 clean/noisy？
3. 学长是否已有固定 key 命名；若有，应在写代码前将本 schema 映射到指定名称。
4. 训练 mixture 是离线固定分配还是 DataLoader 在线随机采样？本文推荐离线固定分配，便于复现和审计。
5. contact/force 的 observation 时刻是否也应作为 T0 预测目标？本文按现有训练逻辑，将它们视为目标序列的一部分。

## 20. 验收标准

数据输入阶段完成需同时满足：

- [ ] `clips.json` 已从审核完成的 episode 生成，且每条 clip 有 17 个 frame/contact/force 条目。
- [ ] VAE physical statistics 只从 `clips_train.json` 计算，并按文档规则使用非零物理像素。
- [ ] train/test 已按 episode 划分，episode overlap 为 0。
- [ ] 所有 active clip 被保留，inactive 抽样数量和实际 active 比例有统计记录。
- [ ] train/test pipeline 已分别从对应 clip index 生成，没有重新抽帧。
- [ ] pipeline 每条样本有 1+16 RGB、1+16 contact、1+16 force 和 16×7 action。
- [ ] action statistics 只由 train 数据生成，test/validation 复用训练统计。
- [ ] VAE clip、split clip、pipeline sample 可通过 `base_sample_id` 逐级追踪。
- [ ] `train_mixture.json` 总样本数与基础 train clip 数一致。
- [ ] 训练实际比例为 T0 40%、T1 20%、T2 20%、T3 20%（只允许不可避免的整数舍入）。
- [ ] 每个测试基础 clip 都有 T0/T1/T2/T3 四个版本。
- [ ] 所有样本的 RGB/contact/force/action 长度正确。
- [ ] 所有显式路径可解析且文件存在。
- [ ] 所有 task mask 与任务定义完全一致。
- [ ] T3 不把 observation RGB 输入模型，并对完整 17 帧 video 加噪、计算 loss。
- [ ] train/test episode 零重叠。
- [ ] 相同 seed 可重复生成完全相同的文件。
- [ ] 统计报告无 missing、NaN、Inf 或非法形状。
- [ ] `ACWMDataset` 能读取新字段并组成 batch。
- [ ] 训练 loss 实际使用 noise/loss mask，而不是只把它们写在 JSON 中。
- [ ] 用极小数据跑通一次完整前向与反向传播。

## 21. 建议执行顺序

```text
第 1 步：确认 source_root 中 episode 已完成 RGB/SAM/contact/force 预处理
    ↓
第 2 步：运行 build_vae_data.py，生成全量 clips.json 和 physical statistics
    ↓
第 3 步：检查 17 帧路径、形状、NaN/Inf 和跳过记录
    ↓
第 4 步：运行 split_train_test_v2.py，生成 episode 级 train/test clip index
    ↓
第 5 步：检查 episode 零泄漏、setting 覆盖和 active/inactive 比例
    ↓
第 6 步：运行 compute_physical_statistics.py，只用 clips_train.json 计算 physical statistics
    ↓
第 7 步：运行 build_pipeline_data.py，生成显式 contact/force 路径
    ↓
第 8 步：分别生成 train/test pipeline；action statistics 只用 train
    ↓
第 9 步：运行 build_modality_mixture.py
    ↓
第 10 步：生成 40/20/20/20 train mixture 和四个同源 test 视图
    ↓
第 11 步：运行端到端 check_data_pipeline.py
    ↓
第 12 步：把新 schema 接入 ACWMDataset
    ↓
第 13 步：把独立 timestep、noise mask、loss mask 接入训练 loss/DiT
    ↓
第 14 步：用极小数据跑通完整前向和反向传播
    ↓
第 15 步：正式训练与四任务评估
    ↓
未来阶段：单独研究并接入 DROID
```

## 22. 本版固定决策摘要

- 任务比例固定为 `T0:T1:T2:T3 = 40:20:20:20`。
- 当前只使用现有 OmniViTac/本地处理数据。
- DROID 暂不参与数据混合。
- 不重新运行 SAM，不重复生成媒体数据。
- 完整数据顺序固定为 VAE clip index → episode 级 split → train/test pipeline → modality mixture。
- contact/force 和 action 的归一化统计都只能由 train split 生成，test 不参与统计。
- VAE clip 固定为 17 帧、默认 stride 3；当前本地正式索引每 episode 最多 25 条。
- pipeline 必须显式携带 RGB、contact、force 和 action，不能依赖文件名猜测物理路径。
- 训练集一条基础 clip 只分配一个任务。
- 测试集每条基础 clip 展开为四种任务。
- 使用显式物理模态路径。
- 条件模态 clean 且 loss 为 0；目标模态加噪且计算 loss。
- RGB observation 不是公共条件：T0 使用首帧；T1/T2 使用包含首帧的 full video；T3 不输入任何 RGB observation。
- 所有随机分配固定 seed，默认 `42`。
