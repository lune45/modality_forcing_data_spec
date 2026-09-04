# Modality Forcing 数据构建工具

本目录实现了技术规格中的完整数据输入流程：

```text
已处理 episode
→ VAE 17 帧 clip index
→ episode 级 train/test split
→ train-only contact/force statistics
→ train/test multimodal pipeline data
→ T0/T1/T2/T3 mixture
→ 端到端验证
```

当前只面向现有 OmniViTac 数据；DROID 暂不接入。工具只生成索引、统计和报告，不复制原始 RGB 或 `.npy` 文件，也不重新运行 SAM。

## 文件说明

- `build_vae_index.py`：生成经过路径和数组检查的 17 帧 clip index。
- `split_train_test.py`：按 setting、episode 划分，并控制 active/inactive 比例。
- `compute_physical_statistics.py`：只从 train clip 计算 Dataset 可直接读取的统计。
- `build_pipeline_data.py`：添加 RGB、显式 contact/force 路径和 16×7 action。
- `build_modality_mixture.py`：生成 40/20/20/20 train mixture 和四套对齐测试数据。
- `check_data_pipeline.py`：验证整条链路、路径、形状、泄漏、比例和 mask。
- `run_data_pipeline.py`：按照配置顺序调用以上工具。
- `MODALITY_FORCING_DATA_SPEC.md`：完整设计、字段和模型侧接口说明。
- `README.md`：从项目宏观目标到核心代码行号的全链路导读。

## 使用方法

1. 安装依赖：

```bash
python -m pip install -r modality_forcing_data_spec/requirements.txt
```

2. 复制示例配置并填写真实的 `source_root`：

```bash
cp modality_forcing_data_spec/configs/omnivitac_data_build_v1.example.json \
   modality_forcing_data_spec/configs/omnivitac_data_build_v1.local.json
```

3. 先检查将执行的命令：

```bash
python modality_forcing_data_spec/run_data_pipeline.py \
  --config modality_forcing_data_spec/configs/omnivitac_data_build_v1.local.json \
  --dry_run
```

4. 正式运行：

```bash
python modality_forcing_data_spec/run_data_pipeline.py \
  --config modality_forcing_data_spec/configs/omnivitac_data_build_v1.local.json
```

成功标准是最后显示 `complete data workflow passed`，并且：

```text
outputs/reports/validation_report.json → status = pass
```

## 重要约束

- `source_root` 必须指向已经完成 RGB、SAM、contact、force 预处理的数据根目录。
- train/test 必须按 episode 划分。
- contact/force 和 action statistics 只能由 train split 生成。
- T3 只有 `action + contact` 条件，不输入 RGB observation，并生成完整 17 帧 video。
- `check_array_content=true` 会完整读取物理数组，速度较慢，但正式产数时不应关闭。
- 旧 pipeline 依靠 RGB 文件名推导 contact/force；本实现始终写入显式路径。

## 测试

```bash
python -m unittest discover -s modality_forcing_data_spec/tests -v
```
