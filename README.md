# mm-opsd

面向单张 **NVIDIA L20 48GB** 的视觉数学 on-policy self-distillation 最小实验仓库。

默认方案：**Qwen3-VL-2B-Instruct + Geometry3K 最终答案训练 + Geometry3K / MathVista 评估**。
Student 只看图文题目；固定 Teacher 额外看标准答案，并对 Student 自己生成的序列提供完整词表前向 KL。
答案条件化无需预先生成 CoT。算法、与原论文的差别及评测口径见 [方法说明](docs/method.md)。

这是可运行的独立实现；尚未实测 L20 显存、吞吐或 benchmark 提升。
本地离线测试结果和环境见 [验证记录](docs/validation.md)。

## 环境

推荐 Ubuntu **22.04.5 LTS**、Python **3.10.16**，Linux NVIDIA 驱动 **570.124.06 或更新兼容版本**。
Conda 创建 Python 环境，pip 安装官方 PyTorch cu128 构建；使用 SDPA 时无需单独安装 CUDA Toolkit。

| 依赖 | 固定版本 |
|---|---|
| torch / torchvision | 2.8.0+cu128 / 0.23.0+cu128 |
| transformers / trl | 4.57.1 / 0.26.0 |
| accelerate / peft | 1.11.0 / 0.17.1 |
| datasets / huggingface-hub | 3.6.0 / 0.36.0 |
| qwen-vl-utils / Pillow | 0.0.14 / 11.3.0 |
| math-verify / safetensors | 0.8.0 / 0.5.3 |
| PyYAML / numpy | 6.0.2 / 1.26.4 |
| wandb | 0.22.3 |

TRL、Accelerate、qwen-vl-utils 和 wandb 保留为方案中的固定依赖；训练循环直接使用 PyTorch / PEFT，
无须登录 wandb。首版不使用 vLLM、DeepSpeed 或 FlashAttention 扩展。

```bash
git clone https://github.com/ruyishu/mm-opsd.git
cd mm-opsd
conda env create -f environment.yml
conda activate mm-opsd
python -m pip check
python -m mm_opsd.doctor
```

`doctor` 核对版本，并在 GPU 上执行 BF16 前向与反向；失败时先处理打印的驱动或依赖问题。
从仓库根目录运行以下命令。不要用 accelerate/torchrun 启动多进程，当前实现目标是单 GPU。

## 先跑短验证

```bash
bash scripts/smoke_l20.sh
```

脚本准备数据，使用 8 条训练题跑 2 次参数更新，再评估 8 条测试题。
它仍下载真实模型和完整数据用于重题排查；首次下载耗时不属于训练耗时。
`runs/smoke/metrics.jsonl` 记录 loss、生成截断率、耗时与 GPU 峰值显存。
短验证只检查运行链路，不用于判断准确率提升。

## 完整实验

```bash
bash scripts/run_l20.sh
```

脚本完成数据准备、训练、Base / OPSD 两侧评估与配对报告。
默认完整 Geometry3K test 601 题、MathVista testmini 1,000 题，均每题贪心生成一次。
已有非空输出目录会拒绝覆盖；断点恢复或新实验请使用下面的单独命令。

也可以逐步运行：

```bash
python -m mm_opsd.prepare --output data/processed
python -m mm_opsd.train --config configs/l20.yaml
python -m mm_opsd.evaluate --config configs/l20.yaml --benchmark geometry3k \
  --output runs/eval/base-geometry3k
python -m mm_opsd.evaluate --config configs/l20.yaml --benchmark geometry3k \
  --adapter runs/mm-opsd --checkpoint best --output runs/eval/opsd-geometry3k
python -m mm_opsd.report --baseline runs/eval/base-geometry3k \
  --trained runs/eval/opsd-geometry3k --output runs/reports/geometry3k.json
```

`--adapter` 接受具体 checkpoint，也接受包含 best/latest 指针的训练目录；
`--checkpoint best` 按 validation 准确率选择，同分保留较早 checkpoint。

## 数据与重题

Geometry3K 原始 train/validation/test 分别为 2,101/300/601 题。
准备脚本将 train 中与 validation、test、MathVista testmini 共享或高度近似图片的题移除，
保留完整评测集。实际训练数量以 `data/processed/manifest.json` 为准。
默认固定版本已经完整准备验证：移除 629 条，保留 **1,472 条训练题**；
其中 625 条因完全相同图片移除，4 条因近似图片移除。

默认模型和数据固定到已核对的 Hugging Face commit SHA；准备脚本只下载需要的 split，
不会为了 testmini 下载无公开答案的完整 test。数据规模和过滤设置写入 manifest；每个 checkpoint
还保存训练数据文件 SHA256。图片优先过滤是保守筛查，无法保证发现裁剪、重画或所有语义重题，
近似图片也可能造成误删。被移除题的 IDs、图像路径和题目保存在 `removed_overlaps.jsonl` 便于检查。

默认每图最大像素面积 `262144`（约 512×512，保持比例），最小面积 `4096`。
先抽查图上的小字与角度是否可读；修改后须为 Base 和 OPSD 使用相同图像预算。
没有自动截断图像占位 token 或问题，超出上下文会显示具体题 ID 并停止。

## 训练参数和恢复

`configs/l20.yaml` 默认 BF16、语言侧 LoRA r=16/alpha=32、batch=1、梯度累积=16、
学习率 5e-6、梯度裁剪 0.1、1 epoch、生成最多 512 token、完整序列最多 2,048 token。
每 25 次参数更新保存 checkpoint，使用 validation 固定前 64 题选择 best。
训练前向与 Teacher 前向均独立处理图像位置；固定 Teacher 不生成另一份 CoT。
完整词表损失每块 32 个位置，保留精确前向 KL 和上游逐词表项裁剪。

```bash
python -m mm_opsd.train --config configs/l20.yaml --resume-from runs/mm-opsd
```

checkpoint 保存 LoRA、processor、优化器、学习率预算、样本游标与 Python/Torch/CUDA RNG。
从最新完整参数更新恢复。允许更改 epochs/max_steps/output_dir，其他训练设置和数据不允许悄悄变化。
扩展训练会继续原学习率 warmup 计划。`training_state.pt` 仅用于恢复本仓库自己产生的 checkpoint。

```bash
# 相同训练数据上的答案 SFT 对照
python -m mm_opsd.train --config configs/l20.yaml --method sft --output-dir runs/answer-sft
# 去除 Teacher 标准答案的负对照
python -m mm_opsd.train --config configs/l20.yaml --method no_answer --output-dir runs/no-answer
```

首轮先比较 Base 与 OPSD。判断训练方法收益时再补 SFT，正式研究还需多个训练随机种子。
生成预算达到上限但没有 EOS 会标记为 truncated；已包含完整最终答案的输出仍正常判分。
验证集显示大量截断时调整生成长度，再同时重跑两侧评估。

## 可选 MathVision

需要从准备阶段纳入 MathVision 的 304 道 testmini，才能让训练侧同时排查其重题：

```bash
python -m mm_opsd.prepare --output data/with-mathvision --include-mathvision
```

把新实验配置 `data.root` 改为 `data/with-mathvision`，训练到新的 output_dir。
然后 `--benchmark mathvision` 评估。不要训练完才新增重题筛查数据并继续使用旧 checkpoint。

## 下载和结果

数据与模型从 Hugging Face 下载，需要可访问对应站点。下载受限时，可以先配置可用的
`HF_ENDPOINT` / `HF_HOME`，或将 `model.name` 改为包含全部模型、tokenizer、processor 文件的本地目录。
默认模型 revision 已固定；改为 `null` 时首次加载会解析并记录 commit SHA。
恢复与 adapter 评估使用 checkpoint 记录的 revision。

输出包括：

- `runs/mm-opsd/checkpoint-*`：LoRA、processor 和恢复状态。
- `runs/mm-opsd/metrics.jsonl`：训练指标、峰值显存和恢复事件。
- `runs/eval/*/predictions.jsonl`：逐题回答、解析结果、对错和截断状态。
- `runs/eval/*/summary.json`：准确率及评测配置。
- `runs/reports/*.json`：配对准确率变化、题目级 bootstrap 区间和 McNemar 检验。

本地规则判分无需 GPT API，不能直接等同官方排行榜分数；Geometry3K 为自由作答口径。
图片、权重、checkpoint 和实验结果均被 `.gitignore` 排除。

## 离线测试

```bash
python -m pip install -e '.[test]'
python -m pytest
```

测试使用随机初始化的小型真实 Qwen3-VL，不下载预训练权重；检查多模态前向/反向、
固定 Teacher、位置对齐、词表损失及梯度等价性、数据筛查、答案判分和恢复状态。

代码采用 MIT，方法来源与数据/模型许可证见 [方法说明](docs/method.md)。
