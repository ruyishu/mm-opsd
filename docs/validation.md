# 验证记录

已完成本地 CPU 代码测试、真实数据准备和官方 processor 接口验证。
**尚未在 L20 上运行完整 2B 模型训练，未测量 GPU 显存、吞吐或准确率提升。**

## 本地测试环境

Windows 11、Python **3.12.4**，项目独立虚拟环境。
PyTorch **2.8.0+cpu**、torchvision **0.23.0+cpu**，其余核心库与仓库固定版本一致，
包括 transformers **4.57.1**、PEFT **0.17.1**、datasets **3.6.0** 和 math-verify **0.8.0**。
目标 L20 环境使用 Python **3.10.16** 和 PyTorch **2.8.0+cu128**，见 `environment.yml`。
两者区别不应被理解为已经验证 Linux/CUDA 安装或 GPU 执行。

| 检查 | 结果 |
|---|---|
| `python -m pytest` | **22 passed** |
| `python -m ruff check .`，ruff 0.13.1 | All checks passed |
| `python -m pip check` | No broken requirements found |
| `python -m mm_opsd.doctor --allow-cpu` | 所有固定库版本匹配，errors 为空 |
| `bash -n scripts/smoke_l20.sh scripts/run_l20.sh` | 通过 |

测试使用随机初始化的小型真实 Qwen3-VL、实际 Qwen3VLProcessor 与 PEFT，包含：

- 带图片的生成、前向、反向及语言侧 LoRA 更新。
- 训练更新后，关闭 LoRA 的 Teacher 输出逐位保持相同，视觉参数没有梯度。
- Student 提示中不含答案，Teacher 与 Student 的图像、生成序列及 M-RoPE 位置正确对齐。
- 分块完整词表 KL 与未分块计算的损失、梯度一致，逐词表项裁剪行为正确。
- OPSD、答案 SFT、去答案负对照的训练，以及优化器和 RNG 恢复。
- 实际 CLI 入口完成本地模型加载 → 训练 → validation → 保存/加载 adapter → Base/训练模型评估 → 配对报告。
- 数据重题筛查、分 split 下载、数学等价、数值精度、重复正确选项及评测 IDs 一致性。

随机模型的恢复测试会产生一条 PEFT 提示：没有预训练基座配置路径；该测试通过。
Windows 下 math-verify 0.8.0 的内置超时使用不可序列化的嵌套进程函数，
因此 Windows CPU 调试关闭该库的超时；目标 Linux 主线程判分保留 2 秒解析/比较超时。

## 真实数据

从固定 Hugging Face commit 下载实际 parquet，完成全部图片解码、保存和训练侧重题筛查：

| split | 实际数量 |
|---|---|
| Geometry3K train，筛查前 | 2,101 |
| Geometry3K train，筛查后 | **1,472** |
| Geometry3K validation | 300 |
| Geometry3K test | 601 |
| MathVista testmini | 1,000 |

移除 629 条训练题，其中 625 条为完全相同 RGB 图片，4 条为 dHash 近似图片。
默认 dHash 距离阈值为 2，图像宽高比例相对差异最多 10%。
该结果是保守筛查记录，不证明所有语义重题均已排除。

对全部保留题检查了唯一 IDs、图片存在，以及将标准答案包进 `\boxed{}` 后能被本地规则判为正确。
真实 MathVista pid 781 有两个同为 18 的正确选项；判分已按选项内容处理，并加入回归测试。
这项标准答案回读检查不代表模型能正确回答或实际输出都能被解析。

可选 MathVision testmini 的 **304 条**也已实际下载转换，其中 190 条有选项，
全部通过标准答案回读。可选数据与默认数据的联合训练过滤尚未作为一次完整 prepare 运行；
启用时必须从新的数据输出目录准备并重新训练。

| 数据/模型 | 固定 revision |
|---|---|
| hiyouga/geometry3k | `fd21e533e1e50d0662a2bf7b223e60511bd5f8b7` |
| AI4Math/MathVista | `2b6ad69445fbb5695c9b165475e8decdbeb97747` |
| MathLLMs/MathVision | `2837ddb3f13abaf6b3997c12d80753e5470bd46a` |
| Qwen/Qwen3-VL-2B-Instruct | `89644892e4d85e24eaac8bacfd4f463576704203` |

## 官方 processor

加载上述 Qwen3-VL-2B-Instruct 的官方 processor/tokenizer，不下载 2B 权重。
确认实际使用 Qwen3VLProcessor + Qwen2VLImageProcessor，`use_fast=False` 和
`min_pixels=4096`、`max_pixels=262144` 生效。
一张 1024×768 图片处理后的 `image_grid_thw=[1,26,36]`，连同简短问题为 247 个输入 token。

每个默认 split 抽查前 4 题及题目文字最长的 4 题，共 32 次编码，
训练侧使用包含答案的 Teacher 提示；这些样本均能保留完整配置生成预算。
抽查得到最大 prompt 长度：训练 275、validation 280、Geometry3K test 248、MathVista 483 token。
这不是对所有模型实际生成长度的预测。

## L20 上的下一步

先执行 README 中的安装命令，再运行 `bash scripts/smoke_l20.sh`。
该脚本对真实 2B 权重进行 8 条训练题、2 次参数更新及小规模评估，
记录 GPU 峰值显存和生成截断率。通过后运行 `bash scripts/run_l20.sh`。
实际实验结果应以运行生成的 metrics、逐题 predictions 和配对报告为准。
