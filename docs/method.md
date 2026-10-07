# 方法与原 OPSD 的对应关系

参考 [OPSD](https://github.com/siyan-zhao/OPSD)，核对的上游提交为
`ae7d2519e94920c4eb6206c0c26de46d9c50abae`。
本仓库独立实现多模态数据、训练与评估流程，没有复制整个上游训练器。

Student 的条件为图片 `I` 与问题 `q`；Teacher 的条件额外包含正确最终答案 `a`。
Student 每题采样一条序列 `y`。两侧都在同一条 `y` 的前缀上计算下一 token 分布：

$$p_S(\cdot\mid I,q,y_{<t}),\quad p_T(\cdot\mid I,q,a,y_{<t}).$$

Teacher 是初始基座模型，Student 是基座加语言侧 LoRA。Teacher 前向关闭 LoRA，
使用 `no_grad`，所有基座参数保持冻结。只共享权重，不共享文本位置或 KV cache。
Teacher 每次前向结束前恢复 LoRA，反向重计算发生在 Student 上。

## 损失

沿用上游 `beta=0` 的前向 KL，在完整词表上计算。
裁剪作用于每个位置、每个词表项的 KL 贡献，而非把整个 token 的 KL 总和裁剪到 0.05。
这类逐项上界裁剪后，标量损失不一定非负，不能把它等同于未经裁剪的标准 KL。

词表损失按生成位置分块计算，并对块使用非重入 gradient checkpointing。
仍然计算每个块的全部词表概率，未使用 top-k 重新归一化或 sampled-token 近似。
默认 temperature=1.1 同时用于采样和蒸馏，保持上游脚本的温度语义。
计算 log-softmax 时转为 FP32，以减少 BF16 下的数值误差。

两侧独立计算完整图文序列的 M-RoPE position IDs。
只取 `[prompt_length-1 : prompt_length-1+completion_length]` 的隐藏状态；
第一个生成 token 对应最后一个 prompt 位置的预测，最后的 EOS 也参与训练。
单样本模式无需 padding。提示词、图片占位 token 和标准答案提示均不作为监督目标。

## 与论文主实验的差别

- 训练数据只有最终答案，没有参考 CoT；这是论文方法允许的条件，但原论文主实验使用 OpenThoughts 解答。
- Qwen3-VL-Instruct 没有文本 Qwen3 的 thinking-on/off 模板开关；两侧使用同一 Instruct 权重与模板。
- 语言侧 LoRA rank 从 64 缩小为 16，视觉编码器、连接模块和词表输出层冻结。
- 初始生成预算 512 token，单卡有效 batch 16；原文收敛速度不能直接套用。
- `no_answer` 负对照让 Teacher 不看答案；LoRA 零初始化时它接近无更新的基座正则化，不是强教师蒸馏。
- `sft` 对照仅在同一训练集上学习最终答案，不能称为 CoT-SFT。

知道正确答案不等于能提供正确推理。答案条件化 Teacher 的分布没有过程正确性保证。
主要验证目标是未见过答案的 Student 在保留的测试题上是否改善。

## 评估口径

Geometry3K 转换版本不保留原四选一选项，所以本仓库采用自由作答；不要与原题四选一准确率直接比较。
MathVista / MathVision 使用选项字母或最终数值。本地规则判分不调用外部 LLM，
与官方使用模型提取答案的完整评估流程不同，输出明确标记 `official_leaderboard_score=false`。
选择题按所选选项内容与答案比较；MathVista 存在重复选项（如 pid 781 的 C、D 都是 18），
重复的正确内容均接受，不会把这类数据判为答案解析失败。
报告解析失败率、生成截断率与配对准确率变化；置信区间描述测试题抽样不确定性，
不代表多个训练随机种子的稳定性。

checkpoint 仅使用 Geometry3K validation 的固定前 64 题选择，绝不使用外部 test 分数选择。
正式评估必须对 Base 和训练模型使用相同图像预算、问题 IDs、解码和生成长度。

## 来源与许可证

- [OPSD 论文](https://arxiv.org/abs/2601.18734)与[原仓库](https://github.com/siyan-zhao/OPSD)：方法与参数来源。
- [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)：模型为 Apache-2.0。
- [Geometry3K 转换数据](https://huggingface.co/datasets/hiyouga/geometry3k)：数据卡标注 MIT，来源为 InterGPS。
- [MathVista](https://huggingface.co/datasets/AI4Math/MathVista)：数据卡标注 CC-BY-SA-4.0。
- [MathVision](https://huggingface.co/datasets/MathLLMs/MathVision)：数据卡标注 MIT。

本仓库代码采用 MIT。模型权重和数据不会上传到 GitHub，各自许可证仍适用。
