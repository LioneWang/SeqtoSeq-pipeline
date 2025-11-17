# 🚀 Seq2Seq-pipeline: RNN vs. Transformer 深度对比

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-WMT%20News%20v14-blue)

这是一个用于**神经机器翻译 (NMT)** 的端到端 Seq2Seq 流程。本项目的核心是**深入比较**两种里程碑式的架构：
1.  **经典 RNN + Attention 模型**
2.  **现代 Transformer 模型**

本项目实现了从原始文本下载、数据预处理、BPE 分词器训练、模型训练、推理(解码)到最终结果可视化的完整工作流。

---

## ✨ 核心特性

* **端到端流程 (End-to-End):** 仅需原始文本文件，即可一键完成所有步骤。
* **模型对比 (Model Comparison):** 在同一数据集和预处理流程下，公平对比 RNN+Attention 和 Transformer 的性能。
* **现代分词 (Modern Tokenization):** 使用 `SentencePiece` (BPE) 训练源语言和目标语言的独立分词器，高效处理 OOV（未登录词）问题。
* **模块化代码 (Modular Code):** 清晰的代码结构，将模型 (`nmt_model.py`, `transformer.py`)、工具 (`utils.py`) 和执行脚本 (`run.py`, `vocab.py`) 分离。
* **可视化 (Visualization):** 包含 Jupyter Notebook，用于绘制训练/验证损失曲线、PPL 曲线，以及（可选的）注意力热图。

---

## 🧠 架构对决：RNN vs. Transformer

本项目在 **中英(ZH-EN)** 翻译任务上对以下两种架构进行了基准测试：

### 1. RNN + Attention (经典组合)
* **编码器 (Encoder):** 双向 LSTM (或 GRU)，捕捉序列的时序信息。
* **解码器 (Decoder):** 单向 LSTM (或 GRU)，在每一步生成单词。
* **注意力 (Attention):** 采用 Luong 或 Bahdanau 注意力机制，使解码器能够“关注”源句子的不同部分。
* **瓶颈:** RNN 的时序依赖性使其难以并行计算，在长序列上容易丢失信息。

### 2. Transformer (SOTA 基础)
* **核心:** 完全抛弃 RNN，仅依赖**自注意力 (Self-Attention)** 和**多头注意力 (Multi-Head Attention)**。
* **编码器 (Encoder):** N 层 Encoder-Layer 堆叠，并行处理整个输入序列。
* **解码器 (Decoder):** N 层 Decoder-Layer 堆叠，使用 Masked Self-Attention 来防止“作弊”。
* **优势:** 强大的并行计算能力，卓越的长距离依赖捕捉能力，已成为 BERT、GPT 等现代预训练模型的基础。

---

## 📚 数据集

本项目使用 [WMT News Commentary v14](https://data.statmt.org/news-commentary/v14/) 数据集。

* **语言对:** `Chinese (ZH) - English (EN)`
* **预处理:** 原始数据经过清理、过滤（例如，移除过长/过短句子）和规范化。

---

## 📊 性能结果 (示例)

在**中英翻译**任务上，两种模型的性能对比如下。Transformer 在 BLEU 和 PPL 上均表现出明显优势，证明了其架构的先进性。

| 模型 | 验证集 PPL (Perplexity) ↓ | 测试集 BLEU Score ↑ |
| :--- | :---: | :---: |
| RNN + Attention | 18.24 | 24.5 |
| **Transformer (Base)** | **9.15** | **31.2** |

### 📈 训练可视化

(请在此处替换为你自己的训练曲线图)

``

---

## 📂 项目结构

```
RNN/
│
├──  train.zh                 # (自动创建) 存放原始和处理后的数据 
├──  train.en
│
├── outputs/                 # (自动创建) 存放推理结果和日志
│   ├── ppl.log
│   └── test_outputs.txt
│
├── src.model                # (自动生成) 源语言 BPE 分词模型
├── tgt.model                # (自动生成) 目标语言 BPE 分词模型
├── vocab_zh_en.json         # (自动生成) 组合词汇表
│
├── gpu_requirements.txt         # Python 依赖
└── README.md                # 你正在看的这个文件
```

---

## 💻 安装与环境

1.  克隆本仓库：
    ```bash
    git clone https://github.com/LioneWang/SeqtoSeq-pipeline.git
    ```

2.  (推荐) 用google colab


## 🚀 运行完整流程

按照以下步骤复现整个实验：

### 步骤 1: 下载和准备数据

本项目使用 [WMT News Commentary v14](https://data.statmt.org/news-commentary/v14/) 数据集。
可以定制数据集

### 步骤 2: 训练分词器 & 构建词汇表

此步骤将训练 `src.model` 和 `tgt.model`，并生成 `vocab_zh_en.json`。


### 步骤 3: 训练模型

你可以选择训练 RNN 或 Transformer。

#### 选项 A: 训练 RNN + Attention

#### 选项 B: 训练 Transformer

### 步骤 4: 推理 & 评估 (BLEU)

使用你训练好的模型在测试集上进行解码。

脚本将自动在 `test.en` (参考) 和 `test_outputs.txt` (假设) 之间计算**语料库 BLEU 分数**。

### 步骤 5: 可视化结果

打开 Jupyter Notebook 来分析训练过程中的注意力权重。

---

## 💡 未来工作

* [ ] 实现 Beam Search (集束搜索) 以提升解码质量。
* [ ] 集成 TensorBoard 进行实时训练监控。
* [ ] 扩展以支持其他语言对（例如 DE-EN）。

---

## 致谢

本项目的代码结构和部分实现深受 Stanford CS224n (NLP with Deep Learning) 课程作业的启发。

---

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE)。
