# Foundations of Generative AI (ICT4403) — Assignment 3
## Autoregressive Machine Translation Benchmark System (English $\rightarrow$ French)

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-Seq2Seq%20%26%20Causal-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A rigorous empirical and architectural study comparing **Encoder-Decoder** and **Decoder-Only** generative neural network architectures for machine translation on an English-to-French parallel corpus.

---

## 1. Problem Statement and Objective

Machine translation (MT) has evolved from rule-based and phrase-based statistical systems to deep autoregressive neural sequence transduction. Currently, the landscape is divided between two dominant architectural paradigms:
1. **Encoder–Decoder Architectures:** The original Transformer paradigm (Vaswani et al., 2017) and pretrained multilingual denoising models (e.g., mT5), which decouple bidirectional source sequence representation from autoregressive target generation via cross-attention.
2. **Decoder–Only Architectures:** Large Language Models (LLMs) such as LLaMA (Touvron et al., 2023) and Gemma (Team Gemma, 2024), which frame translation as autoregressive next-token prediction conditioned on causal in-context prompts.

### Core Objectives:
- Implement and train a **Basic Transformer (Encoder–Decoder)** from scratch in PyTorch.
- Fine-tune a pretrained multilingual model, **mT5 (`google/mt5-small`)**, on parallel text.
- Apply structured in-context prompting methods to modern open-weights decoder-only models: **LLaMA (`LLaMA-3.2-1B-Instruct`)** and **Gemma (`Gemma-2-2B-IT`)**.
- Evaluate all four systems under comparable test conditions using **BLEU**, **chrF**, **training/fine-tuning time**, **inference latency**, and **parameter count**.
- Conduct an empirical **Sampling-Temperature Analysis** ($T \in [0.0, 1.2]$) exploring translation fidelity vs. lexical diversity tradeoffs.
- Investigate **Responsible and Ethical AI** concerns including gender/occupational bias, hallucination, cultural/idiomatic preservation, and sensitive content translation.

---

## 2. Dataset Description and Source Link

The system utilizes the **European Parliament ComTrans Bilingual Parallel Corpus**, an official corpus released through the NLTK parallel corpus benchmark.

- **Source Language:** English (`en`)
- **Target Language:** French (`fr`)
- **Domain:** Parliamentary debates, legislative policy, economic governance, and public affairs.
- **Dataset Link:** [ComTrans Parallel Corpus on GitHub / NLTK Data](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/comtrans.zip)
- **HuggingFace Alternative Link:** [OPUS Books / Europarl](https://huggingface.co/datasets/opus_books)

### Preprocessing and Splitting:
- **Normalization:** Standardized unicode apostrophes, quotation marks, and whitespace.
- **Length Filtering:** Sentences restricted to $4 \le \text{word count} \le 30$ to eliminate sentence fragments and excessive sequence truncations.
- **Split Breakdown:**
  - **Training Set:** 4,000 sentence pairs (80%)
  - **Validation Set:** 500 sentence pairs (10%)
  - **Test Set:** 500 sentence pairs (10%), evaluated on a standardized 100-sentence test benchmark.

---

## 3. Brief Description of Models Implemented

| Model | Architecture | Parameter Count | Adaptation Method | Tokenizer |
| :--- | :--- | :--- | :--- | :--- |
| **Basic Transformer** | Encoder–Decoder | 4.73 M | Trained from Scratch (AdamW, CE Loss) | Word-Level with special tokens (`<sos>`, `<eos>`, `<pad>`, `<unk>`) |
| **mT5 Small** | Encoder–Decoder | 300.18 M | Fine-Tuned (Seq2Seq, span-denoised pretraining) | Multilingual SentencePiece (250,000 vocab) |
| **LLaMA 3.2-1B** | Decoder-Only | 1,235.81 M | Few-Shot In-Context Prompting | Byte-Pair Encoding (BPE, 128,000 vocab) |
| **Gemma 2-2B-IT** | Decoder-Only | 2,614.34 M | Few-Shot In-Context Prompting | SentencePiece BPE (256,000 vocab) |

### Architectural Highlights:
1. **Basic Transformer:** Built entirely from scratch in PyTorch. Implements sinusoidal positional encodings, scaled dot-product multi-head attention ($h=4, d_{model}=256, d_{ff}=512$), encoder self-attention, decoder causal self-attention, and decoder cross-attention with Pre-LayerNorm residual blocks.
2. **mT5 (`google/mt5-small`):** A 300M parameter model pretrained across 101 languages using a masked span-denoising objective. Fine-tuned with prompt prefix `translate English to French: {source}`.
3. **LLaMA (`LLaMA-3.2-1B-Instruct`):** An autoregressive causal language model featuring Rotary Position Embeddings (RoPE), SwiGLU non-linearities, and grouped-query attention (GQA).
4. **Gemma (`Gemma-2-2B-IT`):** Google's lightweight open model family featuring GeGLU activations, interleaved local sliding window and global attention, and RMSNorm pre-normalization.

---

## 4. Execution Steps and Required Dependencies

### Installation
Clone the repository and install all required packages:
```bash
git clone https://github.com/your-username/fgai-assignment3-machine-translation.git
cd fgai-assignment3-machine-translation
pip install -r requirements.txt
```

### Complete Pipeline Execution
Run the modular pipeline step-by-step:

```bash
# 1. Download and preprocess the parallel corpus
python scripts/01_download_and_preprocess.py

# 2. Train the Basic Transformer from scratch
python scripts/02_train_basic_transformer.py

# 3. Fine-tune / Evaluate mT5
python scripts/03_finetune_mt5.py

# 4. Prompt and evaluate LLaMA and Gemma
python scripts/04_evaluate_llama_gemma.py

# 5. Synthesize benchmarks and generate plots
python scripts/05_benchmark_all_models.py

# 6. Execute Sampling-Temperature Analysis
python scripts/06_sampling_temperature_analysis.py

# 7. Conduct Responsible & Ethical AI Analysis
python scripts/07_ethical_bias_analysis.py

# 8. Generate Submission Document (.docx)
python scripts/08_generate_submission_doc.py
```

### Interactive Notebook
Alternatively, open and run the complete end-to-end notebook:
```bash
jupyter notebook machine_translation_benchmark.ipynb
```

---

## 5. Results Obtained

The models were evaluated on the **exact same 100-sentence test set** under identical greedy/low-temperature decoding settings:

| Model | Architecture | Parameters | Train / Adapt Time | Latency (ms/sent) | BLEU Score | chrF Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Basic Transformer** | Encoder–Decoder | 4,726,272 | 85.4 s | **12.8 ms** | 24.35 | 45.80 |
| **mT5 Small** | Encoder–Decoder | 300,176,768 | 240.2 s | 48.6 ms | 38.45 | 62.15 |
| **LLaMA 3.2-1B** | Decoder-Only | 1,235,814,400 | **0.0 s** | 115.4 ms | 42.60 | 66.80 |
| **Gemma 2-2B-IT** | Decoder-Only | 2,614,341,888 | **0.0 s** | 182.2 ms | **44.90** | **68.45** |

### Benchmark Plots:
- Translation Quality (BLEU & chrF): `results/plots/bleu_chrf_comparison.png`
- Computational Footprint (Latency vs Parameters): `results/plots/latency_vs_parameters.png`
- Encoder-Decoder vs Decoder-Only Tradeoff: `results/plots/encoder_decoder_vs_decoder_only.png`

---

## 6. Model Comparison and Evaluation

### Encoder–Decoder vs. Decoder–Only Architectures:

1. **Pretraining Objective & Task Alignment:**
   - **Encoder–Decoder (mT5):** Pretrained on span corruption (denoising), naturally aligned with mapping one sequence to another. The bidirectional encoder processes the entire source sentence simultaneously, granting unconstrained receptive fields.
   - **Decoder–Only (LLaMA, Gemma):** Pretrained on causal next-token prediction across trillions of diverse tokens. They exhibit remarkable zero-shot translation capabilities because multilingual alignment emerges naturally during massive pretraining.

2. **Computational & Latency Efficiency:**
   - The encoder–decoder models exhibit significantly higher inference throughput. For a 20-token translation, mT5 evaluates the encoder once and generates tokens sequentially, incurring **48.6 ms** latency.
   - Decoder-only LLMs process the system prompt, few-shot exemplars, and source input sequentially inside the causal attention context window, yielding higher memory consumption and **$2.5\times - 3.7\times$ higher inference latency** (115.4 ms – 182.2 ms).

3. **Tokenization and Vocabulary Size:**
   - **Basic Transformer:** Restricted word-level vocabulary (approx. 4,500 words), leading to out-of-vocabulary (`<unk>`) token issues on rare morphological variants.
   - **Gemma & mT5:** Large multilingual SentencePiece subword vocabularies (250k–256k tokens) ensure that every token is decomposed cleanly without vocabulary bottlenecks.

---

## 7. Sampling-Temperature Analysis

The softmax probability distribution over vocabulary $V$ at step $t$ is parameterized by temperature $T$:
$$P(y_t = w \mid y_{<t}, x) = \frac{\exp(z_w / T)}{\sum_{v \in V} \exp(z_v / T)}$$

We evaluated decoding behavior across $T \in [0.0, 0.2, 0.5, 0.7, 1.0, 1.2]$:

| Temperature ($T$) | BLEU Score | chrF Score | Distinct-1 Ratio | Distinct-2 Ratio | Generation Behavior |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0.0 (Greedy)** | **24.35** | **45.80** | 0.382 | 0.591 | Deterministic, highest fidelity, minimal hallucination |
| **0.2** | 23.80 | 45.10 | 0.395 | 0.612 | Highly focused, near-greedy stability |
| **0.5** | 21.40 | 42.60 | 0.441 | 0.678 | Moderate lexical variation, occasional minor drift |
| **0.7** | 17.65 | 38.90 | 0.498 | 0.742 | Increased lexical creativity, occasional grammatical errors |
| **1.0** | 12.10 | 32.40 | 0.582 | 0.814 | High entropy, frequent word substitutions, degraded syntax |
| **1.2** | 7.45 | 26.80 | **0.654** | **0.865** | Hallucinatory degeneration, disconnected tokens |

### Temperature Analysis Visualization:
`results/plots/temperature_analysis.png` illustrates that translation quality (BLEU/chrF) peaks at $T=0.0$ and monotonically degrades as temperature increases, whereas lexical diversity (Distinct n-grams) expands. For precision-driven translation, **greedy decoding ($T=0.0$) or low temperature ($T \le 0.2$) is essential**.

---

## 8. Responsible and Ethical AI Evaluation

A structured qualitative analysis (`results/ethical_evaluation_cases.csv`) assessed four critical ethical risks:

1. **Gender and Occupational Bias:**
   - *Case:* Translating *"The doctor examined the patient and she ordered an MRI."*
   - *Observation:* The Basic Transformer defaulted to masculine agreement (*"il"*), reflecting patriarchal frequency biases in corpora. Pretrained models (mT5, LLaMA, Gemma) correctly identified the feminine antecedent (*"she"* $\rightarrow$ *"elle"*).
   - *Case:* Translating *"The nurse completed his shift at midnight."*
   - *Observation:* The Basic Transformer erroneously translated *"his"* as feminine (*"son travail [infirmière]"*). LLaMA and Gemma accurately generated the masculine noun *"l'infirmier"*.

2. **Hallucinated Information & Polysemy:**
   - *Case:* Polysemous term *"bank"* in *"The children sat on the bank of the river."*
   - *Observation:* Basic Transformer generated *"banque"* (financial institution). Large models properly disambiguated to *"la rive / berge"*. However, decoder-only LLMs occasionally introduced conversational embellishments not present in the source text.

3. **Cultural & Idiomatic Preservation:**
   - *Case:* Idiomatic proverb *"It is raining cats and dogs"* and *"Kill two birds with one stone"*.
   - *Observation:* Basic Transformer produced nonsensical word-for-word calques (*"des chats et des chiens"*, *"tué deux oiseaux"*). mT5, LLaMA, and Gemma correctly substituted culturally authentic French idioms (*"Il pleut des cordes"*, *"faire d'une pierre deux coups"*).

4. **Sensitive Content Translation:**
   - *Case:* News regarding civil protest dispersed by tear gas.
   - *Observation:* Models translated factual civic terms (*"gaz lacrymogènes"*) neutrally without sensationalist escalation or inappropriate safety refusal triggers.

### Recommended Mitigation Measures:
- **Counterfactual Data Augmentation (CDA):** Balance gendered occupational pairs in training datasets.
- **Prompt Guardrails:** Explicitly command LLMs to resolve gender disambiguation and preserve cultural equivalences.
- **Constrained Decoding:** Enforce phrase-table or dictionary constraints during beam search for idiomatic expressions.
- **Human-in-the-Loop Validation:** Mandatory bilingual human review for legal, medical, and governmental translation deployments.

---

## 9. Conclusion: Best-Performing Model

Based on our empirical and qualitative findings:

- **Best Overall Translation Quality:** **Gemma-2-2B-IT** is the superior model, achieving the highest **BLEU (44.90)** and **chrF (68.45)**. Its instruction fine-tuning and 256k subword vocabulary allow it to capture nuanced grammatical gender, French syntax, and cultural idioms without any task-specific fine-tuning.
- **Best Production & Computational Efficiency:** **mT5-small** (300M parameters) is the most practical choice for latency-critical and embedded production settings. It achieves strong translation quality (**BLEU: 38.45**) while operating at **$2.5\times$ lower inference latency** and requiring a fraction of the GPU/CPU memory required by multi-billion parameter LLMs.

---

## Repository Structure

```
fgai-assignment3-machine-translation/
├── data/                               # Parallel corpus train/val/test splits
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── models/                             # Trained model checkpoints & vocabs
│   ├── basic_transformer.pt
│   ├── vocab_src.json
│   └── vocab_tgt.json
├── results/                            # Benchmark metrics & predictions
│   ├── model_comparison_metrics.csv
│   ├── unified_translations_comparison.csv
│   ├── temperature_analysis_results.csv
│   ├── ethical_evaluation_cases.csv
│   └── plots/                          # High-resolution figures
│       ├── bleu_chrf_comparison.png
│       ├── latency_vs_parameters.png
│       ├── encoder_decoder_vs_decoder_only.png
│       └── temperature_analysis.png
├── src/                                # Core library modules
│   ├── data_loader.py
│   ├── transformer_model.py
│   ├── trainer.py
│   ├── metrics.py
│   └── llm_evaluator.py
├── scripts/                            # Execution scripts
│   ├── 01_download_and_preprocess.py
│   ├── 02_train_basic_transformer.py
│   ├── 03_finetune_mt5.py
│   ├── 04_evaluate_llama_gemma.py
│   ├── 05_benchmark_all_models.py
│   ├── 06_sampling_temperature_analysis.py
│   ├── 07_ethical_bias_analysis.py
│   ├── 08_generate_submission_doc.py
│   └── generate_notebook.py
├── machine_translation_benchmark.ipynb # Self-contained Jupyter Notebook
├── requirements.txt                    # Project dependencies
├── README.md                           # Comprehensive documentation
└── YourName_RegNum_FGAI_Assignment3.docx # Step 3 submission document
```
