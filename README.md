# LAM3: Leaky Alignment in Multi-Modal Models

**LAM3** explores the alignment vulnerability of multimodal large language models (MLLMs).  
It proposes an adversarial attack pipeline using cross-attention gradient signals and provides a reproducible benchmark for evaluating jailbreak robustness and defense mechanisms.

## 🌐 Overview
- Models: IDEFICS2, LLaVA 1.5
- Tasks: Multimodal jailbreak attacks & defense benchmarking
- Metrics: ASR, Toxicity, PPL, FRR
- Evaluation: Multi-source judgment (LLMs + API + human review)
- Goal: Reveal alignment leakage in current MLLMs and provide reliable baselines for defense design.

## 🧠 Tech Stack
Python 3.10 / PyTorch / Hugging Face Transformers / FastAPI / SQLite / W&B

## 📄 License
MIT License
