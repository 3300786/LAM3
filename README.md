
---

# **LAM3: Leak Alignment in Multi-Modal Models**

**Leak Alignment in Multi-Modal Models (LAM3)** is a research framework for analyzing **alignment leakage**, **cross-modal inconsistencies**, and **multimodal jailbreak vulnerabilities** in modern MLLMs.
LAM3 provides an end-to-end reproducible pipeline for:

* **Multimodal jailbreak attack evaluation**
* **Cross-model inconsistency measurement**
* **Synergy-based vulnerability analysis**
* **Multi-source judge system**
* **Toxicity / PPL / refusal-rate metrics**
* **Dataset processing + visualization + reporting**

LAM3 is designed as a **research collaborator toolkit**, supporting rigorous experiments, metric computation, and paper-ready plots.

---

## **1. Features**

### ✅ **1.1 Multimodal Jailbreak Evaluation**

* Support image-conditioned & text-only jailbreak scenarios
* Compatible with:

  * **IDEFICS2-8B**
  * **LLaVA-1.5-7B**
  * **Qwen2.5-VL-3B**
  * **Llama-3.2-Vision**
* Raw model outputs are stored for reproducibility.

### ✅ **1.2 Synergy-JBV28K Benchmark**

A curated multimodal jailbreak subset (~28K samples).
LAM3 provides:

* Dataset loader
* Meta-builder
* Synergy labeling (“strict”, “textdom”, “imagedom”)
* D(x): multimodal inconsistency score

### ✅ **1.3 Multi-Source Judge System**

Automatically evaluates model outputs using:

* **Qwen-Judge (local)**
* **Detoxify + Toxic-BERT**
* **OpenAI / Perspective API (optional)**
* Combined metrics:

  * **ASR** – Attack Success Rate
  * **Toxicity**
  * **FRR** – False Refusal Rate
  * **PPL** – Language fluency proxy
  * **Refusal intent detector**

Judge outputs are saved as `.jsonl` for further training or auditing.

### ✅ **1.4 Cross-Model Inconsistency**

A unified pipeline to measure:

* Text-only vs image-only vs multimodal responses
* Latent inconsistency:

  * Decomposed into **S(x)**, **I(x)**, **synergy-risk**, etc.
* Full visualization support:

  * Risk histograms
  * ASR bar charts
  * D(x) vs ASR curves
  * Cross-modal correlation plots

### ✅ **1.5 Reproducible Experiment Scripts**

Under `/scripts`:

* **run_synergy_jbv28k.py** – generate raw model outputs
* **eval_synergy_qwen_asr.py** – compute ASR/Refusal/Toxicity
* **cross_model_inconsistency.py** – multimodal inconsistency analysis
* **data tools** – meta builder, CSV tools, filtering tools
* **vis_scripts** – plot synergy maps, histograms, layer-wise visualizations

All scripts accept CLI config via `--cfg` and YAML settings in `/configs`.

---

## **2. Project Structure**

```
LAM3/
│
├── src/
│   ├── models/              # MLLM wrappers (IDEFICS, LLaVA, Qwen-VL, Llama3.2-Vision...)
│   ├── metrics/             # ASR, toxicity, PPL, FRR, synergy metrics, inconsistency
│   ├── judge/               # Qwen-Judge + rule-based evaluators
│   ├── data/                # dataset tools for JBV28K, meta builder
│   └── utils/               # runtime configs, loaders, logging utils
│
├── scripts/
│   ├── run_synergy_jbv28k.py
│   ├── eval_synergy_qwen_asr.py
│   ├── cross_model_inconsistency.py
│   └── vis_*.py
│
├── configs/
│   ├── models.yaml
│   ├── synergy_jbv28k.yaml
│   ├── toxicity.yaml
│   └── runtime.yaml
│
├── data/
│   ├── JailBreakV_28K/
│   └── synergy_jbv28k/
│
└── outputs/
    ├── logs/
    ├── metrics/
    └── plots/
```

---

## **3. Quick Start**

### **3.1 Prepare environment**

```bash
conda create -n lam3-py310 python=3.10
pip install -r requirements.txt
```

### **3.2 Configure models**

Edit:

```
configs/models.yaml
```

Example snippet:

```yaml
llava15_7b:
  repo_id: /data2/.../LLaVA-7B/
  revision: main
```

### **3.3 Run synergy evaluation**

```bash
python -m scripts.run_synergy_jbv28k \
    --cfg configs/synergy_jbv28k.yaml
```

### **3.4 Judge & compute metrics**

```bash
python -m src.metrics.eval_synergy_qwen_asr \
    --raw_in outputs/logs/...raw.jsonl \
    --judged_out outputs/metrics/...qwen_judge.jsonl \
    --metrics_dir outputs/metrics/qwen_asr/
```

### **3.5 Plot results**

```bash
python -m src.metrics.eval_synergy_qwen_asr --plot_only
```

---

## **4. Research Concepts**

### **4.1 Synergy-Leakage**

LAM3 formalizes multimodal jailbreak leakage via:

* **ASR(text+img) > max(ASR(text-only), ASR(img-only))**
* Cross-modal risk amplification
* D(x): latent misalignment score
* Strict / Textdom / Imagedom synergy taxonomy

### **4.2 Cross-Modal Inconsistency**

Defined by comparing model decisions under:

* `txt_only(x)`
* `img_only(x)`
* `txt+img(x)`
* `none(x)` (baseline refusal)

LAM3 decomposes this into:

* Semantic inconsistency
* Safety inconsistency
* Refusal inversion
* Synergy-driven divergence

---

## **5. Visualization Examples**

Under `/outputs/plots/`:

* **asr_refusal_bars.png**
* **risk_diff_hist.png**
* **D_vs_ASR_curve.png**
* **cross_modal_matrix.png**
* **synergy_summary.png**

LAM3 automatically regenerates these during metric evaluation.

---

## **6. Future Work**

* Gradient-based cross-attention adversarial trigger generation
* Llama-3.2-Vision cross-modal anomaly visualizations
* Unified safety alignment leakage benchmark
* Integration with reinforcement-based red-teaming loops
* Support for more models (DeepSeek-VL, Pixtral, MUSE)

---

## **7. License**

MIT License.

---

## **8. Citation**

If LAM3 is used in academic work:

```text
Wang et al. 
“LAM3: Leaky Alignment in Multi-Modal Models.”
2025. GitHub Repository: https://github.com/3300786/LAM3
```

---

