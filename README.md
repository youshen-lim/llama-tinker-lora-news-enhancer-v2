# Llama 3.2-1B LoRA Fine-Tuning for News Analysis
## Comparative Study: Tinker API vs. Unsloth Local Training

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Llama 3.2](https://img.shields.io/badge/model-Llama%203.2--1B-green.svg)](https://huggingface.co/meta-llama/Llama-3.2-1B)

---

---

## 🚀 Project Evolution: From Pilot (v1) to Production (v2)

> **This is Version 2** - A significant expansion and improvement over the pilot project

### 📈 Dataset Expansion: 4.5x Growth

This project represents the **second iteration** of fine-tuning Llama 3.2-1B for AI news metadata generation:

| Version | Dataset Size | Status | Links |
|---------|--------------|--------|-------|
| **v1 (Pilot)** | 101 hand-annotated examples | Proof of concept | [GitHub](https://github.com/youshen-lim/llama-tinker-lora-news-enhancer) • [Hugging Face](https://huggingface.co/Truthseeker87/llama-tinker-lora-news-enhancer) |
| **v2 (Current)** | **460 hand-annotated examples** | Production-ready | This repository |

### 🎯 Key Improvements in v2

- ✅ **4.5x larger dataset** (101 → 460 examples) for more robust training
- ✅ **Three-model comparison** (Baseline, Tinker API, Unsloth) instead of single approach
- ✅ **Comprehensive evaluation** with statistical significance testing
- ✅ **Enhanced documentation** with detailed methodology and reproducibility
- ✅ **Production-ready models** with 100% JSON validity and high F1 scores

This expanded dataset enables more reliable fine-tuning and better generalization to diverse AI news content.


## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Findings](#key-findings)
- [Training Methods](#training-methods)
  - [Method A: Tinker API Fine-Tuning](#method-a-tinker-api-fine-tuning)
  - [Method B: Unsloth Local Fine-Tuning](#method-b-unsloth-local-fine-tuning)
- [Technical Implementation](#technical-implementation)
- [Evaluation Results](#evaluation-results)
- [Visualizations](#visualizations)
- [Installation & Usage](#installation--usage)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Project Overview

This project presents a **comprehensive comparative study** of two LoRA (Low-Rank Adaptation) fine-tuning approaches for adapting **Llama 3.2-1B** to structured news analysis tasks. The goal is to transform unstructured newsletter content into structured JSON metadata for enhanced semantic analysis.

### Research Questions

1. **How do managed API-based fine-tuning (Tinker) and local fine-tuning (Unsloth) compare in performance?**
2. **What are the trade-offs in terms of cost, control, and model quality?**
3. **Can a 1B parameter model achieve production-ready performance on structured output tasks?**

### Dataset

- **Total Examples:** 460 annotated news articles
- **Train Set:** 364 examples (80%)
- **Test Set:** 96 examples (20%)
- **Split Method:** Temporal (chronological)
- **Format:** JSONL with instruction-response pairs

### Task Definition

Transform newsletter content into structured JSON with:
- **Category:** Article classification (AI/ML, Business, Research, etc.)
- **Relevance Score:** 0-10 scale rating
- **Key Insights:** Bullet-point summary of main points
- **Company Names:** Extracted entities
- **Summary:** Concise article summary
- **Sentiment:** Positive/Negative/Neutral
- **Sponsored Ad:** Boolean flag

---

## 🏆 Key Findings

### Performance Summary

| Metric | Baseline (No Fine-Tuning) | Tinker API | Unsloth Local | Winner |
|--------|---------------------------|------------|---------------|--------|
| **Overall F1 Score** | 0.396 | **0.612** | 0.603 | 🥇 Tinker |
| **JSON Validation Rate** | 5.2% | **100%** | **100%** | 🥇 Tie |
| **Relevance MAE** | 6.98 | **1.80** | 1.43 | 🥇 Unsloth |
| **ROUGE-1** | 0.052 | **0.600** | 0.561 | 🥇 Tinker |
| **BERTScore F1** | 0.391 | **0.744** | 0.735 | 🥇 Tinker |
| **Company Names F1** | 0.656 | 0.761 | **0.772** | 🥇 Unsloth |
| **Key Insights F1** | 0.135 | **0.464** | 0.435 | 🥇 Tinker |

### Statistical Significance

**Baseline vs. Tinker:**
- Overall F1: +0.216 (p < 0.001, Cohen's d = -0.71) ✅ **Highly Significant**
- Relevance MAE: -5.18 (p < 0.001, Cohen's d = 1.67) ✅ **Highly Significant**

**Baseline vs. Unsloth:**
- Overall F1: +0.207 (p < 0.001, Cohen's d = -0.68) ✅ **Highly Significant**
- Relevance MAE: -5.55 (p < 0.001, Cohen's d = 1.87) ✅ **Highly Significant**

**Tinker vs. Unsloth:**
- Overall F1: -0.009 (p = 0.698, Cohen's d = 0.03) ❌ **Not Significant**
- Relevance MAE: -0.375 (p = 0.126, Cohen's d = 0.16) ❌ **Not Significant**

**Conclusion:** Both fine-tuned models significantly outperform the baseline, with **no statistically significant difference** between Tinker and Unsloth.

---

## 🔧 Training Methods

### Method A: Tinker API Fine-Tuning

**Tinker** is a managed training API service by Thinking Machines that provides:
- Fully managed infrastructure
- Automatic hyperparameter tuning
- Built-in monitoring and logging
- No GPU setup required

#### Configuration

```python
training_config = {
    "model": "meta-llama/Llama-3.2-1B",
    "dataset": "news_train_data.jsonl",
    "epochs": 5,
    "learning_rate": 2e-4,
    "batch_size": 4,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05
}
```

#### Training Process

1. **Upload Dataset:** JSONL format with `messages` field
2. **Configure Training:** Set hyperparameters via API
3. **Monitor Progress:** Real-time loss tracking via dashboard
4. **Download Model:** LoRA adapters + merged model

#### Results

- **Training Time:** ~45 minutes (5 epochs)
- **Final Loss:** 0.3247
- **Cost:** $X.XX (API credits)
- **Infrastructure:** Managed (no setup required)

---

### Method B: Unsloth Local Fine-Tuning

**Unsloth** is an open-source framework for efficient LoRA fine-tuning with:
- 4-bit quantization (QLoRA)
- Memory-efficient training
- Weights & Biases integration
- Full control over training process

#### Configuration

```python
training_config = {
    "model": "unsloth/Llama-3.2-1B",
    "max_seq_length": 2048,
    "load_in_4bit": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "learning_rate": 2e-4,
    "num_train_epochs": 5,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 10,
    "logging_steps": 1,
    "optim": "adamw_8bit"
}
```

#### Training Process

1. **Setup Environment:** Install Unsloth + dependencies
2. **Load Model:** 4-bit quantized Llama 3.2-1B
3. **Configure LoRA:** Apply adapters to target modules
4. **Train:** Local GPU training with W&B logging
5. **Save Model:** LoRA adapters + merged model

#### Results

- **Training Time:** ~2 hours (5 epochs on T4 GPU)
- **Final Loss:** 0.2891
- **Cost:** Free (Google Colab T4)
- **Infrastructure:** Self-managed (requires GPU setup)

---

## 🔬 Technical Implementation

### Data Format Handling

#### Tinker API Format
```json
{
  "messages": [
    {"role": "system", "content": "You are a news analyst..."},
    {"role": "user", "content": "Analyze this article..."},
    {"role": "assistant", "content": "{\"category\": \"AI/ML\", ...}"}
  ]
}
```

#### Unsloth Format
```python
# Uses ChatML template with special tokens
"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n..."
```

### Training Methodology Comparison

| Aspect | Tinker API | Unsloth Local |
|--------|-----------|---------------|
| **Infrastructure** | Managed cloud | Self-hosted (Colab/local) |
| **GPU Required** | No | Yes (T4/V100/A100) |
| **Quantization** | Full precision | 4-bit (QLoRA) |
| **Batch Size** | 4 | 2 (effective: 8 with grad accum) |
| **Optimizer** | AdamW | AdamW 8-bit |
| **Logging** | Built-in dashboard | Weights & Biases |
| **Cost** | API credits | Free (Colab) / GPU cost |
| **Control** | Limited | Full control |
| **Setup Time** | Minutes | 30-60 minutes |

---

## 📊 Evaluation Results

### Detailed Metrics

#### F1 Scores by Field

| Field | Baseline | Tinker | Unsloth | Best |
|-------|----------|--------|---------|------|
| **Key Insights** | 0.135 | 0.464 | 0.435 | Tinker |
| **Company Names** | 0.656 | 0.761 | 0.772 | Unsloth |
| **Overall** | 0.396 | 0.612 | 0.603 | Tinker |

#### ROUGE Scores

| Metric | Baseline | Tinker | Unsloth |
|--------|----------|--------|---------|
| **ROUGE-1** | 0.052 | 0.600 | 0.561 |
| **ROUGE-2** | 0.012 | 0.377 | 0.343 |
| **ROUGE-L** | 0.052 | 0.552 | 0.524 |

#### BERTScore

| Metric | Baseline | Tinker | Unsloth |
|--------|----------|--------|---------|
| **Precision** | 0.370 | 0.752 | 0.745 |
| **Recall** | 0.417 | 0.741 | 0.731 |
| **F1** | 0.391 | 0.744 | 0.735 |

#### Relevance Prediction

| Metric | Baseline | Tinker | Unsloth |
|--------|----------|--------|---------|
| **MAE** | 6.98 | 1.80 | 1.43 |
| **Adjusted R²** | -4.268 | 0.184 | 0.435 |

#### Output Quality

| Metric | Baseline | Tinker | Unsloth |
|--------|----------|--------|---------|
| **JSON Validation Rate** | 5.2% | 100.0% | 100.0% |
| **Avg Response Length** | 16.2 | 36.7 | 34.9 |
| **Exact Match Rate** | 0.0% | 7.3% | 3.1% |

---

## 📈 Visualizations

### Performance Comparison Charts

#### F1 Score by Field

![F1 Score by Field](./evaluation_results/visualizations/f1_score_by_field_20251111_084543.png)

#### Forest Plot (Statistical Comparison)

![Forest Plot (Statistical Comparison)](./evaluation_results/visualizations/forest_plot_three_models_20251111_084543.png)

#### BERTScore Comparison

![BERTScore Comparison](./evaluation_results/visualizations/bertscore_comparison_20251111_084543.png)

#### ROUGE Scores Comparison

![ROUGE Scores Comparison](./evaluation_results/visualizations/rouge_scores_comparison_20251111_084543.png)

#### JSON Validation Rate

![JSON Validation Rate](./evaluation_results/visualizations/json_validation_comparison_20251111_084543.png)

#### Precision-Recall Comparison

![Precision-Recall Comparison](./evaluation_results/visualizations/precision_recall_comparison_20251111_084543.png)

#### Relevance Error Distribution

![Relevance Error Distribution](./evaluation_results/visualizations/relevance_error_distribution_20251111_084543.png)

#### Response Length Distribution

![Response Length Distribution](./evaluation_results/visualizations/response_length_histogram_20251111_084543.png)

#### Response Length Boxplot

![Response Length Boxplot](./evaluation_results/visualizations/response_length_boxplot_20251111_084543.png)

#### Response Length Violin Plot

![Response Length Violin Plot](./evaluation_results/visualizations/response_length_violin_20251111_084543.png)

#### Word Count Boxplot

![Word Count Boxplot](./evaluation_results/visualizations/word_count_boxplot_20251111_084543.png)

#### Output Consistency

![Output Consistency](./evaluation_results/visualizations/output_consistency_comparison_20251111_084543.png)

#### Statistical Summary Heatmap

![Statistical Summary Heatmap](./evaluation_results/visualizations/statistical_summary_heatmap_20251111_084543.png)

---

## 🚀 Installation & Usage

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (for Unsloth local training)
- 16GB+ RAM recommended
- Google Colab (free T4 GPU) or local GPU

### Installation

```bash
# Clone the repository
git clone https://github.com/youshen-lim/llama-tinker-lora-news-enhancer-v2.git
cd llama-tinker-lora-news-enhancer-v2

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your HuggingFace token and Tinker API key
```

### Environment Variables

Create a `.env` file with:

```bash
HF_TOKEN=your_huggingface_token_here
TINKER_API_KEY=your_tinker_api_key_here
```

### Running the Notebook

```bash
# Launch Jupyter
jupyter notebook llama_tinker_lora_news_enhancer_v2.ipynb
```

Or open in Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/youshen-lim/llama-tinker-lora-news-enhancer-v2/blob/main/llama_tinker_lora_news_enhancer_v2.ipynb)

### Using the Fine-Tuned Models

#### Option 1: Load LoRA Adapters

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    device_map="auto"
)

# Load LoRA adapters (Tinker or Unsloth)
model = PeftModel.from_pretrained(
    base_model,
    "./tinker_models/lora_adapter"  # or "./unsloth_models/lora_adapter"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Generate predictions
prompt = "Analyze this news article: ..."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
```

#### Option 2: Download Merged Models from Hugging Face

Large merged model files (>100MB) are hosted on Hugging Face:

```bash
# Download Tinker merged model (~2.3GB)
huggingface-cli download youshen-lim/llama-3.2-1b-tinker-news-enhancer --local-dir ./tinker_models/merged

# Download Unsloth merged model (~2.3GB)
huggingface-cli download youshen-lim/llama-3.2-1b-unsloth-news-enhancer --local-dir ./unsloth_models/merged
```

See [MODEL_DOWNLOAD_INSTRUCTIONS.md](./MODEL_DOWNLOAD_INSTRUCTIONS.md) for detailed instructions.

---

## 📁 Repository Structure

```
llama-tinker-lora-news-enhancer-v2/
├── README.md                                    # This file
├── LICENSE                                      # MIT License
├── .gitignore                                   # Git ignore rules
├── requirements.txt                             # Python dependencies
├── llama_tinker_lora_news_enhancer_v2.ipynb     # Main notebook
├── news_analyst_version_2_0.py                  # Python script version
├── LLM_FINETUNING_PROJECT_CONSOLIDATED_DOCUMENTATION.md
│
├── training_data/                               # Training datasets
│   ├── news_train_data.jsonl                    # 364 training examples
│   ├── news_test_data.jsonl                     # 96 test examples
│   └── news_training_annotated.jsonl            # 460 total examples
│
├── evaluation_results/                          # Evaluation outputs
│   ├── metrics/                                 # JSON metrics files
│   │   ├── three_model_quality_metrics_aggregated_*.json
│   │   ├── three_model_quality_metrics_per_example_*.json
│   │   └── three_model_statistical_inference_*.json
│   ├── visualizations/                          # 13 PNG charts
│   └── Final_Comparison/                        # Comprehensive results
│
├── tinker_models/                               # Tinker API models
│   └── lora_adapter/                            # LoRA adapters (~50MB)
│
└── unsloth_models/                              # Unsloth local models
    └── lora_adapter/                            # LoRA adapters (~50MB)
```

---

## 🔍 Debugging Journey & Lessons Learned

### Key Issues Encountered

1. **Data Format Mismatch**
   - **Problem:** Tinker API expects `messages` format, Unsloth uses ChatML
   - **Solution:** Created separate preprocessing pipelines for each method

2. **Training Instability**
   - **Problem:** Initial Unsloth training showed high loss variance
   - **Solution:** Enabled `train_on_responses_only=True` to focus on assistant responses

3. **JSON Validation Failures**
   - **Problem:** Baseline model produced invalid JSON 94.8% of the time
   - **Solution:** Fine-tuning with structured examples → 100% validation rate

4. **Relevance Score Prediction**
   - **Problem:** Baseline MAE of 6.98 (on 0-10 scale)
   - **Solution:** Fine-tuning reduced MAE to 1.43-1.80

### Best Practices

✅ **Use `train_on_responses_only=True`** for instruction-following tasks
✅ **Monitor loss curves** to detect overfitting early
✅ **Validate JSON outputs** during evaluation
✅ **Use temporal splits** for time-series data
✅ **Track multiple metrics** (F1, ROUGE, BERTScore) for comprehensive evaluation

---

## 💡 Key Takeaways

### When to Use Tinker API

✅ **Quick prototyping** - Get started in minutes
✅ **No GPU access** - Fully managed infrastructure
✅ **Production deployments** - Built-in monitoring and logging
✅ **Team collaboration** - Centralized training management

### When to Use Unsloth Local

✅ **Full control** - Customize every aspect of training
✅ **Cost optimization** - Free with Colab or local GPU
✅ **Research experiments** - Iterate quickly on hyperparameters
✅ **Privacy requirements** - Keep data on-premises

### Performance Verdict

**Both methods achieve comparable performance** (no statistically significant difference in overall F1 score). Choose based on your infrastructure, budget, and control requirements.

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{lim2025llama32news,
  author = {Lim, Aaron (Youshen)},
  title = {Llama 3.2-1B LoRA Fine-Tuning for News Analysis: Tinker API vs. Unsloth},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/youshen-lim/llama-tinker-lora-news-enhancer-v2}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Meta AI** for Llama 3.2 models
- **Thinking Machines** for Tinker API
- **Unsloth AI** for the Unsloth framework
- **Hugging Face** for transformers and PEFT libraries
- **Google Colab** for free GPU access

---

## 📧 Contact

**Aaron (Youshen) Lim**
- GitHub: [@youshen-lim](https://github.com/youshen-lim)
- LinkedIn: [linkedin.com/in/youshen](https://www.linkedin.com/in/youshen/)
- Hugging Face: [Truthseeker87/llama-tinker-lora-news-enhancer-v2](https://huggingface.co/Truthseeker87/llama-tinker-lora-news-enhancer-v2)

---

**Last Updated:** {datetime.now().strftime('%B %d, %Y')}
