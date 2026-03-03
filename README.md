# LoRA-Tuned Large Language Models for Time-Series Forecasting

**Author**: Saam Nazempour (sn665)  
**Course**: MPhil in Data Intensive Science – M2 Coursework  
**Supervisor**: Dr. Miles Cranmer  

---

## Project Overview

This project explores the use of LoRA (Low-Rank Adaptation) for fine-tuning the Qwen2.5-0.5B-Instruct large language model (LLM) to perform time-series forecasting. Specifically, we target forecasting the predator-prey dynamics governed by the Lotka–Volterra equations.

Using the LLMTIME framework, numerical time series data is encoded as text, allowing Qwen to make autoregressive predictions over future values. The model is trained and evaluated under strict compute constraints, with FLOPs accounting for all experiments.

Our results demonstrate that with efficient fine-tuning and thoughtful hyperparameter selection, instruction-tuned LLMs can accurately forecast nonlinear dynamical systems with strong long-range structure.

---

# 📄 Read the Full Report

**[Click here to view the PDF](Report.pdf)**

---

## Repository Structure

```plaintext
M2_STUFF/
├── csv/                         # Exported data or intermediate CSVs
├── notebooks/                   # Clean Jupyter notebooks for results/analysis
├── plots/                       # Output plots and figures for forecasts/results
├── sn665/                       # (Optional) user-specific or submission directory
├── src/                         # Python source code
├── Development.ipynb            # Very rough dev workflow (kept for interest)
├── lora_skeleton.py             # Training script extended with LoRA
├── lotka_volterra_data.h5       # Provided predator-prey dataset
│
├── llmtime.pdf                  # LLMTIME paper reference
├── qwen.pdf                     # Qwen2.5 architecture report
├── qwen.py                      # Script for loading Qwen2.5-Instruct via HuggingFace
│
├── M2_coursework.pdf            # Coursework instructions
├── Report.pdf                   # Final report
└── README.md                    # README
```

---

## Setup & Usage

All preprocessing, training, forecasting, evaluation, and plotting are performed via Jupyter notebooks in the notebooks/ directory. To reproduce the workflow:


1. Environment Setup

```bash
    python3 -m venv M2_venv
    source M2_venv/bin/activate
    pip install -r requirements.txt
```

2. Launch Notebooks

```bash
    jupyter notebook notebooks/
```


3. Notebook Workflow

Run the notebooks in the following order to fully reproduce the project pipeline:

| Notebook                      | Purpose                                                                 |
|------------------------------|-------------------------------------------------------------------------|
| `FLOPS.ipynb`                | Implements FLOPs estimation functions for training and inference         |
| `Forecasting.ipynb`          | Runs autoregressive forecasting on the tokenized Lotka–Volterra systems |
| `LoRA_LR_Training.ipynb`     | Hyperparameter sweep: LoRA rank × learning rate                         |
| `CTX_Training.ipynb`         | Experiments on context length variation                                 |
| `Final Model Training.ipynb` | Full 15,000-step training of best model configuration                   |
| `PerformanceMetrics.ipynb`   | Evaluation of the final model using MSE, R², DTW, and correlation        |



4. Outputs

- Forecast visualizations are saved to plots/

- Performance metrics and results are logged within each notebook

- FLOPs estimates for all runs are included per the coursework spec




--- 

## Key Results


From our final model (LoRA rank = 8, LR = 1e-4, context = 256):


- Median R²: > 0.85 across 10 evaluation systems

- Total FLOPs: 6.86×10¹⁶ (within coursework constraint of 1×10¹⁷)

    - Metrics:

    - MSE: as low as 0.0052

    - Pearson correlation: up to 0.99

    - DTW: < 1.0 for most systems



--- 

## Feautures

- LoRA adaptation for efficient model fine-tuning

- FLOPs tracking & compute budgeting per experiment

- Reproducible training pipeline using HuggingFace

- Token-level time series modeling with LLMTIME

- Metrics & plots for rigorous forecast evaluation

- Modular folder structure with src/, plots/, notebooks/



---

## License

This repository is part of a university coursework submission and is not intended for public distribution or reuse.


---

## Use of Generative AI

- I used Github's Copilot to help me automatically finish off some code blocks and also to quickly docstring my functions.

- I used LLMs (ChatGPT) to help me create professional looking plots and occasionally to help me debug errors when i implemented something incorrectly.