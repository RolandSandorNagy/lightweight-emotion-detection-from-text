# AI Lab I–II  
## Constraint-Aware Lightweight Multi-Label Emotion Detection

This repository contains the research code and documentation for the AI Lab I–II project on lightweight emotion detection in text.

The project focuses on **multi-label emotion classification under explicit resource constraints**, with particular emphasis on rare (tail) emotion categories and decision strategies (thresholding and calibration).

---

## 🎯 Research Focus

The primary research scenario investigates:

> Which transformer architectures and decision strategies satisfy predefined resource constraints (model size, memory, latency), and among those, which configuration maximizes performance on rare emotion categories?

The study is conducted on the GoEmotions dataset (28-label multi-label setup).

Key aspects:

- Multi-label transformer models (e.g., BERT-base, DistilBERT, MiniLM)
- Fixed vs optimized threshold strategies
- Global vs per-class thresholds
- Calibration techniques (e.g., temperature scaling)
- Multi-label vs single-label formulation (tail performance focus)
- Constraint-aware model comparison

---

## 🧪 Experimental Components

- Multi-label training with BCE loss
- Threshold optimization (global and per-class)
- Calibration analysis
- Rare emotion (tail label) performance evaluation
- Macro-F1, Micro-F1, per-class metrics
- Latency and model size measurements

---

## 📂 Repository Structure (planned)


├── proposal/ # Research proposal documents
│
├── configs/ # Training & experiment configs
│
├── src/ # Source code
│ │
│ ├── data/
│ │
│ ├── models/
│ │
│ ├── training/
│ │
│ ├── evaluation/
│ │
│ └── utils/
│
├── experiments/ # Saved experiment logs and results
│
├── notebooks/ # Exploratory notebooks (optional)
│
└── README.md


---

## ⚙️ Reproducibility

The project aims to ensure:

- Fixed random seeds
- Config-based experiment definitions
- Explicit reporting of hardware constraints
- Clear evaluation protocol

---

## 📌 Status

This is an active AI Lab research project.  
The initial proposal (v0.1) defines the scenario, constraints, and research questions.  
Experiments are currently being structured.

---

## 📜 License

To be determined.