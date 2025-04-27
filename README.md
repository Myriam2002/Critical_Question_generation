
# 🧠 Critical Questions Generation 
This repository contains the setup for our **Critical Question Generation** project for argumentative texts, combining **ML**, **LLM**, and **Reinforcement Learning** approaches.  
Developed by:  
**Alaa Elsetohy · Sama Hadhoud · Mariam Barakat**

## 📚 Project Description

This project aims to automatically generate and evaluate critical questions over argumentative texts.  
We experiment with:

- Machine Learning classifiers
- LLM-based generation and feedback loops
- Reinforcement Learning fine-tuning
- Theory-based approaches using argumentation schemes
- Logical fallacy detection baselines

---

## 🛠️ Setup 

You can prepare your environment while waiting for the finalized code.

```bash
git clone https://github.com/Myriam2002/Critical_Question_generation.git
cd YOUR_REPOSITORY
python3 -m venv venv
source venv/bin/activate     # Linux / Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## 🚀 How to Run (Placeholder)
WIP
## 🔝 Most Import Files 

Scripts will be finalized soon. Expected entry points:

| File                          | Description                                |
|-------------------------------|--------------------------------------------|
| `Data Analysis/dependency_analysis.ipynb`       | Lexical Analysis of the validation dataset        |
| `Iteratively Agentic Approach/Approach*.ipynb`       | Pipeline to use llama405b to generate CQ using ML evaluator, LLM evaluator, and ML & LLM evvaluator|
| `Iteratively Agentic Approach/ml_model_CQ.py`   | Train ML model to classify question quality |
| `rl_finetuning.py`            | Reinforcement Learning fine-tuning         |
| `argumentation_scheme_mapper.py` | Map texts to argumentation schemes       |
| `logical_fallacy_detector.py` | Logical fallacy detection baseline         |


## 📂 Repository Structure (

```bash
├── Data_Analysis/                   # Exploratory Data Analysis (EDA) scripts and results
├── Iteratively Agentic Approach/    # ML models and agentic improvement pipeline
├── data_splits/                     # Dataset splits (train, validation, test)
├── eval_scripts/                    # Evaluation scripts and RL fine-tuning code
├── trial_submission/                # Trial submissions and experiment outputs
├── .env                             # Environment variables (OpenAI API keys, etc.)
├── .gitignore                       # Git ignored files configuration
├── LICENSE                          # Project license
├── README.md                        # Project overview (this file)
├── eval.log                         # Evaluation logs
```

