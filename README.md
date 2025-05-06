
# 🧠 Critical Questions Generation 
This repository contains the setup for our **Critical Question Generation** project for argumentative texts, combining **ML**, **LLM**, and **Reinforcement Learning** approaches.  
Developed by:  
**Alaa Elsetohy · Sama Hadhoud · Mariam Barakat**

Repo Link: https://github.com/Myriam2002/Critical_Question_generation

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

RL environment (optional)

Running the RL scripts requires extra dependencies.
Create the Conda env from the provided spec:

```bash
conda env create -f RL/RL_environment.yml
```

## 🚀 How to Run

### Benchmarking:
```bash
python trial_submission/generate_benchmark.py
```
Results will appear under `trial_submission/experiments_results_benchmark/`
All reported benchmarks are in this folder also

### Iterative Agentic Approach
The code for the Iterative Agentic Approach is located in the `Iteratively Agentic Approach/` folder. It includes three Jupyter notebooks, each implementing a different feedback architecture:

| File               | Description                              |
|--------------------|------------------------------------------|
| `Approach_1.ipynb` | ML-only feedback loop (ML evaluator)     |
| `Approach_2.ipynb` | LLM-only feedback loop (LLM evaluator)   |
| `Approach_3.ipynb` | Hybrid ML + LLM feedback loop            |

#### 🟢 Steps to Run

1. **Navigate to the folder:**
   ```bash
   cd "Iteratively Agentic Approach"
   Open a notebook using Jupyter Notebook or any compatible IDE (e.g., VS Code, Google Colab).
2. Open a notebook using Jupyter Notebook or any compatible IDE (e.g., VS Code, Google Colab).
3. Run all cells in the chosen notebook. Each notebook will:
     - Load a set of intervention texts and candidate questions
     - Apply the respective feedback loop (ML, LLM, or Hybrid)
     - Output improved and ranked critical questions
### Theory-based approaches using argumentation schemes

All argumentation schemes are in `theory/templates.json`
To run with them use SchemechoosePrompt in `theory/prompts.py`

Results can be found in `theory/experiments_results_theory`

### Logical fallacy detection baselines

All fallacies are in `theory/fallacies.json`
To run with them use LogicalFallaciesPrompt in `theory/prompts.py`

Results can be found in `theory/experiments_results_theory`

### Reinforcement-learning inference:

Pre-trained LoRA adapters are on the Hugging Face Hub:

| Base model   | Epochs | Adapter repo                                                              |
| ------------ | ------ | ------------------------------------------------------------------------- |
| Qwen 2.4-3B  | 3      | `samahadhoud/critical_questions_generation_qwen_lora_RL_fintuned_3epoch`  |
| Llama 3.1-8B | 3      | `samahadhoud/critical_questions_generation_llama_lora_RL_fintuned_3epoch` |
| Llama 3.1-8B | 5      | `samahadhoud/critical_questions_generation_llama_lora_RL_fintuned_5epoch` |
| Llama 3.1-8B | 7      | `samahadhoud/critical_questions_generation_llama_lora_RL_fintuned_7epoch` |

To run inference, edit `repo_id` inside `RL/RL_inference.py`, then:

```bash
python RL/RL_inference.py 
```
### Reinforcement-learnin Training (optional):
```bash
# Llama-3.1-8B + GRPO
python RL/RL_llama_3.1_8b_GPRO.py

# Qwen-2.5-3B + GRPO
python RL/RL_qwen_2.5_3b_GPRO.py
```
All reward-function details and prompt templates live in `RL/RL_data_prep.py`.

## 🔝 Most Import Files 

Scripts will be finalized soon. Expected entry points:

| File                          | Description                                |
|-------------------------------|--------------------------------------------|
| `Data Analysis/dependency_analysis.ipynb`       | Lexical Analysis of the validation dataset        |
| `Iteratively Agentic Approach/Approach*.ipynb`       | Pipeline to use llama405b to generate CQ using ML evaluator, LLM evaluator, and ML & LLM evvaluator|
| `Iteratively Agentic Approach/ml_model_CQ.py`   | Train ML model to classify question quality |
| `RL/RL_llama_3.1_8b_GPRO.py` and    `python RL/RL_qwen_2.5_3b_GPRO.py`       | Reinforcement Learning fine-tuning         |
| `SchemechoosePrompt in theory/prompts.py` | Map texts to argumentation schemes       |
| `LogicalFallaciesPrompt in theory/prompts.py and its handling in generate.py` | Logical fallacy detection baseline         |


## 📂 Repository Structure

```bash
Critical_Question_generation/
├── Data_Analysis/                 # Exploratory notebooks & scripts
├── Iteratively Agentic Approach/  # ML + agentic improvement pipeline
├── data_splits/                   # Train / val / test JSON files
├── eval_scripts/                  # Evaluation helpers & RL code
├── trial_submission/              # Benchmark runs & outputs
├── RL/                            # RL training & inference code
├── theory/                        #  Theory-based approaches using argumentation schemes and Logical fallacy detection baseline
├── requirements.txt               # Base Python deps
├── RL/RL_environment.yml          # Conda spec for RL experiments
├── .env                           # API keys (ignored by Git)
└── README.md                      # You are here 🚀
```

