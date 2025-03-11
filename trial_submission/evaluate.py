import os
import subprocess

# Define the arrays for data files, prompt names, and model names.
data_files = ["sample", "validation"]
prompt_names = ["zero_shot_with_instructions", "zero_shot", "few_shot", "comprehensive_few_shot"]
models = [
    # "deepseek-reasoner", 
    # "Qwen/Qwen2.5-14B-Instruct", 
    # "Qwen/Qwen2.5-72B-Instruct", 
    "Qwen/Qwen2.5-7B-Instruct"
]
temperature = 0.1
threshold = 0.6

# Loop over every combination
for data_file in data_files:
    for prompt in prompt_names:
        for model in models:
            # Replace "/" with "_" in model names for file naming consistency.
            model_name_in_file = model.replace("/", "_")
            submission_file = f"experiments_results/{data_file}_output_{prompt}_{model_name_in_file}_t{temperature}.json"
            input_path = f"../data_splits/{data_file}.json"
            
            # Check if the submission file exists before evaluation.
            if not os.path.exists(submission_file):
                print(f"Skipping evaluation. Submission file not found: {submission_file}")
                continue
            
            # Construct the command to run the evaluation script.
            cmd = [
                "python",
                "../eval_scripts/evaluation.py",
                "--metric", "similarity",
                "--input_path", input_path,
                "--submission_path", submission_file,
                "--threshold", str(threshold)
            ]
            
            print(f"Running evaluation for: {submission_file}")
            # Call the evaluation script.
            subprocess.run(cmd)
