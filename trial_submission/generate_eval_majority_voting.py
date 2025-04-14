import json
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import transformers
# import torch
import logging
import tqdm
import re
import os
# from query_model import query_model, deepinfra_models, openrouter_models, deepseek_models
from prompts_eval import *
from collections import OrderedDict
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import pprint
prompt_classes = {
    "comprehensive_few_shot": ComprehensiveFewShotPrompt
}
LABELS = ["USEFUL", "UNHELPFUL", "INVALID", "UNKNOWN"]

def compute_metrics(confusion_matrix):
    """
    Computes Precision, Recall, and F1 Score for each label using the confusion matrix.
    """
    metrics = {}
    total_correct = 0
    total_samples = 0
    weighted_f1 = 0

    for label in LABELS:
        tp = confusion_matrix[label][label]  # True Positives
        fp = sum(confusion_matrix[l][label] for l in LABELS if l != label)  # False Positives
        fn = sum(confusion_matrix[label][l] for l in LABELS if l != label)  # False Negatives
        tn = sum(confusion_matrix[l][k] for l in LABELS for k in LABELS if l != label and k != label)  # True Negatives

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

        total_correct += tp
        total_samples += tp + fn
        weighted_f1 += f1_score * (tp + fn)

    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    macro_f1 = sum(metrics[label]["f1_score"] for label in LABELS) / len(LABELS)
    weighted_f1 /= total_samples if total_samples > 0 else 0

    metrics["overall_accuracy"] = overall_accuracy
    metrics["macro_f1"] = macro_f1
    metrics["weighted_f1"] = weighted_f1

    return metrics

from collections import Counter

def most_repeated_word(word_list):
    # Count occurrences of each word in the list.
    counts = Counter(word_list)
    
    # Check if all words occur exactly once.
    if all(count == 1 for count in counts.values()):
        return None
    
    # Find the word with the highest frequency.
    # .most_common(1) returns a list with the top (word, count) tuple.
    most_common_word, _ = counts.most_common(1)[0]
    return most_common_word

def main():
    merged_out = {}
    temperature = 0.0
    selected_prompt_names = ["comprehensive_few_shot"]
    models = ['meta-llama/Llama-3.1-8B-Instruct', 
    "Qwen/Qwen2.5-72B-Instruct", 
    "meta-llama/Llama-3.3-70B-Instruct"]
    # models = ["Qwen/Qwen2.5-72B-Instruct"]
    # models = ["deepseek-ai/DeepSeek-V3-0324"]
    # data_files = ["sample", "validation"]
    data_files = ["validation"]


    for data_file in data_files:
        with open(f'../data_splits/{data_file}.json') as f:
            data = json.load(f)

        

        # Label-specific accuracy tracking
        label_counts = {label: 0 for label in LABELS}
        correct_per_label = {label: 0 for label in LABELS}

                # Initialize Confusion Matrix
        confusion_matrix = {true_label: {pred_label: 0 for pred_label in LABELS} for true_label in LABELS}

        total_questions = 0
        correct_predictions = 0
        out = {}
        for model_name in models:
            for selected_prompt_name in selected_prompt_names:



                model_name_in_file = model_name.replace("/", "_")
                target_file_name = f'eval_experiments_results/{data_file}_output_{selected_prompt_name}_{model_name_in_file}_t{temperature}.json'
                
                if os.path.exists(target_file_name):
                    with open(target_file_name, 'r') as f_out:
                        out[f"{model_name}_{selected_prompt_name}"] = json.load(f_out)
                else:
                    print(f"File {target_file_name} does not exist.")
                    exit(1)

        print(f"Starting evaluation: {target_file_name}")
        for key, line in tqdm.tqdm(data.items()):
            text = line['intervention']
            intervention_id = line.get('intervention_id')
            # Store all question results for this intervention
            intervention_results = {
                "intervention_id": intervention_id,
                "intervention_text": text,
                "questions": []
            }

            for question_dict in line['cqs']:
                cq = question_dict['cq']
                golden_label = question_dict['label'].upper()
                question_id = question_dict['id']
                # If the intervention is already processed in the JSON, update global metrics
                print(out.keys)
                predicted_labels_with_name = {}
                predicted_labels = []

                for trial, out_trial in out.items():
                    intervention_results = out_trial[intervention_id]

                    found_it = False
                    for question in intervention_results["questions"]:
                        if question["question_id"] == question_id:
                            # Extract the predicted label from the JSON

                            predicted_label = question["predicted_label"]
                            predicted_labels.append(predicted_label)
                            predicted_labels_with_name[trial] = predicted_label
                            found_it = True
                            break
                    if not found_it:
                        print(f"Question ID {question_id} not found in intervention results.")
                        print("in trial: ", trial)
                        exit(1)

                # predicted_label = most_repeated_word(predicted_labels)
                if predicted_labels != ["USEFUL", "USEFUL", "USEFUL"]:
                    predicted_label = "UNKNOWN"
                if predicted_label is None:
                    predicted_label = "UNKNOWN"
                # Update label-specific tracking
                if golden_label in label_counts:
                    label_counts[golden_label] += 1
                    if predicted_label == golden_label:
                        correct_per_label[golden_label] += 1

                # Update confusion matrix
                if golden_label in LABELS and predicted_label in LABELS:
                    confusion_matrix[golden_label][predicted_label] += 1

                total_questions += 1
                if predicted_label == golden_label:
                    correct_predictions += 1

                

                # Append question result
                intervention_results["questions"].append({
                    "question_id": question_id,
                    "question_text": cq,
                    "golden_label": golden_label,
                    "predicted_label": predicted_label,
                    "predicted_labels_with_name": predicted_labels_with_name,
                })
                print(f"Question ID: {question_id}, Predicted: {predicted_label}, Golden: {golden_label}")
                overall_accuracy = correct_predictions / total_questions if total_questions > 0 else 0
                print(f"Total Questions: {total_questions}, Accuracy: {overall_accuracy}")
                per_label_accuracy = {
                    label: correct_per_label[label] / label_counts[label] if label_counts[label] > 0 else 0
                    for label in label_counts
                }
                print(f"Per-Label Accuracy: {per_label_accuracy}")
                print("\nConfusion Matrix:")
                for true_label, preds in confusion_matrix.items():
                    print(f"{true_label}: {preds}")


            # Store results under intervention ID
            merged_out[intervention_id] = intervention_results
            metrics = compute_metrics(confusion_matrix)
            final_output = OrderedDict()
            final_output["accuracy"] = OrderedDict([
                ("overall", metrics["overall_accuracy"]),
                ("macro_f1", metrics["macro_f1"]),
                ("weighted_f1", metrics["weighted_f1"]),
                ("confusion_matrix", confusion_matrix)
            ])
            merged_out["accuracy"] = final_output

            with open("merged.json", 'w') as o:
                json.dump(merged_out, o, indent=4)
        
        # Compute accuracy
        overall_accuracy = correct_predictions / total_questions if total_questions > 0 else 0
        per_label_accuracy = {
            label: correct_per_label[label] / label_counts[label] if label_counts[label] > 0 else 0
            for label in label_counts
        }

    metrics = compute_metrics(confusion_matrix)

# Ensuring accuracy metrics are first in JSON output
    final_output = OrderedDict()
    final_output["accuracy"] = OrderedDict([
        ("overall", metrics["overall_accuracy"]),
        ("macro_f1", metrics["macro_f1"]),
        ("weighted_f1", metrics["weighted_f1"]),
        ("confusion_matrix", confusion_matrix)
    ])
    merged_out["accuracy"] = final_output

    with open("merged.json", 'w') as o:
        json.dump(merged_out, o, indent=4)

    print(f"\nOverall Accuracy for {data_file}: {metrics['overall_accuracy']:.2%}")
    print(f"Macro F1 Score: {metrics['macro_f1']:.2%}")
    print(f"Weighted F1 Score: {metrics['weighted_f1']:.2%}")
    print("\nConfusion Matrix:")
    for true_label, preds in confusion_matrix.items():
        print(f"{true_label}: {preds}")

    useful_row = confusion_matrix["USEFUL"]
    TP = useful_row["USEFUL"]
    FN = sum(value for label, value in useful_row.items() if label != "USEFUL")

    # For actual OTHER (merging UNHELPFUL, INVALID, and UNKNOWN rows):
    #   - False Positives (FP) = predictions of USEFUL in these rows.
    #   - True Negatives (TN) = predictions of OTHER (non-USEFUL) in these rows.
    other_rows = ["UNHELPFUL", "INVALID", "UNKNOWN"]
    FP = sum(confusion_matrix[label]["USEFUL"] for label in other_rows)
    TN = sum(sum(value for cat, value in confusion_matrix[label].items() if cat != "USEFUL")
            for label in other_rows)

    # Create the aggregated confusion matrix
    aggregated_cm = {
        "USEFUL": {"USEFUL": TP, "OTHER": FN},
        "OTHER": {"USEFUL": FP, "OTHER": TN}
    }

    print("Aggregated Confusion Matrix:")
    pprint.pprint(aggregated_cm)

    # --- Step 2: Calculate Metrics ---
    # Total number of samples
    total = TP + FN + FP + TN

    # Overall accuracy: correctly predicted instances / total samples
    accuracy = (TP + TN) / total

    # Precision for USEFUL: TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # Recall for USEFUL: TP / (TP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # F1 Score for USEFUL: harmonic mean of precision and recall
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\nMetrics (Binary aggregation - USEFUL vs OTHER):")
    print(f"Overall Accuracy: {accuracy:.2%}")
    print(f"Precision for USEFUL: {precision:.2%}")
    print(f"Recall for USEFUL: {recall:.2%}")
    print(f"F1 Score for USEFUL: {f1:.2%}")

if __name__ == "__main__":
    main()
