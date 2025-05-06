
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import numpy
from collections import Counter
import sys
import argparse
import logging

from prompts_eval import *
from openai import OpenAI
import re
import os
from dotenv import load_dotenv
logger = logging.getLogger(__name__)

api_key = "geGMD3MAFq8QPX5W12k2PriDtXZT6SvR"
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")

def merge_dicts(dict1, dict2):
    """
    Recursively merge dict2 into dict1.
    If a key exists in both and both values are dictionaries, merge them.
    Otherwise, the value from dict2 overwrites the one in dict1.
    """
    for key, value in dict2.items():
        if key in dict1:
            if isinstance(dict1[key], dict) and isinstance(value, dict):
                merge_dicts(dict1[key], value)
            elif isinstance(dict1[key], list) and isinstance(value, list):
                dict1[key].extend(x for x in value if x not in dict1[key])
            else:
                dict1[key] = value
        else:
            dict1[key] = value
    return dict1

def merge_json_files(file1: str, file2: str, output_file: str):
    """
    Merge two JSON files and write the merged result to output_file.
    
    Parameters:
        file1 (str): Path to the first JSON file.
        file2 (str): Path to the second JSON file.
        output_file (str): Path for saving the merged JSON.
    
    Returns:
        The merged JSON object.
    """
    with open(file1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    
    with open(file2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    
    # Check if both JSON objects are of the same type
    if isinstance(data1, dict) and isinstance(data2, dict):
        merged_data = merge_dicts(data1, data2)
    elif isinstance(data1, list) and isinstance(data2, list):
        merged_data = data1 + data2
    else:
        raise ValueError("Both JSON files must contain data of the same type (both dicts or both lists).")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)
    
    return merged_data

def extract_label(response):
    """
    Extracts the label (USEFUL, UNHELPFUL, INVALID) from a given response.
    
    Parameters:
    response (str): The user's response which includes the label in the format "ANSWER: label"

    Returns:
    str: Extracted label or 'UNKNOWN' if no valid label is found.
    """
    match = re.search(r'ANSWER:\s*(USEFUL|UNHELPFUL|INVALID)', response, re.IGNORECASE)
    if not match:
        match = re.search(r'Label:\s*(USEFUL|UNHELPFUL|INVALID)', response, re.IGNORECASE)
    if not match:
        match = re.search(r':\s*(USEFUL|UNHELPFUL|INVALID)', response, re.IGNORECASE)
    if not match:
        match = re.search(r'\s*(USEFUL|UNHELPFUL|INVALID)', response, re.IGNORECASE)
    extracted = match.group(1).upper() if match else "UNKNOWN"
    logging.debug("Extracted label: %s from response: %s", extracted, response)
    return extracted


def evluate_answer_corr_with_llm(answer, intervention):
    instruction = ComprehensiveFewShotPrompt().format(intervention=intervention, cq=answer)
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": instruction}]
    full_output, reasoning, _, _ = query_deepinfra("meta-llama/Llama-3.3-70B-Instruct", messages, temperature=0.0)
    label = extract_label(full_output)
    logging.debug("LLM evaluation for answer '%s' on intervention '%s' returned label: %s", 
                  answer, intervention, label)
    return label.lower()

def query_deepinfra(model_name, messages, temperature, max_new_tokens=2048):
    openai = OpenAI(
        api_key=api_key,
        base_url="https://api.deepinfra.com/v1/openai",
    )

    chat_completion = openai.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature
    )

    response = chat_completion.choices[0].message.content
    input_token_count = chat_completion.usage.prompt_tokens
    output_token_count = chat_completion.usage.completion_tokens

    # logging.debug("Queried DeepInfra with model %s: prompt tokens=%d, completion tokens=%d", 
    #               model_name, input_token_count, output_token_count)
    logging.debug("Queried DeepInfra with model %s: response=%s", model_name, response)
    return response, None, input_token_count, output_token_count


def eval_func(threshold, golden_path, submission_path):

    #logger
    logging.basicConfig(filename='eval.log', level=logging.INFO)

    model = SentenceTransformer("stsb-mpnet-base-v2") 

    # load the whole dataset
    with open(golden_path,encoding="utf-8" ) as f:
        reference=json.load(f)

    with open(submission_path, encoding="utf-8") as f:
        new = json.load(f)

    # start the evaluation
    predicted_labels = []
    punctuations = []
    llm_labeled = 0

    for instance in new.keys(): # for each intervention
        if instance not in reference:
            logger.warning('Intervention '+instance+' is not in the golden set')
            continue
        punctuation = 0
        reference_set = [ref['cq'] for ref in reference[instance]['cqs']]
        if new[instance]['cqs'] != 'Missing CQs':
            cqs_check = [cq['cq'] for cq in new[instance]['cqs']]
            if len(cqs_check) != len(set(cqs_check)): # check the generated CQs are not repeated
                logger.warning('There are repeated CQs in '+instance)
            for i, line in enumerate(new[instance]['cqs']): # look into each question of the new cqs and find the most similar question in the references
                if line['cq'] == 'Missing CQs':
                    label = 'Missing CQ'
                    predicted_labels.append(label)
                    new[instance]['cqs'][i]['label'] = label
                    continue

                winner = None
               
                sentence_embedding = model.encode(line['cq'])
                reference_embedding = model.encode(reference_set)
                sims = model.similarity(sentence_embedding, reference_embedding).tolist()[0]
                    
               
                winner = np.argmax(sims)
                    # Save the matched reference information
                new[instance]['cqs'][i]['matched_reference'] = {
                    'index': int(winner),
                    'cq': reference[instance]['cqs'][winner]['cq'],
                    'label': reference[instance]['cqs'][winner]['label'],
                    'similarity': float(sims[winner])
                }
                # make sure the similarity of the winning reference sentence is at least 0.6
                if sims[winner] > threshold:
                    label = reference[instance]['cqs'][winner]['label']
                    if label == 'Useful':
                        punctuation += 1/3
                else: 
                    logging.debug("Similarity below threshold for '%s'. Using LLM fallback evaluation.", line['cq'])
                    print(f"Similarity below threshold for '{line['cq']}'. Using LLM fallback evaluation.")
                    label = evluate_answer_corr_with_llm(line['cq'], reference[instance]['intervention'])
                    logging.debug("LLM fallback label: %s", label)
                    print("LLM fallback label:", label)
                    if label == 'useful':
                        punctuation += 1/3

                    # label = 'not_able_to_evaluate'
                    llm_labeled += 1
                    label = f"LLM_{label}"

                predicted_labels.append(label)
                new[instance]['cqs'][i]['label'] = label
        else:
            predicted_labels.extend(['Missing CQ', 'Missing CQ', 'Missing CQ']) # this should disapear with a proper prompt that makes sure there are always 3 questions

        punctuations.append(punctuation)

    # metrics
    print('Distribution of the labels:', Counter(predicted_labels))
    print('Distribution of the intervention punctuation:', Counter(punctuations))
    if punctuations:
        x = sum(punctuations) / len(punctuations)
        print('Overall punctuation', x )
    else:
        x = 0
        print('Overall punctuation: N/A (no data)')
    new["Overall_stats"] = {
        "Distribution of the labels": Counter(predicted_labels),
        "Distribution of the intervention punctuation": Counter(punctuations),
        "Overall punctuation": x,
        "LLM labeled": (llm_labeled/len(predicted_labels)) * 100

    }
    return new



# temperature = 0.8
# selected_prompt_names = [
#     "rl_prompt"
#     # "schema_prompt"
#     # "zero_shot", "zero_shot_with_instructions2", "few_shot", "comprehensive_few_shot"
#     ]

# models = ['deepseek-ai/DeepSeek-V3-0324', 'deepseek-reasoner', 'Qwen/Qwen2.5-14B-Instruct', 'Qwen/Qwen2.5-72B-Instruct', 'Qwen/Qwen2.5-7B-Instruct', 'meta-llama/Llama-3.2-3B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct'] 
# models= ['RL']
# models = ["meta-llama/Llama-3.1-8B-Instruct"]
# data_files = ["sample", "validation"]
# golden_path = "D:\\My_working_area\\Masters\\Semester 2\\NLP804\\Project\\Critical_Question_generation\\data_splits\\testing_dataset.json"

# # for model_name in models:
#     # model_name_in_file = model_name.replace("/", "_") 
#     # for selected_prompt_name in selected_prompt_names:
# target_file = "D:\\My_working_area\\Masters\\Semester 2\\NLP804\\Project\\Critical_Question_generation\\trial_submission\\Mariam\\results.json"
# print("Starting", target_file)
# result_file_name = "Evaluation_results_trial.json"
# result  = eval_func(threshold = 0.6, golden_path=  golden_path, submission_path = target_file)
# with open(result_file_name, 'w') as o:
#     json.dump(result, o, indent=4)