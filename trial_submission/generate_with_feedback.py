import json
from transformers import AutoTokenizer, LlamaForCausalLM, GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import logging
import tqdm
import re
import os
from query_model import query_model, deepinfra_models, openrouter_models, deepseek_models
from generate_eval import output_label
from prompts import *
import prompts_eval

logging.basicConfig()
logger = logging.getLogger()

with open("hf_token.txt", "r") as token_file:
    hf_token = token_file.read().strip()

def output_cqs(model_name, text, prompt_obj: BasePrompt, model, tokenizer, pipeline, temperature, new_params, feedback = []):
    instruction = prompt_obj.format(intervention=text)

    messages = [{"role": "system", "content": ""},
                {"role": "user", "content": instruction}]
    if feedback:
        for item in feedback:
            messages.append(item)

    out, reasoning, input_token_count, output_token_count = query_model(
        messages,
        tokenizer=tokenizer,
        model=model,
        pipeline=pipeline,
        model_name=model_name,
        temperature=temperature
    )
   
    return out, reasoning

def structure_output(whole_text):
    cqs_list = whole_text.split('\n')
    final = []
    valid = []
    not_valid = []
    for cq in cqs_list:
        if re.match('.*\?(\")?( )?(\([a-zA-Z0-9\.\'\-,\? ]*\))?([a-zA-Z \.,\"\']*)?(\")?$', cq):
            valid.append(cq)
        else:
            not_valid.append(cq)
    still_not_valid = []
    for text in not_valid:
        new_cqs = re.split("\?\"", text + 'end')
        if len(new_cqs) > 1:
            for cq in new_cqs[:-1]:
                valid.append(cq + "?\"")
        else:
            still_not_valid.append(text)
    for i, cq in enumerate(valid):
        occurrence = re.search(r'[A-Za-z]', cq)
        if occurrence:
            final.append(cq[occurrence.start():])
        else:
            continue
    output = []
    if len(final) >= 3:
        for i in [0, 1, 2]:
            output.append({'id': i, 'cq': final[i]})
        return output
    else:
        i = -1
        for i, cq in enumerate(final):
            output.append({'id': i, 'cq': final[i]})
        for x in range(i+1, 3):
            output.append({'id': x, 'cq': 'Missing CQs'})
        logger.warning('Missing CQs')
        return output

prompt_classes = {
    "zero_shot_with_instructions": ZeroShotWithInstructionsPrompt,
    "zero_shot_with_instructions2": ZeroShotWithInstructionsPrompt2,
    "zero_shot": ZeroShotPrompt,
    "few_shot": FewShotPrompt,
    "comprehensive_few_shot": ComprehensiveFewShotPrompt,
    "schema_prompt": SchemePrompt
}

def main():
    temperature = 0.1
    selected_prompt_names = [
        
        "zero_shot", 
        # "zero_shot_with_instructions2", 
        # "few_shot", "comprehensive_few_shot", "schema_prompt",
    ]
    models = ["Qwen/Qwen2.5-72B-Instruct"]
    data_files = ["testing_dataset"]

    for data_file in data_files:
        with open(f'../data_splits/{data_file}.json') as f:
            data = json.load(f)
        out = {}
        for model_name in models:
            remove_instruction = False
            model = ""
            tokenizer = ""
            pipeline = ""
            if model_name not in deepinfra_models and "meta-llama" in model_name:
                pipeline = transformers.pipeline(
                    "text-generation",
                    model=model_name,
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    device_map="auto",
                    # use_auth_token=hf_token
                )
            elif model_name not in deepinfra_models and model_name not in openrouter_models and model_name not in deepseek_models:     
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
            for selected_prompt_name in selected_prompt_names:
                if selected_prompt_name in prompt_classes:
                    prompt_obj = prompt_classes[selected_prompt_name]()
                else:
                    raise ValueError(f"Invalid prompt type: {selected_prompt_name}")
                    
                model_name_in_file = model_name.replace("/", "_")
                target_file_name = f'experiments_results/with_feedback_{data_file}_output_{selected_prompt_name}_{model_name_in_file}_t{temperature}.json'
                print("Starting", target_file_name)
                # If the file exists, load it to check for missing entries.
                if os.path.exists(target_file_name):
                    with open(target_file_name, 'r') as f_out:
                        out = json.load(f_out)
                else:
                    out = {}

                new_params = None
                logger.info(model_name)
                logger.info('Loaded ' + model_name)

                for key, line in tqdm.tqdm(data.items()):
                    text = line['intervention']
                    intervention_id = line.get('intervention_id')
                    if intervention_id in out and "iterations" in out[intervention_id]:
                        print(f"Skipping {intervention_id} as it already exists in the output file.")
                        continue

                    iteration_history = []
                    attempt = 0
                    feedback = []
                    while attempt < 5:
                        attempt += 1
                        cqs, reasoning = output_cqs(model_name, text, prompt_obj, model, tokenizer, pipeline, temperature, new_params, feedback)
                        structured_cqs = structure_output(cqs)
                        for candidate in structured_cqs:
                            if candidate['cq'] == 'Missing CQs':
                                candidate['label'] = 'Missing CQs'
                                candidate['label_model_output'] = 'Missing CQs'
                                candidate['label_reasoning'] = 'Missing CQs'
                            else:
                                eval_prompt_obj = prompts_eval.ComprehensiveFewShotPrompt()
                                label, model_output, label_reasoning = output_label(
                                    "meta-llama/Llama-3.3-70B-Instruct", text, candidate['cq'], eval_prompt_obj, model, tokenizer, pipeline, temperature=0.0
                                )
                                candidate['label'] = label
                                candidate['label_model_output'] = model_output
                                candidate['label_reasoning'] = label_reasoning
                        feedback_candidates = [candidate['cq'] for candidate in structured_cqs]
                        candidate_responses = [f"For {candidate['cq']}: {candidate['label_model_output']}" for candidate in structured_cqs]
                        feedback= [{"role": "assistant", "content": f"{feedback_candidates}"},
                                   {"role": "user", "content": f"{candidate_responses} \n Now please provide USEFUL questions for the intervention: {text}.\nProvide one question per line. Do not include any special characters or numbering except for the question mark."}]

                        iteration_history.append({
                            "attempt": attempt,
                            "candidates": structured_cqs,
                            "full_response": cqs,
                            "reasoning": reasoning
                        })

                        if all(candidate.get('label').lower() == "useful" for candidate in structured_cqs):
                            break
                        else:
                            print(f"Not all candidate questions labeled as useful for intervention {intervention_id}. Retrying... (attempt {attempt})")
                    
                    # Record the iteration history along with final outputs in the JSON
                    line['iterations'] = iteration_history
                    line['cqs'] = structured_cqs
                    line['full_response'] = cqs
                    line['reasoning'] = reasoning
                    out[line['intervention_id']] = line

                    with open(target_file_name, 'w') as o:
                        json.dump(out, o, indent=4)

if __name__ == "__main__":
    main()
