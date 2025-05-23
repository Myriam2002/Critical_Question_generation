import json
from transformers import AutoTokenizer, LlamaForCausalLM, GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import logging
import tqdm
import re
import os
from query_model import query_model, deepinfra_models, deepseek_models, openrouter_models
from prompts import *
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline as hf_pipeline
)

logging.basicConfig()
logger = logging.getLogger()
FALLACY_THRESHOLD = 0.05


def output_cqs(model_name, text, prompt_obj: BasePrompt, model, tokenizer, pipeline, temperature, new_params, remove_instruction=False, schemes_list=None, print_prompt=False, fallacies_scores=None):
    
    # If it's a SchemechoosePrompt and schemes_list is provided, use the scheme-specific format
    if isinstance(prompt_obj, SchemechoosePrompt) and schemes_list:
        instruction = prompt_obj.format(intervention=text, schemes_list=schemes_list)
    elif isinstance(prompt_obj, LogicalFallaciesPrompt) and fallacies_scores is not None:
        # Use fallacy-based formatting
        instruction = prompt_obj.format(intervention=text, fallacies_scores=fallacies_scores)
    else:
        instruction = prompt_obj.format(intervention=text)

    # Print the prompt just once for verification
    if print_prompt:
        print("\n----- PROMPT START -----\n")
        print(instruction)
        print("\n----- PROMPT END -----\n")
        # Set to False so it only prints once
        print_prompt = False

    messages = [{"role": "system", "content": ""},
            {"role": "user", "content": instruction}]
    out, reasoning, input_token_count, output_token_count = query_model(messages, tokenizer=tokenizer, model=model, pipeline=pipeline, model_name=model_name, temperature=temperature)

    if remove_instruction:
        try:
            out = out.split('<|assistant|>')[1]
        except IndexError:
            out = out[len(instruction):]

    return out, reasoning, print_prompt

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def structure_output(whole_text):
    whole_text = extract_xml_answer(whole_text)
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
        new_cqs = re.split("\?\"", text+'end')
        if len(new_cqs) > 1:
            for cq in new_cqs[:-1]:
                valid.append(cq+'?\"')
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
            output.append({'id':i, 'cq':final[i]})
        return output
    if len(final) == 0 and len(not_valid) >=3:
        for i in [0, 1, 2]:
            output.append({'id':i, 'cq':not_valid[i] + "?\""})
        return output
    else:
        i = -1
        for i, cq in enumerate(final):
            output.append({'id':i, 'cq':final[i]})
        for x in range(i+1, 3):
            output.append({'id':i, 'cq':'Missing CQs'} )
  
        logger.warning('Missing CQs')
        return output


prompt_classes = {
    "zero_shot_with_instructions": ZeroShotWithInstructionsPrompt,
    "zero_shot_with_instructions2":ZeroShotWithInstructionsPrompt2,
    "zero_shot": ZeroShotPrompt,
    "few_shot":FewShotPrompt,
    "comprehensive_few_shot":ComprehensiveFewShotPrompt,
    "schema_prompt": SchemePrompt,
    "rl_prompt": RlPrompt,
    "schemechoose_prompt": SchemechoosePrompt,  
    "LogicalFallaciesPrompt": LogicalFallaciesPrompt,
}

def is_small_model(model_name):
    """Determine if a model is small enough to be run locally"""
    small_model_indicators = [
        "3B", "7B", "8B", "1.5B", "2B", "Qwen2.5-7B",
        "Meta-Llama-3.1-8B-Instruct", "Llama-3.2-3B"
    ]
    
    # Check if any of the small model indicators are in the model name
    return any(indicator in model_name for indicator in small_model_indicators)

def main():
    temperature = 0.1
    selected_prompt_names = [
        # "rl_prompt",    
        "schema_prompt"
        # "schemechoose_prompt"  # Change this to use your new prompt
        # "zero_shot", "zero_shot_with_instructions2", "few_shot", 
        # "comprehensive_few_shot", 
        # "LogicalFallaciesPrompt",
        ]
    
    tokenizer = AutoTokenizer.from_pretrained(
    "q3fer/distilbert-base-fallacy-classification", 
    use_fast=True
)
    
    fallacy_classifier = hf_pipeline(
        'text-classification',
        model='q3fer/distilbert-base-fallacy-classification',
        return_all_scores=True
    )
    models = [
        # 'meta-llama/Meta-Llama-3.1-405B-Instruct',
        # 'meta-llama/Llama-3.3-70B-Instruct',
        # 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'meta-llama/Llama-3.2-3B-Instruct',
        # 'Qwen/Qwen2.5-14B-Instruct', 
        
        # 'Qwen/Qwen2.5-72B-Instruct', 
        # 'Qwen/Qwen2.5-7B-Instruct',
        # 'deepseek-ai/DeepSeek-V3-0324', 
        # 'deepseek-reasoner'
    ]

    data_files = ["testing_dataset"]

    for data_file in data_files:
        with open(f'../data_splits/{data_file}.json') as f:
            data = json.load(f)
        
        for model_name in models:
            remove_instruction = False
            model = None
            tokenizer = None
            pipeline = None
            
            # Decision logic for API vs local pipeline
            use_api = (model_name in deepinfra_models or 
                      model_name in deepseek_models or 
                      model_name in openrouter_models or 
                      not is_small_model(model_name))
            
            if use_api:
                # For API models, keep model and tokenizer as None
                # query_model will handle the API call
                logger.info(f"Using API for {model_name}")
            else:
                # For small models, try to load locally via pipeline
                try:
                    logger.info(f"Loading {model_name} locally via pipeline")
                    pipeline = transformers.pipeline(
                        "text-generation",
                        model=model_name,
                        model_kwargs={"torch_dtype": torch.bfloat16},
                        device_map="auto",
                    )
                except Exception as e:
                    logger.warning(f"Failed to load {model_name} via pipeline: {e}")
                    logger.info(f"Attempting to load model directly")
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name, 
                            torch_dtype=torch.float16, 
                            device_map="auto", 
                            trust_remote_code=True
                        )
                    except Exception as e2:
                        logger.error(f"Failed to load {model_name} directly: {e2}")
                        logger.info(f"Falling back to API for {model_name}")
                        # Reset these to None so query_model will use API
                        model = None
                        tokenizer = None
            
            for selected_prompt_name in selected_prompt_names:
                if selected_prompt_name in prompt_classes:
                    prompt_obj = prompt_classes[selected_prompt_name]()
                else:
                    raise ValueError(f"Invalid prompt type: {selected_prompt_name}")
                    
                model_name_in_file = model_name.replace("/", "_")
                target_file_name = f'experiments_results/{data_file}_output_{selected_prompt_name}_{model_name_in_file}_t{temperature}.json'
                print("Starting", target_file_name)
                
                # If the file exists, load it to check for missing entries
                out = {}
                if os.path.exists(target_file_name):
                    with open(target_file_name, 'r') as f_out:
                        out = json.load(f_out)

                new_params = None
                logger.info(f'Processing with {model_name}')
                
                print_prompt_once = True
                for key, line in tqdm.tqdm(data.items()):
                    text = line['intervention']
                    intervention_id = line.get('intervention_id')
                    if intervention_id in out and "full_response" in out[intervention_id] and out[intervention_id]['cqs'] != "Missing CQs":
                        print(f"Skipping {intervention_id} as it already exists in the output file.")
                        continue
                    
                    # Get the schemes list if available
                    schemes_list = line.get('schemes', [])
                    fallacies_scores = None
                    if selected_prompt_name == 'LogicalFallaciesPrompt':
                        # Detect fallacies and filter by threshold
                        raw = fallacy_classifier(
                            text,
                            truncation=True,
                            max_length=512,
                        )[0]
                        scores = {item['label']: item['score'] for item in raw}
                        fallacies_scores = {name: scr for name, scr in scores.items() if scr >= FALLACY_THRESHOLD}


                    cqs, reasoning, print_prompt_once = output_cqs(
                        model_name, text, prompt_obj, model, tokenizer, pipeline, 
                        temperature, new_params, remove_instruction, schemes_list, print_prompt_once, fallacies_scores=fallacies_scores
                    )
                    line['cqs'] = structure_output(cqs)
                    line['full_response'] = cqs
                    line['reasoning'] = reasoning
                    out[line['intervention_id']] = line

                    # Save after each iteration to avoid losing progress
                    with open(target_file_name, 'w') as o:
                        json.dump(out, o, indent=4)

if __name__ == "__main__":
    main()
