from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from generate import structure_output

import json
from transformers import AutoTokenizer, LlamaForCausalLM, GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import logging
import tqdm
import re
import os
from query_model import query_model, deepinfra_models, openrouter_models,deepseek_models
from prompts import *

logging.basicConfig()
logger = logging.getLogger()

with open("hf_token.txt", "r") as token_file:
    hf_token = token_file.read().strip()

max_seq_length = 4084 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-3B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def call_gpro_trained_model(messages, temperature=0.1):

    text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)

    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature = temperature,
        top_p = 0.95,
        max_tokens = 1024,
    )
    output = model.fast_generate(
        text,
        sampling_params = sampling_params,
        lora_request = model.load_lora("grpo_saved_lora_trial_2"),
    )[0].outputs[0].text

    return output

def output_cqs(text, prompt_obj: BasePrompt, temperature):

    instruction = prompt_obj.format(intervention=text)

    # inputs = tokenizer(instruction, return_tensors="pt")
    # inputs = inputs.to('cuda')

    # if new_params:
    #     outputs = model.generate(**inputs, **new_params) 
    # else:
    #     outputs = model.generate(**inputs)
    messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': instruction}
            ]
    out = call_gpro_trained_model(messages, temperature)


    return out

prompt_classes = {
    "zero_shot_with_instructions": ZeroShotWithInstructionsPrompt,
    "zero_shot_with_instructions2":ZeroShotWithInstructionsPrompt2,
    "zero_shot": ZeroShotPrompt,
    "few_shot":FewShotPrompt,
    "comprehensive_few_shot":ComprehensiveFewShotPrompt
}

def run_all_testing():
    model_name_in_file = "RL"
    temperature = 0.8
    with open(f'../data_splits/testing_dataset.json') as f:
        data=json.load(f)
    selected_prompt_names = [
        "zero_shot",
     "zero_shot_with_instructions2", "few_shot", 
    "comprehensive_few_shot"
    ]
    for selected_prompt_name in selected_prompt_names:
        if selected_prompt_name in prompt_classes:
            prompt_obj = prompt_classes[selected_prompt_name]()
        else:
            raise ValueError(f"Invalid prompt type: {selected_prompt_name}")
        target_file_name = f'experiments_results/testing_output_{selected_prompt_name}_{model_name_in_file}_t{temperature}.json'
        print("Starting", target_file_name)
        # If the file exists, load it to check for missing entries.
        if os.path.exists(target_file_name):
            with open(target_file_name, 'r') as f_out:
                out = json.load(f_out)
        else:
            out = {}

        new_params = None
        # generation_config = GenerationConfig.from_pretrained(model_name)
        # logger.info(generation_config)
        


        for key,line in tqdm.tqdm(data.items()):
            text = line['intervention']
            intervention_id = line.get('intervention_id')
            if intervention_id in out and "full_response" in out[intervention_id] and out[intervention_id]['cqs'] != "Missing CQs":
                print(f"Skipping {intervention_id} as it already exists in the output file.")
                continue
            cqs= output_cqs( text, prompt_obj, temperature)
            line['cqs'] = structure_output(cqs)
            line['full_response'] = cqs
            out[line['intervention_id']]=line


        with open(target_file_name, 'w') as o:
            json.dump(out, o, indent=4)

if __name__ == "__main__":
    template = (
        "Suggest 3 critical questions that should be raised before accepting the arguments in this text:\n\n"
        "\"{intervention}\"\n\n"
        "Give one question per line. Make the questions simple, and do not provide any explanation regarding why the question is relevant. "
        "Do not include any special characters or numbering except for the question mark."
    )

    intervention = "MT: \"Claire\u2019s absolutely right about that\nBut then the problem is that that form of capitalism wasn\u2019t generating sufficient surpluses\nAnd so therefore where did the money flow\nIt didn\u2019t flow into those industrial activities\n because in the developed world that wasn\u2019t making enough money\"",

    messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': template.format(intervention=intervention)}
            ]

    print(call_gpro_trained_model(messages))

    # run_all_testing()