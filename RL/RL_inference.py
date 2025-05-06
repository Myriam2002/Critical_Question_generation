from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from generate import structure_output
from huggingface_hub import snapshot_download
import json
from transformers import AutoTokenizer, LlamaForCausalLM, GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import logging
import tqdm
import re
import os

from prompts import *

logging.basicConfig()
logger = logging.getLogger()

with open("hf_token.txt", "r") as token_file:
    hf_token = token_file.read().strip()


repo_id = "samahadhoud/critical_questions_generation_llama_lora_RL_fintuned_3epoch"

max_seq_length = 8084 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/meta-Llama-3.1-8B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
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


def call_gpro_trained_model(messages, temperature=0.8):

    lora_path = snapshot_download(repo_id=repo_id)

    text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)

    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature = temperature,
        top_p = 0.95,
        max_tokens = 2000,
    )
    output = model.fast_generate(
        text,
        sampling_params = sampling_params,
        lora_request = model.load_lora(lora_path),
    )[0].outputs[0].text

    return output

template = (
    "You are a teacher in a critical thinking class. Your goal is to help students learn to critically evaluate argumentative texts. "
    "To do this, you need to generate critical questions that challenge the validity of the arguments presented. "
    "A question is considered USEFUL if it makes the reader reflect on the text in a way that could potentially diminish its perceived validity. "
    "Avoid questions that are common sense, reading-comprehension, too general, or that introduce new concepts not present in the text.\n\n"
    "Guidelines:\n"
    "1. USEFUL QUESTION:\n"
    "   - Should be raised before accepting the arguments in this text.\n"
    "   - Challenges the text’s argument in a meaningful way.\n"
    "   - Prompts critical reflection that can weaken the argument’s validity if answered.\n"
    "   - Focuses on details already present in the text without introducing external ideas.\n\n"
    "2. UNHELPFUL QUESTION:\n"
    "   - Although related to the text, it asks about aspects that are either common sense, well-known facts, or too complicated.\n\n"
    "3. INVALID QUESTION:\n"
    "   - Unrelated to the text or introduces new concepts.\n"
    "   - Uses vague language or fails to challenge the argument’s core reasoning.\n\n"
    "Now, using the guidelines above, generate three USEFUL critical questions for the following text:\n\n"
    "TEXT:\n\"{intervention}\"\n\n"
    "Let’s think step by step, Respond in the following format:\n"
    "<reasoning>\nYour reasoning here.\n</reasoning>\n<answer>\nProvide three questions one question per line. Do not include any special characters or numbering except for the question mark.\n</answer>"
)
def output_cqs(text, prompt_obj: BasePrompt, temperature):

    instruction = template.format(intervention=text)

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

    cqs = extract_xml_answer(out) 
    return out, cqs

prompt_classes = {
    "zero_shot_with_instructions": ZeroShotWithInstructionsPrompt,
    "zero_shot_with_instructions2":ZeroShotWithInstructionsPrompt2,
    "zero_shot": ZeroShotPrompt,
    "few_shot":FewShotPrompt,
    "comprehensive_few_shot":ComprehensiveFewShotPrompt,
    "rl_prompt": RlPrompt,
}

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def run_all_testing():
    model_name_in_file = "RL"
    temperature = 0.8
    # with open(f'../data_splits/testing_dataset.json') as f:
    with open(f'../data_splits/testing_dataset.json') as f:
        data=json.load(f)
    selected_prompt_names = [
    #     "zero_shot",
    #  "zero_shot_with_instructions2", "few_shot", 
    # "comprehensive_few_shot",
    "rl_prompt",
    ]
    for selected_prompt_name in selected_prompt_names:
        if selected_prompt_name in prompt_classes:
            prompt_obj = prompt_classes[selected_prompt_name]()
        else:
            raise ValueError(f"Invalid prompt type: {selected_prompt_name}")
        target_file_name = f'RL_experiments/testing_dataset_output_{selected_prompt_name}_{model_name_in_file}_llama_trial_1_7epoch_t{temperature}.json'
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
            while True:
                model_out, cqs= output_cqs( text, prompt_obj, temperature)
                cqs_struct = structure_output(cqs)
                one_missing = False
                for cq in cqs_struct:
                    if cq["cq"] == "Missing CQs":
                        one_missing = True
                        break
                if not one_missing:
                    break
            line['cqs'] = cqs_struct
            line['full_response'] = model_out
            out[line['intervention_id']]=line


        with open(target_file_name, 'w') as o:
            json.dump(out, o, indent=4)

if __name__ == "__main__":
    template = (
    "You are a teacher in a critical thinking class. Your goal is to help students learn to critically evaluate argumentative texts. "
    "To do this, you need to generate critical questions that challenge the validity of the arguments presented. "
    "A question is considered USEFUL if it makes the reader reflect on the text in a way that could potentially diminish its perceived validity. "
    "Avoid questions that are common sense, reading-comprehension, too general, or that introduce new concepts not present in the text.\n\n"
    "Guidelines:\n"
    "1. USEFUL QUESTION:\n"
    "   - Should be raised before accepting the arguments in this text.\n"
    "   - Challenges the text’s argument in a meaningful way.\n"
    "   - Prompts critical reflection that can weaken the argument’s validity if answered.\n"
    "   - Focuses on details already present in the text without introducing external ideas.\n\n"
    "2. UNHELPFUL QUESTION:\n"
    "   - Although related to the text, it asks about aspects that are either common sense, well-known facts, or too complicated.\n\n"
    "3. INVALID QUESTION:\n"
    "   - Unrelated to the text or introduces new concepts.\n"
    "   - Uses vague language or fails to challenge the argument’s core reasoning.\n\n"
    "Now, using the guidelines above, generate three USEFUL critical questions for the following text:\n\n"
    "TEXT:\n\"{intervention}\"\n\n"
    "Respond in the following format:\n"
    "<reasoning>\nYour reasoning here.\n</reasoning>\n<answer>\nProvide three questions one question per line. Do not include any special characters or numbering except for the question mark.\n</answer>"
    )

    run_all_testing()