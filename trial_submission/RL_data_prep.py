import re
import logging
from datasets import load_dataset, Dataset
from prompts_eval import *
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import datetime

# Generate a timestamp and create a log filename with it.
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"reward_functions_{timestamp}.log"

# Setup logging configuration to log to a file with the timestamp in its name.
logging.basicConfig(
    filename=log_filename,  # Log file with timestamp.
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

model_eval = SentenceTransformer("stsb-mpnet-base-v2")

with open("API_KEY", "r") as f:
    api_key = f.read().strip()

# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# template = (
#     "Suggest 3 critical questions that should be raised before accepting the arguments in this text:\n\n"
#     "\"{intervention}\"\n\n"
#     "Give one question per line. Make the questions simple, and do not provide any explanation regarding why the question is relevant in the answer tag. "
#     "Do not include any special characters or numbering except for the question mark in the answer tag."
# )

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


def extract_answer(example):
    """
    Extracts the 'cqs' (critical questions) from the example and formats each question with its label.
    Each line in the answer will be in the form:
    <question> (Label: <label>)
    """
    cqs = example.get("cqs", [])
    answer_lines = []
    for cq in cqs:
        question = cq.get("cq", "")
        label = cq.get("label", "")
        answer_lines.append({"cq": question, "label": label, "intervention": example["intervention"]})
    return answer_lines

def get_critical_questions_dataset(split="train"):
    file_path = "../data_splits/training_dataset.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # If data is a dictionary, convert it to a list of examples.
    if isinstance(data, dict):
        examples = list(data.values())
    else:
        examples = data
    
    # Map each example to include prompt and answer fields.
    for example in examples:
        example["prompt"] = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': template.format(intervention=example['intervention'])}
        ]
        example["answer"] = extract_answer(example)
    
    return examples

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
                valid.append(cq + '?\"')
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
            output.append({'id':i, 'cq':final[i]})
        for x in range(i+1, 3):
            output.append({'id':i, 'cq':'Missing CQs'} )
  
        return output

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

def evluate_answer_corr_with_llm(answer, intervention):
    instruction = ComprehensiveFewShotPrompt().format(intervention=intervention, cq=answer)
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": instruction}]
    full_output, reasoning, _, _ = query_deepinfra("meta-llama/Llama-3.3-70B-Instruct", messages, temperature=0.0)
    label = extract_label(full_output)
    logging.debug("LLM evaluation for answer '%s' on intervention '%s' returned label: %s", 
                  answer, intervention, label)
    return label.lower()
    
def evaluate_answer_corr(golden_answers, answer):
    punctuation = 0
    reference_set = [ref['cq'] for ref in golden_answers[0]]
    for i, line in enumerate(answer):  # Evaluate each critical question
        logging.debug("Evaluating answer '%s' against reference set.", line['cq'])
        if line['cq'] == 'Missing CQs':
            continue
        sentence_embedding = model_eval.encode(line['cq'])
        reference_embedding = model_eval.encode(reference_set)
        sims = model_eval.similarity(sentence_embedding, reference_embedding).tolist()[0]
        winner = np.argmax(sims)
        logging.debug("Similarity scores for '%s': %s", line['cq'], sims)
        if sims[winner] > 0.6:
            label = golden_answers[0][winner]['label'].lower()
            logging.debug("Matched reference label: %s with similarity %.3f", label, sims[winner])
            if label == 'useful':
                punctuation += 1/3
        else:
            logging.debug("Similarity below threshold for '%s'. Using LLM fallback evaluation.", line['cq'])
            label = evluate_answer_corr_with_llm(line['cq'], golden_answers[0][0]['intervention'])
            logging.debug("LLM fallback label: %s", label)
            if label == 'useful':
                punctuation += 1/3
    return punctuation

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    logging.debug("Correctness reward function called with %d completions.", len(completions))
    responses = [completion[0]['content'] for completion in completions]
    for response in responses:
        logging.debug("Response in correctness_reward_func: %s", response)
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards = []
    for response in extracted_responses:
        logging.debug("Processing response in correctness_reward_func: %s", response)
        structured = structure_output(response)
        logging.debug("Structured output: %s", structured)
        if structured == 'Missing CQs' or not isinstance(structured, list) or len(structured) < 3:
            logging.info("Response does not contain enough CQs. Assigning reward 0.0")
            rewards.append(0.0)
        else:
            score = evaluate_answer_corr(answer, structured)
            computed_reward = (score * 3)/2
            # computed_reward = score
            logging.info("Computed correctness reward: score=%.3f, reward=%.3f", score, computed_reward)
            rewards.append(computed_reward)
    return rewards

def structure_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards = []
    for response in extracted_responses:
        structured = structure_output(response)
        reward = 0
        for q in structured:
            if q["cq"] != 'Missing CQs':
                reward+=1
        
        if reward<3:
            logging.info("Structured reward: Missing CQs detected. Reward 0.0")
            rewards.append(0.0)
        else:
            logging.info("Structured reward: Valid CQs found. Reward 0.5")
            rewards.append(0.5)
    return rewards

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.[\s\S]*\n</reasoning>\n<answer>\n.[\s\S]*\n</answer>$"
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for r in responses:
        if re.match(pattern, r):
            logging.debug("Strict format matched for response.")
            rewards.append(0.5)
        else:
            logging.debug("Strict format not matched for response.")
            rewards.append(0.0)
    return rewards

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for r in responses:
        if re.match(pattern, r, flags=re.DOTALL):
            logging.debug("Soft format matched for response.")
            rewards.append(0.5)
        else:
            logging.debug("Soft format not matched for response.")
            rewards.append(0.0)
    return rewards

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        # count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        # count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for c in contents:
        xml_reward = count_xml(c)
        logging.debug("XML count reward for response: %.3f", xml_reward)
        rewards.append(xml_reward)
    return rewards
