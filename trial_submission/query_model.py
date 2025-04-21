
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
with open("API_KEY", "r") as f:
    api_key = f.read().strip()
with open("OPENROUTER_API_KEY", "r") as f:
    OPENROUTER_API_KEY = f.read().strip()
with open("DEEPSEEK_API_KEY", "r") as f:
    DEEPSEEK_API_KEY = f.read().strip()
with open("hf_token.txt", "r") as token_file:
    hf_token = token_file.read().strip()

# Load Qwen2.5-3B-Instruct model and tokenizer
deepinfra_models = ["meta-llama/Meta-Llama-3.1-8B-Instruct", 
                    "deepseek-ai/DeepSeek-V3-0324", 
                    "deepseek-ai/DeepSeek-V3", 
                    "deepseek-ai/DeepSeek-R1", 
                    "Qwen/Qwen2.5-72B-Instruct", 
                     "meta-llama/Llama-3.3-70B-Instruct", 
                     "meta-llama/Llama-3.2-3B-Instruct",
                    #  "Qwen/Qwen2.5-7B-Instruct",
                     "meta-llama/Meta-Llama-3.1-405B-Instruct"]

openrouter_models = ["qwen/qwq-32b:free", "deepseek/deepseek-r1:free"]
deepseek_models = ["deepseek-reasoner"]


def query_deepinfra(model_name, messages, temperature, max_new_tokens = 2048):
    # Assume openai>=1.0.0
    
    # Create an OpenAI client with your deepinfra token and endpoint
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

    return response, None, input_token_count, output_token_count

def query_openrouter(model_name, messages, temperature, max_new_tokens = 2048):

    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key= OPENROUTER_API_KEY,
    )

    completion = client.chat.completions.create(
    # extra_headers={
    #     "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    #     "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
    # },
    extra_body={},
    model=model_name,
    messages=messages,
    max_tokens=130000,
    temperature=temperature,
    )
    print("completion", completion)
    # print("completion.choices[0].message", completion.choices[0].message.keys())
    # print("completion.choices[0].message.content", completion.choices[0].message.content)
    # print("completion.choices[0].message.reasoning", completion.choices[0].message.reasoning)
    response = completion.choices[0].message.content
    reasoning = completion.choices[0].message.reasoning
    # print("response", response)
    input_token_count = completion.usage.prompt_tokens
    output_token_count = completion.usage.completion_tokens

    return response, reasoning, input_token_count, output_token_count

def query_deepseek(model_name, messages, temperature, max_new_tokens = 2048):
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model=model_name,
        messages=messages
    )

    reasoning_content = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content
    input_token_count = response.usage.prompt_tokens
    output_token_count = response.usage.completion_tokens

    return content, reasoning_content, input_token_count, output_token_count


def query_model(messages, tokenizer , model,  model_name, pipeline, temperature=0.7, max_new_tokens = 2048):
    """
    Queries the Qwen2.5-3B-Instruct model with a given prompt and counts input/output tokens.
    """
    if model_name in deepinfra_models:   
        print("query_deepinfra")
        return query_deepinfra(model_name, messages, temperature, max_new_tokens = max_new_tokens)
    if model_name in openrouter_models:
        print("query_openrouter")
        return query_openrouter(model_name, messages, temperature, max_new_tokens = max_new_tokens)
    if model_name in deepseek_models:
        print("query_deepseek")
        return query_deepseek(model_name, messages, temperature, max_new_tokens = max_new_tokens)
    if "meta-llama" in model_name:
        outputs = pipeline(
            messages,
            max_new_tokens=max_new_tokens,
            # temperature=temperature,
            do_sample=True
            # use_auth_token=hf_token
        )
        response = outputs[0]["generated_text"][-1]['content']
        print(response)
        return response, None, None, None
    # Convert messages to ChatML format and tokenize
    chatml_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True )
    inputs = tokenizer([chatml_input], return_tensors="pt").to("cuda")

    # print(inputs.input_ids)

    input_token_count = inputs.input_ids.shape[1]  # Count input tokens

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens = max_new_tokens, 
                                        do_sample=True, 
                                        # temperature=temperature
                                        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_token_count = sum(len(ids) for ids in generated_ids)  # Count output tokens

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"Input tokens: {input_token_count}, Output tokens: {output_token_count}")

    return response, None, input_token_count, output_token_count



# def extract_final_answer(messages, full_response):
#     """
#     Takes the model's full response and extracts only the final answer while counting tokens.
#     """
#     extraction_prompt = f"""
#     For this problem: {messages[1]["content"]}
#     The following is a response from an LLM that includes reasoning and analysis.

#     -----
#     {full_response}
#     -----

#     Extract only the final answer (without explanation, reasoning, or any additional text, no English: or Answer:).
    
#     ANSWER:
#     """

#     messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": extraction_prompt}
#     ]
    
#     extracted_answer, input_tokens, output_tokens = query_model(messages, tokenizer = tokenizer_2, model = model_2,  model_name = model_name_2, max_new_tokens = 100)
    
#     return extracted_answer, input_tokens, output_tokens
