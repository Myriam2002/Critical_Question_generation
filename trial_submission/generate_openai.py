import json
import openai
import logging
import tqdm
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Ensure OpenAI API key is set
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OpenAI API key is missing! Set OPENAI_API_KEY in the .env file.")
    exit(1)

# Initialize OpenAI client with the correct method
client = openai.OpenAI(api_key=api_key)

def output_cqs_openai(model_name, text, prefix):
    """
    Generates critical questions using OpenAI API instead of Llama.
    """
    instruction = prefix.format(**{'intervention': text})

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an AI assistant that generates critical questions."},
                {"role": "user", "content": instruction}
            ],
            temperature=0.7,
            max_tokens=200
        )

        # Extract generated text
        output = response.choices[0].message.content.strip()
        return output
    except Exception as e:
        logger.error(f"OpenAI API Error: {e}")
        return "Missing CQs"

def structure_output(whole_text):
    """
    Cleans and extracts valid critical questions from OpenAI output.
    """
    cqs_list = whole_text.split("\n")
    final = []
    valid = []
    not_valid = []

    # Validate questions based on regex pattern
    for cq in cqs_list:
        if re.match(r'.*\?(")?( )?(\([a-zA-Z0-9\.\'\-,\? ]*\))?([a-zA-Z \.,\"\']*)?(\")?$', cq):
            valid.append(cq)
        else:
            not_valid.append(cq)

    still_not_valid = []
    for text in not_valid:
        new_cqs = re.split("\?\"", text + 'end')
        if len(new_cqs) > 1:
            for cq in new_cqs[:-1]:
                valid.append(cq + '?"')
        else:
            still_not_valid.append(text)

    for i, cq in enumerate(valid):
        occurrence = re.search(r'[A-Z]', cq)
        if occurrence:
            final.append(cq[occurrence.start():])
        else:
            continue

    # Select first 3 valid questions
    output = []
    if len(final) >= 3:
        for i in range(3):
            output.append({'id': i, 'cq': final[i]})
        return output
    else:
        logger.warning('Missing CQs')
        return 'Missing CQs'

def main():
    """
    Main function to process interventions using OpenAI API.
    """
    prefixes = ["""Suggest 3 critical questions that should be raised before accepting the arguments in this text:
                
                "{intervention}"
                
                Give one question per line. Make the questions simple, and do not give any explanation regarding why the question is relevant."""]
    
    # Load input data
    with open('data_splits/sample.json') as f:
        data = json.load(f)

    # Use OpenAI GPT-4o model instead of Llama
    model_name = "gpt-4o-mini"

    out = {}
    logger.info(f"Using model: {model_name}")

    for key, line in tqdm.tqdm(data.items()):
        for prefix in prefixes:
            text = line['intervention']
            cqs = output_cqs_openai(model_name, text, prefix)
            line['cqs'] = structure_output(cqs)
            out[line['intervention_id']] = line

    # Save results
    with open('trial_submission/output_gpt.json', 'w') as o:
        json.dump(out, o, indent=4)

if __name__ == "__main__":
    main()
