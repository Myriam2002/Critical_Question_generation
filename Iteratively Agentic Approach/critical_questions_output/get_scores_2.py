import json
import os
import csv     # ← add this
from collections import Counter

golden_path   = "D:\\My_working_area\\Masters\\Semester 2\\NLP804\\Project\\Critical_Question_generation\\data_splits\\testing_dataset.json"
txt_file  = "stats/results.txt"
csv_file  = "stats/results.csv"
summary_file = "stats/model_summary.csv"


from collections import Counter, defaultdict
with open(txt_file, 'w') as f:
    f.write("=== Results ===\n")

# pattern helpers
PREFIX = 'testing_dataset_output_'
SUFFIX = '_eval_similarity_0.6.json'

# load reference
with open(golden_path) as f:
    reference = json.load(f)

# prepare results container
results = []

selected_prompt_names = [
    "zero_shot_with_instructions",
    "zero_shot_with_instructions2",
    "rl_prompt",
    "schema_prompt",
    "zero_shot",
    "few_shot",
    "comprehensive_few_shot"
]

matched_prompt_names = {
    "rl_prompt": "cot",
    "zero_shot_with_instructions2": "Minimal",
    "few_shot": "FewShot",
    "comprehensive_few_shot": "few_shot_full_cot",
}


results = []

for fname in os.listdir('.'):
    if not (fname.startswith(PREFIX) and fname.endswith(SUFFIX)) :
        continue

    core = fname[len(PREFIX):-len(SUFFIX)]
    # core → "{prompt}_{model}_{runid}_t{temp}"

    # 1) pull off the temperature
    # try:
    #     before_t, temp_str = core.rsplit('_t', 1)
    #     temperature = float(temp_str)
    # except ValueError:
    #     print(f"Bad filename (no _t): {fname}")
    #     continue

    # 2) identify which prompt it is by comparing to your list
    for p in selected_prompt_names:
        if core.startswith(p + '_'):
            prompt_name = p
            if p in matched_prompt_names:
                prompt_name = matched_prompt_names[p]
            remainder  = before_t[len(p) + 1:]  # drop "p_"
            break
    else:
        print(f"Unknown prompt in {fname}")
        continue

    # 3) now remainder → "{model}_{runid}"
    # print(fname)
    # print(prompt_name)
    # print(remainder)
    # model_name, _ = remainder.split('_', 1)
    model_name = remainder

    print("Starting", fname)
    with open(fname) as f:
        data = json.load(f)

    # only process instances that are in your golden set
    for inst in reference:
        if inst not in data:
            print(f"not Processing {fname} as it is in the reference set.")
            print(inst)
            continue
    # if not any(inst in file for inst in reference):
    #     print(f"Skipping {file} as it is not in the reference set.")
    #     continue



    punctuations            = []
    punctuations_with_llm   = []
    llm_labeled             = []
    not_llm_labeled         = []
    all_labeled            = []

    for instance, info in data.items():
        if instance == "Overall_stats" or instance not in reference:
            continue

        p, p_llm = 0.0, 0.0
        for line in info['cqs']:
            if not isinstance(line, dict):
                continue
            lbl = line.get('label')
            if not lbl or lbl == 'Missing CQs':
                continue
            all_labeled.append(lbl)
            if lbl.startswith("LLM_"):
                if lbl == "LLM_useful":
                    p_llm += 1/3
                llm_labeled.append(lbl)
            elif lbl == "Useful":
                p     += 1/3
                p_llm += 1/3
                not_llm_labeled.append(lbl)
            else:
                not_llm_labeled.append(lbl)

        punctuations.append(p)
        punctuations_with_llm.append(p_llm)

    # compute aggregates
    overall_p     = sum(punctuations)          / len(punctuations)
    overall_p_llm = sum(punctuations_with_llm) / len(punctuations_with_llm)
    # print(all_labeled)

    overall_llm_labeled_percentage = (len(llm_labeled) / max(len(all_labeled), 1)) * 100

    # # print & append to your .txt as before
    # print('Distribution of the labels:', Counter(llm_labeled))
    # print('Distribution of the intervention punctuation:', Counter(punctuations))
    # print('Overall punctuation', overall_p)
    # print('Overall punctuation with llm', overall_p_llm)
    # print('Overall llm labeled %', overall_llm_labeled_percentage)

        # write to your human‑readable txt
    with open(txt_file, 'a') as outf:
        outf.write(f"{fname}:\n")
        outf.write(f"  ° prompt: {prompt_name}\n")
        outf.write(f"  ° model:  {model_name}\n")
        outf.write(f"  ° temp:   {temperature}\n")
        outf.write(f"  ° overall punctuation:           {overall_p}\n")
        outf.write(f"  ° overall punctuation with llm:  {overall_p_llm}\n")
        outf.write(f"  ° overall llm labeled %:         {overall_llm_labeled_percentage}\n")
        outf.write(f"  ° label distro:                  {Counter(llm_labeled)}\n\n")

    for result in results:
        if result['prompt'] == prompt_name and \
              result['model'] == model_name and \
                result['temperature'] == temperature:
            break
    # collect for raw CSV
    results.append({
        "fname": fname,
        'prompt': prompt_name,
        'model': model_name,
        'temperature': temperature,
        'overall_punctuation': overall_p,
        'overall_punctuation_with_llm': overall_p_llm,
        'overall_llm_labeled_percentage': overall_llm_labeled_percentage
    })

# 1) write the raw CSV
with open(csv_file, 'w', newline='') as cf:
    fieldnames = ["fname", 'prompt','model','temperature',
                  'overall_punctuation',
                  'overall_punctuation_with_llm',
                  'overall_llm_labeled_percentage']
    writer = csv.DictWriter(cf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

# 2) group by model and write a summary CSV
grouped = defaultdict(list)
for row in results:
    grouped[row['model']].append(row)

with open(summary_file, 'w', newline='') as sf:
    fieldnames = ['model',
                  'runs',
                  'mean_punctuation',
                  'mean_punctuation_with_llm',
                  'mean_llm_labeled_percentage']
    writer = csv.DictWriter(sf, fieldnames=fieldnames)
    writer.writeheader()
    for model, rows in grouped.items():
        n = len(rows)
        mean_p    = sum(r['overall_punctuation'] for r in rows)          / n
        mean_p_llm= sum(r['overall_punctuation_with_llm'] for r in rows) / n
        mean_pct  = sum(r['overall_llm_labeled_percentage'] for r in rows)      / n
        writer.writerow({
            'model': model,
            'runs': n,
            'mean_punctuation': mean_p,
            'mean_punctuation_with_llm': mean_p_llm,
            'mean_llm_labeled_percentage': mean_pct
        })

print(f"✔ Raw results → {csv_file}")
print(f"✔ Per‑model summary → {summary_file}")

import pandas as pd
# 1) make a DataFrame
metrics = [
    'overall_punctuation',
    'overall_punctuation_with_llm',
    'overall_llm_labeled_percentage'
]

df = pd.DataFrame(results)   # your raw list of dicts

# pivot + flatten exactly as in the last snippet
matrix = df.pivot_table(
    index=['model','temperature'],
    columns='prompt',
    values=metrics
).swaplevel(0,1, axis=1).sort_index(axis=1, level=0)

# flatten MultiIndex
matrix.columns = [
    f"{prompt}_{metric}"
    for prompt, metric in matrix.columns.to_flat_index()
]

matrix = matrix.reset_index()

selected_prompt_names_shown = list(set([ f if f not in matched_prompt_names else matched_prompt_names[f] for f in selected_prompt_names ]))
# 2) For each prompt, build a single composite column
for p in selected_prompt_names_shown:
    if p in matched_prompt_names:
        p = matched_prompt_names[p]
    c1 = f"{p}_overall_punctuation"
    c2 = f"{p}_overall_punctuation_with_llm"
    c3 = f"{p}_overall_llm_labeled_percentage"
    # produce strings like "0.123 - 0.234 - 12.3%"
    matrix[p] = matrix.apply(
        lambda row: f"{row[c1]:.3f} - {row[c2]:.3f} - {row[c3]:.1f}%",
        axis=1
    )
    # drop the raw metric columns
    matrix.drop([c1,c2,c3], axis=1, inplace=True)

# 3) Reorder columns: model, temperature, then all your prompts

final_cols = ['model','temperature'] + selected_prompt_names_shown
matrix = matrix[final_cols]

# 4) Save to CSV
matrix.to_csv('stats/results_matrix_combined.csv', index=False)
print("✔ Saved results_matrix_combined.csv")