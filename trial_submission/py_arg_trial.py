# from py_arg import Argument, ArgumentationFramework

# a = Argument("A", "Eating sugar causes hyperactivity.")
# b = Argument("B", "Multiple meta‑analyses find no effect.")

# af = ArgumentationFramework()
# af.add_argument(a); af.add_argument(b)
# af.add_attack(b, a)    # B attacks A
# # No support relations needed

# res = af.evaluate(semantics="grounded")
# print("Accepted:", res.accepted)   # {B}
# print("Rejected:", res.rejected)   # {A}


# from transformers import pipeline

# zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# candidate_labels = [
#     "ad hominem", "straw man", "false causality", "circular reasoning",
#     "false dilemma", "oversimplification", "appeal to emotion",
#     # …add all fallacies you care about…
# ]

# result = zero_shot(full_text, candidate_labels, multi_label=True)
# # result['labels'] and result['scores'] give you each label’s confidence
# print(list(zip(result["labels"], result["scores"])))
# # e.g. → [('false causality', 0.68), ('oversimplification', 0.45), …]


# from transformers import pipeline

# # load the DistilBERT classifier fine‑tuned on 14 common fallacies
# fallacy_pipe = pipeline(
#     "text-classification",
#     model="q3fer/distilbert-base-fallacy-classification",
#     return_all_scores=True
# )

# snippet = """MT: "Claire’s absolutely right about that
# But then the problem is that that form of capitalism wasn’t generating sufficient surpluses
# And so therefore where did the money flow
# It didn’t flow into those industrial activities because in the developed world that wasn’t making enough money\""""
# preds = fallacy_pipe(snippet)
# print(preds)

# lines = snippet.split("\n")
# for line in lines:
#     preds = fallacy_pipe(line)
#     # sort by descending score
#     top_label, top_score = max(preds[0], key=lambda x: x["score"]).values()
#     print(f"\"{line}\" → {top_label} ({top_score:.2f})")
import re
import logging
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
  
        # logger.warning('Missing CQs')
        return output
txt = "What products are being referred to in the discussion about salt and fat content\nIs the comparison between food products and credit card companies a fair one\nHow does the speaker define irresponsible behavior in this context"

print(structure_output(txt))


