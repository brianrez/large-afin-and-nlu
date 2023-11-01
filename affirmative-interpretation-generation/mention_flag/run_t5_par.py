from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)

def paraphrase(
    question,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=1,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(device)
    
    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res

import json
import pickle

with open("./negations.pkl", "rb") as F:
    NEG_CUES = pickle.load(F)

from random import shuffle
def paraphraser(file_path, save_path, key):
    print(f"Paraphrasing {file_path} to {save_path}")
    with open(file_path) as f:
        all_sentences = [json.loads(line) for line in f.readlines()]

    additional_sentences = []

    didnt_work = []
    index = 0
    for row in all_sentences:
        sentence = row["sentence"]
        par = paraphrase(sentence)
        negated = False
        for cue in NEG_CUES:
            if cue in par[0]:
                negated = True
                break
        if negated:
            row[key] = par[0]
            additional_sentences.append(row)
        else:
            didnt_work.append(row)
        index += 1
        print(f"{index}/{len(all_sentences)}", end="\r")
    print("finished first pass, starting second pass")
    for row in didnt_work:
        sentence = row["sentence"]
        pars = paraphrase(sentence, num_return_sequences=5)
        index = 0
        for par in pars:
            negated = False
            for cue in NEG_CUES:
                if cue in par:
                    negated = True
                    break
            if negated:
                row[key] = par
                additional_sentences.append(row)
                break
            index += 1
            print(f"{index}/{len(pars)}", end="\r")

    new_all = all_sentences + additional_sentences
    shuffle(new_all)

    with open(save_path, "w") as f:
        for row in new_all:
            f.write(json.dumps(row) + "\n")

    print(f"Paraphrased {file_path} to {save_path}")
        
# paraphraser("./data/afin/train.jsonl", "./data/afin/train-extra.jsonl", "pi")
# paraphraser("./data/afin/test.jsonl", "./data/afin/test-extra.jsonl", "pi")
paraphraser("./data/afin/dev.jsonl", "./data/afin/dev-extra.jsonl", "pi")

paraphraser("./data/large-afin/large-afin.jsonl", "./data/large-afin/large-afin-extra.jsonl", "affirmative_interpretation")



    

    