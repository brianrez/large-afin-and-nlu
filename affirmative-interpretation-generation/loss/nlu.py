from test import generate

import json
class jsonl:
    def read(self, path):
        with open(path) as f:
            all_sentences = [json.loads(line) for line in f.readlines()]
        return all_sentences

    def write(self, path, data):
        with open(path, "w") as f:
            for line in data:
                f.write(json.dumps(line) + "\n")


def mo_par(path, destination, key): 
    train = jsonl().read(path + "/train.jsonl")
    new_data = []
    for line in train:
        new_data.append({"sentence": line[key]})
    
    jsonl().write("temp.jsonl", new_data)
    affirs = generate()

    for i in range(len(train)):
        sent = train[i][key]
        train[i][key] = sent + "Affirmative Interpretation: " + affirs[i]
    
    jsonl().write(destination + "/train.jsonl", train)

    print("train done")

    val = jsonl().read(path + "/val.jsonl")
    new_data = []
    for line in val:
        new_data.append({"sentence": line[key]})
    
    jsonl().write("temp.jsonl", new_data)
    affirs = generate()

    for i in range(len(val)):
        sent = val[i][key]
        val[i][key] = sent + " " + affirs[i]
    
    jsonl().write(destination + "/val.jsonl", val)

    print("val done")
    

import requests

mo_par("./data/commonsenseqa", "./newdata/commonsenseqa/mo", "question")
print("commonsenseqa done")  

requests.post("https://ntfy.sh/mhrnlpmodels",
    data="Chameleon MO. Commonsenseqa Done!".encode(encoding='utf-8'))

mo_par("./data/wsc", "./newdata/wsc/mo", "text")
print("wsc done")

requests.post("https://ntfy.sh/mhrnlpmodels",
    data="Chameleon MO. WSC Done!".encode(encoding='utf-8'))

mo_par("./data/wic", "./newdata/wic/mo", "sentence1")
mo_par("./newdata/wic/mo", "./newdata/wic/mo", "sentence2")
print("wic done")

requests.post("https://ntfy.sh/mhrnlpmodels",
    data="Chameleon MO. WIC Done!".encode(encoding='utf-8'))

mo_par("./data/stsb", "./newdata/stsb/mo", "text_a")
mo_par("./newdata/stsb/mo", "./newdata/stsb/mo", "text_b")
print("stsb done")

requests.post("https://ntfy.sh/mhrnlpmodels",
    data="Chameleon MO. STSB Done!".encode(encoding='utf-8'))


mo_par("./data/qnli", "./newdata/qnli/mo", "hypothesis")
mo_par("./newdata/qnli/mo", "./newdata/qnli/mo", "premise")
print("qnli done")

requests.post("https://ntfy.sh/mhrnlpmodels",
    data="Chameleon MO. QNLI Done!".encode(encoding='utf-8'))
