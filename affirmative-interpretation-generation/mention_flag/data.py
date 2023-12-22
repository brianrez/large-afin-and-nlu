# -*- coding: utf-8 -*-
"""
@author: Md Mosharaf Hossain
"""
from torch.utils.data import Dataset
import re
import pickle
import torch
from mf_cal_2 import find_index_sublist, mention_flag

class AFINDataset(Dataset):
    def __init__(self, tokenizer, data_list, params, target_attribute_name="affirmative_interpretation"):
        self.input_len = params.input_len
        self.target_len = params.target_len

        self.tokenizer = tokenizer
        self.target_attribute_name = target_attribute_name
    
        self.inputs = []
        self.targets = []

        self.mention_flags = []
        self.negations = []
        with open("negations.pkl", "rb") as handle:
            self.negations = pickle.load(handle)

        self.negation_ids =  tokenizer(self.negations)['input_ids']
        self.negation_ids = [l[:-1] for l in self.negation_ids]
        self.negation_dict = {}

        self.limited_negation_ids = ['not']
        self.limited_negation_ids = tokenizer(self.limited_negation_ids)['input_ids']
        self.limited_negation_ids = [l[:-1] for l in self.limited_negation_ids]

        self.original_cues = []
        for i in range(len(self.negations)):
            self.negation_dict[self.negations[i]] = self.negation_ids[i]
        

        self._process(data_list)
  
    def __len__(self):
        # print(len(self.inputs))
        return len(self.inputs)
        # return 10
    




    def _process(self, data_list):
        not_found = 0
        not_found_index = []  
        # for data_dict in data_list[:10]:
        for i in range(len(data_list)):
        # for i in range(10):
            data_dict = data_list[i]
            input_ = "sentence: {}".format(data_dict["sentence"].strip())                
            # input_ = input_ + "neg cues:"
            # input_ = input_ + ", ".join(self.negations)
            #  target_ = "<pad>affirmative_interpretation: {}".format( data_dict[self.target_attribute_name].strip() )
            target_ = "<pad>affirmative_interpretation: {}".format(" ")
            target_ = target_
    
            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
              [input_], padding='max_length', max_length=80, truncation=True, return_tensors="pt"
            ) 
            
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
              [target_], padding='max_length', max_length=50, truncation=True, return_tensors="pt"
            ) 
            found = False
            '''
            for negation_cue in self.negation_dict:
                if negation_cue in data_dict["sentence"]:
                    negation_cue_ids = self.negation_dict[negation_cue]
                    self.original_cues.append(negation_cue_ids)
                    # tokenized_inputs_ = torch.tensor([tokenized_inputs["input_ids"].clone().tolist()])
                    # tokenized_targets_ = torch.tensor([tokenized_targets["input_ids"].clone().tolist()])
                    tokenized_inputs_ = tokenized_inputs["input_ids"].clone()
                    tokenized_targets_ = tokenized_targets["input_ids"].clone()
                    mention_flag_matrix = mention_flag(tokenized_inputs_, tokenized_targets_, list([negation_cue_ids]), self.negation_ids)
                    self.mention_flags.append(mention_flag_matrix[0])
                    found = True
                    break
            '''

            for negation_id in self.negation_ids:
                if find_index_sublist(tokenized_inputs["input_ids"][0].tolist(), negation_id) is not None:
                    negation_cue_ids = negation_id
                    self.original_cues.append(negation_cue_ids)
                    # tokenized_inputs_ = torch.tensor([tokenized_inputs["input_ids"].clone().tolist()])
                    # tokenized_targets_ = torch.tensor([tokenized_targets["input_ids"].clone().tolist()])
                    tokenized_inputs_ = tokenized_inputs["input_ids"].clone()
                    tokenized_targets_ = tokenized_targets["input_ids"].clone()
                    mention_flag_matrix = mention_flag(tokenized_inputs_, tokenized_targets_, list([negation_cue_ids]), self.limited_negation_ids)
                    self.mention_flags.append(mention_flag_matrix[0])
                    found = True
                    break
            
            if not found:
                mention_flag_matrix = torch.zeros((1, 50, 80))
                self.original_cues.append([])
                self.mention_flags.append(mention_flag_matrix[0])
                self.inputs.append(tokenized_inputs)
                self.targets.append(tokenized_targets)
                
                not_found_index.append(i)
                not_found += 1
                # discarded out of the dataset 
            if found:
                self.inputs.append(tokenized_inputs)
                self.targets.append(tokenized_targets)
        # print(f"not found: {not_found}")
        # for index in not_found_index:

    def __getitem__(self, index):
        # print(len(self.inputs))
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask    = self.inputs[index]["attention_mask"].squeeze()  
        target_mask = self.targets[index]["attention_mask"].squeeze()  
        # print(len(self.original_cues))
        original_cues = self.original_cues[index]

        mention_flags = self.mention_flags[index].squeeze()

        return {"source_ids": source_ids,
                "source_mask": src_mask,
                "target_ids": target_ids,
                "target_mask": target_mask,
                "mention_flags": mention_flags,
                "original_cues": original_cues}

def collate_(batch):
    src_ids = []
    src_mask = []
    trg_ids = []
    trg_mask = []
    mention_flags = []
    original_cues = []
    for instance in batch:
        src_ids.append(instance["source_ids"].tolist())
        src_mask.append(instance["source_mask"].tolist())
        trg_ids.append(instance["target_ids"].tolist())
        trg_mask.append(instance["target_mask"].tolist())
        mention_flags.append(instance["mention_flags"].tolist())
        original_cues.append(instance["original_cues"])
    src_ids = torch.tensor(src_ids, dtype=torch.long)
    src_mask = torch.tensor(src_mask, dtype=torch.long)
    trg_ids = torch.tensor(trg_ids, dtype=torch.long)
    trg_mask = torch.tensor(trg_mask, dtype=torch.long)
    mention_flags = torch.tensor(mention_flags, dtype=torch.long)

    return_dict = {
        "source_ids": src_ids,
        "source_mask": src_mask,
        "target_ids": trg_ids,
        "target_mask": trg_mask,
        "mention_flags": mention_flags,
        "original_cues": original_cues
    }
    return return_dict

class NewDataset(Dataset):
    def __init__(self, tokenizer, data_list, params):
        self.input_len = params.input_len
        self.target_len = params.target_len
        self.tokenizer = tokenizer    
        self.inputs = []

        self._process(data_list)
  
    def __len__(self):
        return len(self.inputs)
  
    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        src_mask   = self.inputs[index]["attention_mask"].squeeze()  

        return {"source_ids": source_ids, 
                "source_mask": src_mask}

    def _process(self, data_list):
        for text in data_list:
            input_ = "sentence: {}".format(text.strip())                
            input_ = input_ + ' </s>'
    
            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
              [input_], max_length=self.input_len, padding='max_length', truncation=True, return_tensors="pt"
            ) 
                
            self.inputs.append(tokenized_inputs)


