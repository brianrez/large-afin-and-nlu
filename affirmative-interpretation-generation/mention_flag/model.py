# -*- coding: utf-8 -*-
"""
@author: Md Mosharaf Hossain
"""

import torch.nn as nn
# from transformers import (
#     T5ForConditionalGeneration
# )

from t5 import T5ForConditionalGeneration

class AFINGenerator(nn.Module):
    def __init__(self, params, device, tokenizer):
        super(AFINGenerator, self).__init__()
        
        self.device = device
        self.params = params
        self.tokenizer = tokenizer
        self.transf_model = T5ForConditionalGeneration.from_pretrained(params.T5_path[params.T5_type])
        self.negation_ids = []

    def forward(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        
        outputs = self.transf_model(
            input_ids=batch["source_ids"].to(self.device),
            attention_mask=batch["source_mask"].to(self.device),
            labels=lm_labels.to(self.device),
            decoder_copy_mention_flag=batch["mention_flags"].to(self.device),
            decoder_attention_mask=batch['target_mask'].to(self.device),
            original_cues = batch["original_cues"],
            all_negations = self.negation_ids
        )
        loss = outputs[0]

        return loss
    
    def set_negations(self, negations):
        self.negation_ids = negations

    def predict(self, batch):
        self.eval()
        input_ids = batch["source_ids"].to(self.device)
        attention_mask = batch["source_mask"].to(self.device)


        
        # print("input_ids: ", input_ids.shape)
        # print("input_ids: ", input_ids[0])
        bad_words_ids = [self.tokenizer(bad_word).input_ids for bad_word in self.params.bad_words]

        outputs = self.transf_model.search(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=self.params.num_beams,
            max_length=self.params.target_len,            
            early_stopping=self.params.early_stopping,
            do_sample=self.params.do_sample,
            top_k=self.params.top_k,
            top_p=self.params.top_p,
            repetition_penalty=self.params.repetition_penalty,
            num_return_sequences=self.params.num_return_sequences,
            bad_words_ids=bad_words_ids,
            decoder_original_input_ids=input_ids,
            original_cues = batch["original_cues"],
            all_negations = self.negation_ids
        )
        
        outputs = [self.tokenizer.decode(ids) for ids in outputs]
        self.train()
        return outputs
    
