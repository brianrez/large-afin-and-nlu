# -*- coding: utf-8 -*-
"""
@author: Md Mosharaf Hossain
"""

import random
import numpy as np
import torch
import tqdm
import argparse
import json
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    Adafactor,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

from model import AFINGenerator
from data import AFINDataset
from config import Config
import time
import utils


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)  
      torch.cuda.manual_seed_all(seed)    
    
    
def get_optimizer(params, model):
    no_decay =           ["bias", "LayerNorm.weight"]
    grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": params.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    if params.optimizer == "AdamW":
        optimizer = AdamW(grouped_parameters, lr=params.learning_rate, eps=params.adam_epsilon)
    elif params.optimizer == "Adafactor":
        optimizer = Adafactor(grouped_parameters, lr=params.learning_rate, scale_parameter=False,
                         relative_step=False)
    return optimizer


def run(init_path, save_path):
    torch.cuda.empty_cache()

    start_time = time.time()
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-c", "--config_path", help="path of the configuration file", required=True)  
    args        = argParser.parse_args()
    config_path = args.config_path



    # Step 1: Read parameters and set a seed ----------------------------------------------------
    with open(config_path) as json_file_obj: 
        params = json.load(json_file_obj)

    params = Config(params)
    set_seed(params.seed)

    # Step 2: Get appropriate device ------------------------------------------------------------
    if torch.cuda.is_available()==True and params.use_gpu: 
        device = torch.device("cuda:"+str(params.device))
    else: 
        device = torch.device("cpu") 



    # Step 3: Get tokenizer and initialize the model------------------------------------------------
    tokenizer = T5Tokenizer.from_pretrained(params.T5_path[params.T5_type])
    model = AFINGenerator(params, device, tokenizer) 
    model.to(device)  
    print("Model is successfully loaded!")
    # state = dict(model=model.state_dict())
    model.load_state_dict(torch.load(params.best_model_path)['model'])


    # Step 4: Prepare the datasets for data loader -------------------------------------------------
    target_attribute_name = "affirmative_interpretation"
    '''
    train_data = utils.read_data(params.data_path["train"])
    train_dataset_gold = AFINDataset(tokenizer, train_data, params, target_attribute_name)
    '''
    # train_size = len(train_dataset_gold)
    # print("\n train size: {}".format(train_size))
    dev_data   = utils.read_data(init_path)
    print("\n dev size: {}".format(len(dev_data)))
    dev_dataset   = AFINDataset(tokenizer, dev_data, params, target_attribute_name)
    dev_size   = len(dev_dataset)
    print("\n dev size: {}".format(dev_size))


    # Step 6: Train the model and save the best model--------------------------------------------------------
    min_loss = float('inf')
    dev_output = []
    dev_ids    = []

    dev_loader = DataLoader(dev_dataset, batch_size=params.batch_size, shuffle=False)
    dev_loss = 0
    for batch_idx, batch in enumerate(dev_loader):                     
        #model.eval()
        
        # temp_loss = model(batch)
        
        # dev_loss += temp_loss.item()
        #print(f"batch_idx: {batch_idx}, dev batch: {len(batch)}, dev_loss: {dev_loss/(batch_idx+1)}")   
        
        if params.use_multi_gpus:
            dev_batch_output = model.module.predict(batch)
        else:
            dev_batch_output = model.predict(batch)
        dev_output.extend(dev_batch_output)  
        # dev_ids.append(batch["ids"])

    dev_output = [output.strip() for output in dev_output]
    # final_output = [dev_output, dev_ids]
    print("dev_output: {}".format(len(dev_output)))
    import pickle
    # step -test: write the predicted results
    with open(save_path, 'wb') as f:
        pickle.dump(dev_output, f)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Completed! Training duration: {} hours".format(elapsed_time/3600.0))
    time.strftime("%H:%M:%S", time.gmtime(elapsed_time))  

run("./data/condaqa/train_uniques.jsonl", "./outputs_conda/train.pkl")
run("./data/condaqa/val_uniques.jsonl",   "./outputs_conda/val.pkl"  )
run("./data/condaqa/test_uniques.jsonl",  "./outputs_conda/test.pkl" )