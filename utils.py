import math
import numpy as np 
import re, string

import torch
import torch.nn as nn
import numpy as np 
from tqdm import tqdm
import os
import random
import json

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


#compute logits from https://github.com/szhang42/Calibration_qa/tree/05b4354e20b746ab0ea36c16573076a2a4fdc298
def compute_softmax_score_generative(answers,temperature=1.0):
    """Compute softmax probability over raw logits."""
    if not answers:
        return []

    max_score = None
    for answer in answers:
        if max_score is None or answer["logit_score"] > max_score:
            max_score = answer["logit_score"]/temperature

    exp_scores = []
    answer_texts = []
    total_sum = 0.0
    for answer in answers:
        score = answer["logit_score"]/temperature
        x = math.exp(score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    processed_scores = []
    for idx, score in enumerate(exp_scores):
        probs.append(score / total_sum)
        processed_scores.append({ "text": answers[idx]['text'], "prob": score / total_sum })
    return processed_scores

#compute logits from https://github.com/szhang42/Calibration_qa/tree/05b4354e20b746ab0ea36c16573076a2a4fdc298
def compute_softmax_score_extractive(answers,temperature=1.0):
    """Compute softmax probability over raw logits."""
    if not answers:
        return []

    max_score = None
    for answer in answers:
        if max_score is None or answer["logit_score"] > max_score:
            max_score = answer["logit_score"]/temperature

    exp_scores = []
    answer_texts = []
    total_sum = 0.0
    for answer in answers:
        score = answer["logit_score"]/temperature
        x = math.exp(score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    processed_scores = []
    for idx, score in enumerate(exp_scores):
        probs.append(score / total_sum)
        processed_scores.append({ "text": answers[idx]['text'], "softmax": score / total_sum })
    return processed_scores




def train_model(model, tokenizer, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, device, save_path, num_epochs=25):
    model.to(device)

    best_loss = np.inf
    
    #loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

    for epoch in range(num_epochs):
        print(scheduler.get_last_lr())
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        # Iterate over data.
        for inputs in tqdm(train_dataloader):
            batch = {k: inputs[k].to(device) for k in inputs}
            
            start_labels = batch["start_positions"]
            end_labels = batch["end_positions"]
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(**batch)
            
            start_loss = criterion(outputs.start_logits, start_labels)
            end_loss = criterion(outputs.end_logits, end_labels)
            
            loss = (start_loss + end_loss) / 2
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            # statistics
            running_loss += loss.item() * start_labels.shape[0]
            total += start_labels.shape[0]

        
        epoch_loss = running_loss / total
        
        valid_loss = test_model(model, valid_dataloader,device)
        if valid_loss < best_loss:
            print('{} Loss: {:.4f} Best loss: {:.4f}'.format("validation ", valid_loss, best_loss))
            best_loss = valid_loss
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            best_model = model
        model.save_pretrained(save_path+os.sep+str(epoch))
        tokenizer.save_pretrained(save_path+os.sep+str(epoch))
        print('{} Loss: {:.4f} Total: {}'.format("train", epoch_loss, total))

    return best_model

def test_model(model, test_dataloader,device):
    model.to(device)
    model.eval()  # Set model to test mode

    total = 0
    running_loss = 0
    
    loss_fct = nn.CrossEntropyLoss()
    
    # Iterate over data.
    with torch.no_grad():
        for inputs in tqdm(test_dataloader):
            batch = {k: inputs[k].to(device) for k in inputs}
            
            start_labels = batch["start_positions"]
            end_labels = batch["end_positions"]
            
            # forward
            outputs = model(**batch)
            
            start_loss = loss_fct(outputs.start_logits, start_labels)
            end_loss = loss_fct(outputs.end_logits, end_labels)
            
            loss = (start_loss + end_loss) / 2
            
            running_loss += loss.item() * start_labels.shape[0]
            total += start_labels.shape[0]

    epoch_loss = running_loss / total
    print("validation", epoch_loss, total)
    
    return epoch_loss


def test_model_conf(model, test_dataloader,device):
    model.to(device)
    model.eval()  # Set model to test mode

    total = 0
    running_loss = 0
    result_conf = []
    
    # Iterate over data.
    with torch.no_grad():
        for inputs in tqdm(test_dataloader):
            batch = {k: inputs[k].to(device) for k in inputs}
            
            start_label = batch["start_positions"][0]
            end_label = batch["end_positions"][0]
            
            # forward
            outputs = model(**batch)
            
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            #print(torch.softmax(start_logits, dim=1)[0])
            #print(torch.max(torch.softmax(start_logits, dim=1)))
            start_conf = torch.softmax(start_logits, dim=1)[0][start_label].item()
            end_conf = torch.softmax(end_logits, dim=1)[0][end_label].item()
            
            result_conf.append(start_conf*end_conf)
            
            #print("start label", start_label, "end label", end_label, "start_conf", start_conf, "end_conf", end_conf)
    
    return result_conf


def test_model_embedding(model, test_dataloader,device):
    model.to(device)
    model.eval()  # Set model to test mode

    total = 0
    running_loss = 0
    result_conf = []
    
    # Iterate over data.
    with torch.no_grad():
        for inputs in tqdm(test_dataloader):
            batch = {k: inputs[k].to(device) for k in inputs}
            
            start_label = batch["start_positions"][0]
            end_label = batch["end_positions"][0]
            
            # forward
            outputs = model(**batch, output_hidden_states=True)
    
    return result_conf