import transformers
import os, sys
import datasets
import evaluate
import numpy as np 

from datasets import load_dataset, concatenate_datasets
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, MBart50TokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import set_seed
import argparse
import torch
from tqdm import tqdm
from operator import itemgetter

import string
import re

import math
import json

import pandas as pd

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

def train_model(args):
    
    config = AutoConfig.from_pretrained(args.path)
    tokenizer = AutoTokenizer.from_pretrained(args.path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.path, config=config)

    seed = 523


    def add_eos_to_examples(example):
        example['input_text'] = 'question: %s context: %s ' % (example['question'], example['context'])
        example['target_text'] = '%s ' % example['answers']['text'][0]
        
        return example
    
    def convert_to_features(example_batch):
        input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'], padding="max_length", truncation=True, max_length=512)
        target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], padding="max_length", truncation=True, max_length=512)
        
        
        target_encodings["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in target_encodings["input_ids"]
        ]
        
        encodings = {
            'input_ids': input_encodings['input_ids'], 
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids'],
            'decoder_attention_mask': target_encodings['attention_mask']
        }
        
        return encodings

    # Metric
    metric = evaluate.load("bleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        #result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
    
    label_pad_token_id = -100 
    training_args = Seq2SeqTrainingArguments(output_dir="./results_"+args.save_path, save_strategy="epoch",learning_rate=3e-5,per_device_train_batch_size=16,
            per_device_eval_batch_size=32,num_train_epochs=args.num_of_epochs,weight_decay=0.01, save_total_limit=20, fp16=False, evaluation_strategy = "epoch",load_best_model_at_end=True)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )


    # Get datasets
    
    #squad = load_dataset('json', data_files={'train':args.train_path, 'validation':args.validation_path}, field='data')
    #train_dataset = squad['train']
    #valid_dataset = squad['validation']
    train_dataset = load_dataset('squad', split='train')
    valid_dataset = load_dataset('squad', split='validation')
    # map add_eos_to_examples function to the dataset example wise & map convert_to_features batch wise
    train_dataset = train_dataset.map(add_eos_to_examples)
    train_dataset = train_dataset.map(convert_to_features, batched=True)

    valid_dataset = valid_dataset.map(add_eos_to_examples, load_from_cache_file=False)
    valid_dataset = valid_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)
    
    # set the tensor type and the columns which the dataset should return
    columns = ['input_ids', 'labels', 'attention_mask', 'decoder_attention_mask']
    train_dataset.set_format(type='torch', columns=columns)
    valid_dataset.set_format(type='torch', columns=columns)
    
    print(train_dataset)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    set_seed(seed)
    
    trainer.train()
    trainer.save_model(args.save_path)
    
    
def train_model_mix(args, language_list, mix_strategy):
    
    config = AutoConfig.from_pretrained(args.path)
    tokenizer = AutoTokenizer.from_pretrained(args.path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.path, config=config)

    seed = 523


    def add_eos_to_examples(example):
        example['input_text'] = 'question: %s context: %s ' % (example['question'], example['context'])
        example['target_text'] = '%s ' % example['answers']['text'][0]
        
        return example
    
    def convert_to_features(example_batch):
        input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'], padding="max_length", truncation=True, max_length=512)
        target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], padding="max_length", truncation=True, max_length=64)
        
        
        target_encodings["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in target_encodings["input_ids"]
        ]
        
        encodings = {
            'input_ids': input_encodings['input_ids'], 
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids'],
            'decoder_attention_mask': target_encodings['attention_mask']
        }
        
        return encodings

    # Metric
    save_path = args.save_path + '-mix-learning-'+mix_strategy+' '+str(args.mix_number)
    
    metric = evaluate.load("bleu")
    
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        #result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
    
    label_pad_token_id = -100 
    training_args = Seq2SeqTrainingArguments(output_dir="./results_"+save_path, save_strategy="epoch",learning_rate=5e-5,per_device_train_batch_size=16,
            per_device_eval_batch_size=32,num_train_epochs=args.num_of_epochs,weight_decay=0.01, save_total_limit=20, fp16=False, evaluation_strategy = "epoch",load_best_model_at_end=True)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,)

    squad = load_dataset('squad')
    if mix_strategy == 'original_mix':
        # Get datasets
        train_dataset = load_dataset('squad', split='train')
        valid_dataset = load_dataset('squad', split='validation')
        
        for lang in tqdm(language_list):            
            other_dataset = load_dataset('json', data_files={'train':'./data/xquad_translated_train_all.json', 'validation':'./data/xquad_translated_dev_all.json'}, field=lang, features=train_dataset.features)
            temp_dataset = other_dataset['train'].select(range(0,args.mix_number))
            train_dataset = concatenate_datasets([train_dataset, temp_dataset])
            
    elif mix_strategy == 'select_same_mix':
        # Get datasets
        temp_dataset = load_dataset('json', data_files={'train':'./data/xquad_translated_train_all.json', 'validation':'./data/xquad_translated_dev_all.json'}, field='en', features=squad['train'].features)
        train_dataset = temp_dataset['train'].select(range(0,args.mix_number))
        valid_dataset = load_dataset('squad', split='validation')
        
        for lang in tqdm(language_list):            
            other_dataset = load_dataset('json', data_files={'train':'./data/xquad_translated_train_all.json', 'validation':'./data/xquad_translated_dev_all.json'}, field=lang, features=train_dataset.features)
            temp_dataset = other_dataset['train'].select(range(0,args.mix_number))
            train_dataset = concatenate_datasets([train_dataset, temp_dataset])
    
    elif mix_strategy == 'select_split_mix':
        step = int(59574/(len(language_list)+1))
        temp_dataset = load_dataset('json', data_files={'train':'./data/xquad_translated_train_all.json', 'validation':'./data/xquad_translated_dev_all.json'}, field='en', features=squad['train'].features)
        train_dataset = temp_dataset['train'].select(range(0,step))
        valid_dataset = load_dataset('squad', split='validation')
        
        for idx, lang in tqdm(enumerate(language_list)):            
            other_dataset = load_dataset('json', data_files={'train':'./data/xquad_translated_train_all.json', 'validation':'./data/xquad_translated_dev_all.json'}, field=lang, features=train_dataset.features)
            temp_dataset = other_dataset['train'].select(range((idx+1)*step,(idx+2)*step))
            train_dataset = concatenate_datasets([train_dataset, temp_dataset])
        
    elif mix_strategy == 'select_no_mix':
        temp_dataset = load_dataset('json', data_files={'train':'./data/xquad_translated_train_all.json', 'validation':'./data/xquad_translated_dev_all.json'}, field='en', features=squad['train'].features)
        train_dataset = temp_dataset['train'].select(range(0,args.mix_number))
        valid_dataset = load_dataset('squad', split='validation')
        
    elif mix_strategy == 'original_no_mix':
        train_dataset = load_dataset('squad', split='train')
        valid_dataset = load_dataset('squad', split='validation')
        
                
    # map add_eos_to_examples function to the dataset example wise & map convert_to_features batch wise
    train_dataset = train_dataset.map(add_eos_to_examples)
    train_dataset = train_dataset.map(convert_to_features, batched=True)

    valid_dataset = valid_dataset.map(add_eos_to_examples, load_from_cache_file=False)
    valid_dataset = valid_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)
    
    # set the tensor type and the columns which the dataset should return
    columns = ['input_ids', 'labels', 'attention_mask', 'decoder_attention_mask']
    train_dataset.set_format(type='torch', columns=columns)
    valid_dataset.set_format(type='torch', columns=columns)
    
    print(train_dataset)
    
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    set_seed(seed)
    
    trainer.train()
    trainer.save_model(save_path)
    
    with open(save_path +os.sep + 'mix-training-config.json', 'w') as file:
        json.dump({'epoch':args.num_of_epochs, 'model_path':args.path, 'lang_list': language_list, 'save_path':save_path, 'mix_training':args.mix_training, 'fewshot-size':args.mix_number, 'mix-strategy':mix_strategy}, file)
    file.close()  
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess qamr file into huggingface format')
    parser.add_argument('--path',type=str, default="roberta-base", help='path for reading pretrained model')
    parser.add_argument('--train_model', type=bool, default=False, help='whether to train the model')
    parser.add_argument('--num_of_epochs', type=int, default=1, help='the number of epochs')
    parser.add_argument('--save_path', type=str, default='mbart-qa', help='the path for saving trained models')
    parser.add_argument('--evaluate_model', type=bool, default=False, help='whether to evaluate the model')
    parser.add_argument('--evaluate_path', type=str, default='mbart-large-qa', help='the path for saving trained models')
    parser.add_argument('--dataset', type=str, default='xquad', help='the path for saving trained models')
    parser.add_argument('--split', type=str, default='xquad.en', help='the path for saving trained models')
    parser.add_argument('--train_path', type=str, default='xquad_translated_train_10000.json', help='the path for saving trained models')
    parser.add_argument('--validation_path', type=str, default='xquad_translated_dev_0.json', help='the path for saving trained models')
    parser.add_argument('--mix_training',type=bool, default=False, help='whether to mix translated examples for training')
    parser.add_argument('--mix_number',type=int, default=100, help='the number of examples from other languages')
    parser.add_argument('--mix_strategy',type=str, default='original_no_mix', choices=['original_mix','original_no_mix','select_same_mix','select_split_mix','select_no_mix'], help='whether to mix translated examples for training')
    
    args = parser.parse_args()
    mix_strategy = args.mix_strategy
    print(args)
    
    if args.train_model:
        train_model(args)

    if args.mix_training:
        language_list = ['ar','de','es','hi','vi','zh']
        train_model_mix(args, language_list,mix_strategy)
        
        
    
