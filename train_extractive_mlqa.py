from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoConfig, BertForQuestionAnswering
from transformers import TrainingArguments, Trainer, DefaultDataCollator
import transformers
from evaluate import load
from tqdm import tqdm
import torch
import json
import numpy as np
import collections
import argparse
import sys, os
import matplotlib.pyplot as plt
import time
import random
import math
from utils import train_model, test_model
import torch 
from torch import nn

#following huggingface tutorial of question answering: https://huggingface.co/course/chapter7/7?fw=pt
# to call the script, python3 train_multilang_question_answering.py --evaluate_path squad_bert-base-multilingual-cased --compute_ECE True
# to run the program python3 train_multilang_question_answering.py --path xlm-robert-base --mix_strategy select_same_mix/select_split_mix/select_no_mix --mix_training True --min_number 8510 --num_of_epochs 3 --save_path xlm-mix

torch.manual_seed(523)
random.seed(523)
#compress tokenizer warning 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"


def preprocess_function(examples):
    #print(examples["question"])
    questions = [q.lstrip() for q in examples["question"]]

    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
        return_overflowing_tokens=True,
        stride=128       
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_mapping = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        
        sample_index = sample_mapping[i]
        #print(sample_index)
        answer = examples["answers"][sample_index]
        #answer = answers[i]
        #print("offset",offset,answer,len(inputs))
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    
    return inputs

def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
        return_overflowing_tokens=True,
        stride=128 
    )

    
    sample_mapping = inputs.pop("overflow_to_sample_mapping")
    example_ids = []
    
    start_positions = []
    end_positions = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_mapping[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else (None,None) for k, o in enumerate(offset)
        ]
        
    
    for i, offset in enumerate(inputs["offset_mapping"]):
    
        sample_index = sample_mapping[i]
        #print(sample_index)
        answer = examples["answers"][sample_index]
        #answer = answers[i]
        #print("offset",offset,answer,len(inputs))
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    inputs["example_id"] = example_ids
    
    return inputs


def evaluate_question_answering(model, n_best=20,max_answer_length =30, split="validation"):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model  = model.to(device)
    model.eval()
    predicted_answers=[]
    references=[]
    
    for i in tqdm(range(0,len(squad[split]),100)):
        small_eval_set = squad[split].select(range(i,min(len(squad[split]),i+100)))
        eval_set = small_eval_set.map(preprocess_validation_examples,batched=True,remove_columns=small_eval_set.column_names,)
        eval_set_for_model = eval_set.remove_columns(["example_id", "offset_mapping"])
        eval_set_for_model.set_format("torch")
        
        batch = {k: eval_set_for_model[k].to(device) for k in eval_set_for_model.column_names}
        with torch.no_grad():
            predictions =model(**batch)

        start_logits = predictions.start_logits.cpu().numpy()
        end_logits = predictions.end_logits.cpu().numpy()
        example_to_features = collections.defaultdict(list)
        offset_list = np.array(eval_set["offset_mapping"])


        for idx, feature in enumerate(eval_set):
            example_to_features[feature["example_id"]].append(idx)

        start = 0

        for example in small_eval_set:
            
            example_id = example["id"]
            context = example["context"]
            answers = []
            
            for feature_index in example_to_features[example_id]:
                
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = offset_list[feature_index]
                
                start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
                
                
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length.
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        answers.append(
                            {
                                "text": context[offsets[start_index][0] : offsets[end_index][1]],
                                "logit_score": start_logit[start_index] + end_logit[end_index],
                            }
                        )
            #if len(answers) > 0:
            #    best_answer = max(answers, key=lambda x: x["logit_score"])
            #    predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})
            #else:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})
            references.append({"id":example_id,"answers":example["answers"]})
            #print('-----------------------------------------------')
            #print(context)
            #print("answers: ",{"id": example_id, "prediction_text": best_answer["text"]}, len(context),example["question"])
            #print("reference: ", {"id":example_id,"answers":example["answers"]})
    return predicted_answers,references    
 
       
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess qamr file into huggingface format')
    parser.add_argument('--path',type=str, default="roberta-base", help='path for reading pretrained model')
    parser.add_argument('--train_model',type=bool, default=False, help='whether to train a new question answering model')
    parser.add_argument('--evaluate_path',type=str, default=None, help='path for fine-tuned model')
    parser.add_argument('--evaluate_model',type=bool, default=False, help='whether to evaluate ')
    parser.add_argument('--result_folder',type=str, default=None, help='folder of training logs ')
    parser.add_argument('--save_path',type=str, default='few-shot', help='whether to compare ECE')
    parser.add_argument('--num_of_epochs',type=int, default=3, help='number of epochs ')
    parser.add_argument('--min_number',type=int, default=10000, help='number of epochs ')
    parser.add_argument('--choice', type=str, default = "squad")
    parser.add_argument('--validate_models',type=bool, default=False, help='whether to train validation steps ')
    parser.add_argument('--steps',type=int, default=20000, help='number of steps ')
    parser.add_argument('--dataset_name',type=str, default='xquad', help='dataset we want to load')
    parser.add_argument('--dataset_split',type=str, default='xquad.ar', help='dataset we want to load')
    parser.add_argument('--model_name',type=str, default='multilingual-cased', help='')
    parser.add_argument('--ls_factor',type=float, default=0.0, help='label smoothing factor')
    parser.add_argument('--run_temperature',type=bool, default=False, help='whether to run temperature scaling')
    parser.add_argument('--mix_training',type=bool, default=False, help='whether to mix translated examples for training')
    parser.add_argument('--mix_strategy',type=str, default='original_no_mix', choices=['original_mix','original_no_mix','select_same_mix','select_split_mix','select_no_mix'], help='whether to mix translated examples for training')
    parser.add_argument('--train_data_path', type=str, default='./data/xquad_translated_train_5000_0.json', help='the path for saving trained models')
    parser.add_argument('--validation_data_path', type=str, default='./data/xquad_translated_dev_0.json', help='the path for saving trained models')

    args = parser.parse_args()

    path = args.path
    evaluate_path = args.evaluate_path
    mix_strategy = args.mix_strategy
    print(args)
        
    #   print(squad)
            
    if args.train_model:
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained(path,use_fast=True)
        squad = load_dataset('squad')
        train_dataset = squad['train']
        valid_dataset = squad['validation']

        tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
        tokenized_squad.set_format("torch")
        model = AutoModelForQuestionAnswering.from_pretrained(path)
        print(model)
        
        criterion = nn.CrossEntropyLoss(label_smoothing = args.ls_factor)
        
        training_args = TrainingArguments(output_dir="./results_"+args.save_path, save_strategy="epoch",learning_rate=3e-5,per_device_train_batch_size=16,
            per_device_eval_batch_size=32,num_train_epochs=args.num_of_epochs,weight_decay=0.01, save_total_limit=20, label_smoothing_factor=args.ls_factor)
        print(training_args)
        
        train_dataloader = torch.utils.data.DataLoader(tokenized_squad['train'], batch_size=16, shuffle=True)
        valid_dataloader = torch.utils.data.DataLoader(tokenized_squad['validation'], batch_size=16, shuffle=True)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        num_training_steps = args.num_of_epochs * len(train_dataloader)
        scheduler = transformers.get_scheduler('linear',optimizer,num_warmup_steps=0,num_training_steps=num_training_steps)
        
        train_model(model, tokenizer, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, device, args.save_path, num_epochs=args.num_of_epochs)


    if args.evaluate_model:
        split = 'validation'
        if evaluate_path:
            tokenizer = AutoTokenizer.from_pretrained(evaluate_path,use_fast=True)
            eval_model = AutoModelForQuestionAnswering.from_pretrained(evaluate_path)
            predicted_answers,references = evaluate_question_answering(eval_model,split=split)
            squad_metric = load('squad')
            results = squad_metric.compute(predictions=predicted_answers, references=references)
            with open(evaluate_path+os.sep+args.choice+'_'+split+'_result.json', 'w') as f:
                json.dump(results, f)
            print(results)
        
    if args.mix_training:
        language_list = ['ar','de','es','hi','vi'] #,
        
        num_of_epochs = args.num_of_epochs
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained(path,use_fast=True)
        squad = load_dataset('squad')
        
        if mix_strategy == 'original_mix':
            # Get datasets
            train_dataset = load_dataset('squad', split='train')
            valid_dataset = load_dataset('squad', split='validation')
            
            for lang in tqdm(language_list):            
                other_dataset = load_dataset('json', data_files={'train':'./data/xquad_translated_train_all.json', 'validation':'./data/xquad_translated_dev_all.json'}, field=lang, features=train_dataset.features)
                temp_dataset = other_dataset['train'].select(range(0,args.min_number))
                
                train_dataset = concatenate_datasets([train_dataset, temp_dataset])
                
        elif mix_strategy == 'select_same_mix':
            # Get datasets
            temp_dataset = load_dataset('json', data_files={'train':'./data/xquad_translated_train_all.json', 'validation':'./data/xquad_translated_dev_all.json'}, field='en', features=squad['train'].features)
            train_dataset = temp_dataset['train'].select(range(0,args.min_number))
            valid_dataset = load_dataset('squad', split='validation')
            
            for lang in tqdm(language_list):            
                other_dataset = load_dataset('json', data_files={'train':'./data/xquad_translated_train_all.json', 'validation':'./data/xquad_translated_dev_all.json'}, field=lang, features=train_dataset.features)
                temp_dataset = other_dataset['train'].select(range(0,args.min_number))
                train_dataset = concatenate_datasets([train_dataset, temp_dataset])
                print(temp_dataset[0])
                
        elif mix_strategy == 'select_split_mix':
            step = int(59574/(len(language_list)+1))
            print(step)
            temp_dataset = load_dataset('json', data_files={'train':'./data/xquad_translated_train_all.json', 'validation':'./data/xquad_translated_dev_all.json'}, field='en', features=squad['train'].features)
            train_dataset = temp_dataset['train'].select(range(0,step))
            valid_dataset = load_dataset('squad', split='validation')
            
            for idx, lang in tqdm(enumerate(language_list)):            
                other_dataset = load_dataset('json', data_files={'train':'./data/xquad_translated_train_all.json', 'validation':'./data/xquad_translated_dev_all.json'}, field=lang, features=train_dataset.features)
                temp_dataset = other_dataset['train'].select(range((idx+1)*step,(idx+2)*step))
                train_dataset = concatenate_datasets([train_dataset, temp_dataset])
            
        elif mix_strategy == 'select_no_mix':
            temp_dataset = load_dataset('json', data_files={'train':'./data/xquad_translated_train_all.json', 'validation':'./data/xquad_translated_dev_all.json'}, field='en', features=squad['train'].features)
            train_dataset = temp_dataset['train'].select(range(0,args.min_number))
            valid_dataset = load_dataset('squad', split='validation')
            
        elif mix_strategy == 'original_no_mix':
            train_dataset = load_dataset('squad', split='train')
            valid_dataset = load_dataset('squad', split='validation')

            #for temp_data in tqdm(temp_dataset):
            #    train_dataset = train_dataset.add_item(temp_data)
        
        print(train_dataset)
        
        tokenized_squad = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
        tokenized_squad.set_format("torch")
        
        tokenized_valid = valid_dataset.map(preprocess_function, batched=True, remove_columns=valid_dataset.column_names)
        tokenized_valid.set_format("torch")
        
        model = AutoModelForQuestionAnswering.from_pretrained(path)
        print(model)
        
        save_path = args.save_path + '-mix-learning-'+args.mix_strategy+' '+str(args.min_number)
        
        criterion = nn.CrossEntropyLoss()
        
        train_dataloader = torch.utils.data.DataLoader(tokenized_squad, batch_size=32, shuffle=True)
        valid_dataloader = torch.utils.data.DataLoader(tokenized_valid, batch_size=16, shuffle=False)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        num_training_steps = num_of_epochs * len(train_dataloader)
        scheduler = transformers.get_scheduler('linear',optimizer,num_warmup_steps=0,num_training_steps=num_training_steps)
        train_model(model, tokenizer, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, device, save_path, num_epochs=num_of_epochs)
        
        with open(save_path +os.sep + 'mix-training-config.json', 'w') as file:
            json.dump({'epoch':num_of_epochs, 'model_path':path, 'lang_list': language_list, 'save_path':save_path, 'mix_training':args.mix_training, 'fewshot-size':args.min_number, 'mix-strategy':args.mix_strategy}, file)
        file.close()  
