from datasets import load_dataset
import argparse
from tqdm import tqdm
import torch
import json
import numpy as np
import collections
import argparse
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, AutoModel
import os
import pandas as pd
from temperature_scaling import set_temperature
from utils import compute_softmax_score_generative, compute_softmax_score_extractive, normalize_answer
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

#dump the information needed for temperature scaling for extractive model such as mBERT or XLM
def extractive_temperature_scaling(save_folder, dataset_name, dataset_split, model_name):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
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
            answer = examples["answers"][sample_index]
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

    
    model  = model.to(device)
    model.eval()
    
    start_logits_list=[]
    end_logits_list = []
    start_labels = []
    end_labels = []
    
    total = 0
    
    if dataset_name == "tydiqa":
        squad =  load_dataset('json', data_files={'train':'./data/tydiqa-processed-train.json','validation':'./data/tydiqa-processed-dev.json'}, field=dataset_split)
        split = "train"
    elif dataset_name == 'mlqa':
        squad = load_dataset(dataset_name, 'mlqa.'+dataset_split+'.'+dataset_split)
        split = "validation"
    elif dataset_name == 'squad':
        squad = load_dataset(dataset_name, dataset_split)
        split = "validation"
    else:
        squad = load_dataset(dataset_name, dataset_split)
        split = "validation"
        
    for i in tqdm(range(0,len(squad[split]),100)):
        small_eval_set = squad[split].select(range(i,min(len(squad[split]),i+100)))
        eval_set = small_eval_set.map(preprocess_validation_examples,batched=True,remove_columns=small_eval_set.column_names,)
        eval_set_for_model = eval_set.remove_columns(["example_id", "offset_mapping"])
        eval_set_for_model.set_format("torch")
        
        batch = {k: eval_set_for_model[k].to(device) for k in eval_set_for_model.column_names}
        with torch.no_grad():
            predictions =model(**batch)

        start_logits = predictions.start_logits.cpu()
        end_logits = predictions.end_logits.cpu()
        example_to_features = collections.defaultdict(list)
        start_labels_features = collections.defaultdict()
        end_labels_features = collections.defaultdict()
        offset_list = np.array(eval_set["offset_mapping"])

        for idx, feature in enumerate(eval_set):
            example_to_features[feature["example_id"]].append(idx)
            start_labels_features[feature["example_id"]] = feature["start_positions"]
            end_labels_features[feature["example_id"]] = feature["end_positions"]

        for example in small_eval_set:
            total += 1
            example_id = example["id"]

            
            for feature_index in example_to_features[example_id]:
                
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = offset_list[feature_index]
                
                start_logits_list.append(start_logit.tolist())
                end_logits_list.append(end_logit.tolist())
                
                start_labels.append(start_labels_features[example_id])
                end_labels.append(end_labels_features[example_id])
        
    #store the logits for temperature scaling
    save_file = save_folder + os.sep + model_name.split('/')[-1] + '_' +dataset_name + '_' +dataset_split + '_temperature_scaling_logits.json'
    
    with open(save_file, 'w') as json_file:
        json.dump({'start_logits': start_logits_list , 'start_labels': start_labels, 'end_logits':end_logits_list, 'end_labels': end_labels}, json_file)
    

# set the temperature using temperature scaling method 
def set_extractive_temperature(extractive_logit_file):
    
    with open(extractive_logit_file, 'r') as json_file:
        data = json.load(json_file)
    
    start_logits_list = data['start_logits']
    end_logits_list = data['end_logits']
    start_labels = data['start_labels']
    end_labels = data['end_labels']
    
    start_logits_list = torch.FloatTensor(start_logits_list)
    end_logits_list = torch.FloatTensor(end_logits_list)
    start_labels = torch.LongTensor(start_labels)
    end_labels = torch.LongTensor(end_labels)
    
    start_temp = set_temperature(start_logits_list, start_labels)
    end_temp = set_temperature(end_logits_list, end_labels)
    
    print("start temperature is set to {} end temperature is set to {} ".format(start_temp, end_temp))
    
    #store the temperature into a file
    result_file = 'extractive_result' + os.sep + 'temperature_'+extractive_logit_file.split('/')[-1].replace('_temperature_scaling_logits','')
    
    with open(result_file, 'w') as json_file:
        json.dump({'start_temp': start_temp , 'end_temp': end_temp, 'extractive_logit_file':extractive_logit_file}, json_file)
    
    return start_temp, end_temp

# set the temperature for merging all mlqa dataset using temperature scaling method 
def set_extractive_temperature_merge(save_folder, model_name, dataset_name):
    
    if dataset_name == 'mlqa':
        lang_list = ['en', 'ar', 'de','es','hi','vi','zh']
    elif dataset_name == 'tydiqa':
        lang_list = ['english','arabic','indonesian','russian','telugu']
    else:
        print("Dataset is not valid for pooled temperature scaling")
        return 0
    
    start_logits_list = []
    end_logits_list = []
    start_labels = []
    end_labels = []
    
    for lang in lang_list:
        extractive_logit_file = save_folder + os.sep + model_name.split('/')[-1] + '_' +dataset_name + '_' +lang + '_temperature_scaling_logits.json'
        with open(extractive_logit_file, 'r') as json_file:
            data = json.load(json_file)
        
        start_logits_list = start_logits_list + data['start_logits']
        end_logits_list = end_logits_list + data['end_logits']
        start_labels = start_labels + data['start_labels']
        end_labels = end_labels + data['end_labels']
    
    start_logits_list = torch.FloatTensor(start_logits_list)
    end_logits_list = torch.FloatTensor(end_logits_list)
    start_labels = torch.LongTensor(start_labels)
    end_labels = torch.LongTensor(end_labels)
    
    start_temp = set_temperature(start_logits_list, start_labels)
    end_temp = set_temperature(end_logits_list, end_labels)
    
    print("start temperature is set to {} end temperature is set to {} ".format(start_temp, end_temp))
    
    #store the temperature into a file
    result_file = 'extractive_result' + os.sep + 'temperature_'+model_name.split('/')[-1] + '_' +dataset_name + '_' + 'merge' + '.json'
    
    with open(result_file, 'w') as json_file:
        json.dump({'start_temp': start_temp , 'end_temp': end_temp, 'dataset_name':dataset_name + '_merge'}, json_file)
    
    return start_temp, end_temp

# dump the extractive logits (temperature scaling happens here)
def dump_extractive_logits_info(save_folder, dataset_name, dataset_split, source_lang,  model_name, n_best=20,max_answer_length=30, start_temp = 1.0, end_temp=1.0):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    
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
            answer = examples["answers"][sample_index]
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

    model  = model.to(device)

    model.eval()
    predicted_answers=[]
    references=[]
    
    saved_processed_answers = []
    
    if dataset_name == "tydiqa":
        squad =  load_dataset('json', data_files={'train':'./data/tydiqa-processed-train.json','validation':'./data/tydiqa-processed-dev.json'}, field=dataset_split)
        split = "validation"
    elif dataset_name == 'mlqa':
        squad = load_dataset(dataset_name, 'mlqa.'+dataset_split+'.'+dataset_split)
        split = "test"
    elif dataset_name == 'squad':
        squad = load_dataset(dataset_name)
        split = "validation"
    elif dataset_name == 'xquad':
        squad = load_dataset(dataset_name, 'xquad.'+dataset_split)
        split = "validation"
    else:
        squad = load_dataset(dataset_name, dataset_split)
        split = "validation"
        
    total = 0
        
    for i in tqdm(range(0,len(squad[split]),50)):
        small_eval_set = squad[split].select(range(i,min(len(squad[split]),i+50)))
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

        for example in small_eval_set:
            total += 1
            example_id = example["id"]
            context = example["context"]
            answers = []
            start_logits
            
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
                                "logit_score": start_logit[start_index]/start_temp + end_logit[end_index]/end_temp,
                            }
                        )
            
            if not answers:
                answers.append({
                                "text": "null",
                                "logit_score": 0.0,
                            })

            best_answer = sorted(answers, key=lambda x: x["logit_score"], reverse=True)
            
            answers = answers[0:n_best]
            saved_processed_answers.append(answers)
            references.append({"id":example_id,"answers":example["answers"]})
            #print(example['answers'], best_answer['text'])
            torch.cuda.empty_cache()
            
        # dump the raw answers, logits and gold references into file 
    save_file = save_folder + os.sep + model_name.split('/')[-1] + '_' + dataset_name + '_' + dataset_split +  '_' + source_lang + '_' + str(start_temp) + '_' + str(end_temp) + '.json'
    with open(save_file, 'w') as json_file:
        json.dump({'references': references , 'answers': saved_processed_answers}, json_file)
            
# compute ECE for extractive model with pre-dumped logits
def compute_extractive_ECE(save_folder, dataset_name, dataset_split, source_lang,  model_name,  start_temp = 1.0, end_temp=1.0):
    
    if not os.path.exists('extractive_result' + os.sep + dataset_name):
        os.mkdir('extractive_result' + os.sep +dataset_name)
    
    if not os.path.exists('extractive_result' + os.sep + dataset_name + os.sep + dataset_split):
        os.mkdir('extractive_result' + os.sep +dataset_name + os.sep + dataset_split)
    
    
    #load the logits from saved file
    save_file = save_folder + os.sep + model_name.split('/')[-1] + '_' + dataset_name + '_' + dataset_split+ '_' + source_lang + '_' + str(start_temp) + '_' + str(end_temp) + '.json'
    if os.path.exists(save_file):
        with open(save_file, 'r') as json_file:
            data = json.load(json_file)
    else:
        print("Please generate outputs first")
        return 0
        
    answers = data['answers']
    references = data['references']
    
    
    accuracy_list = []
    confidence_list = []
    
    bin_gap = 0.1
    total_bin = 10
    
    orig_acc_bins = { i*bin_gap:[] for i in range(total_bin)}
    orig_conf_bins = { i*bin_gap:[] for i in range(total_bin)}

    total = 0
    corr = 0
    
    predicted_answers = []
    
    for answer, reference in zip(answers, references):
        total += 1
        processed_answers = compute_softmax_score_extractive(answer)
        best_answer = max(processed_answers, key=lambda x: x["softmax"])
        conf = best_answer['softmax']
        predicted_answers.append({"prediction_text": best_answer["text"]})
        #print(example['answers'], best_answer['text'])
        torch.cuda.empty_cache()
        
        correct = False
        for i in orig_conf_bins.keys():
            if (conf >= float(i)) and (conf < (float(i) + bin_gap)):
                orig_conf_bins[i].append(conf)
                confidence_list.append(conf)
                for gold_answer in reference['answers']['text']:
                    if gold_answer == best_answer['text']:
                        #print(gold_answer,best_answer['text'],conf)
                        correct = True
                        orig_acc_bins[i].append(1)
                        accuracy_list.append(1)
                        corr += 1
                        break
                if not correct:
                    #print(example['answers']['text'],best_answer['text'],conf)
                    orig_acc_bins[i].append(0)
                    accuracy_list.append(0)

    acc = corr/total
    ECE_ori = 0
    ece = 0
    for i in orig_conf_bins.keys():
        conf_list = orig_conf_bins[i]
        acc_list = orig_acc_bins[i]
        if len(conf_list) >  0:
            ECE_ori += len(conf_list)/total * abs(sum(conf_list)/len(conf_list) - sum(acc_list)/len(acc_list))

    assert (len(accuracy_list) == len(confidence_list)) 
 
    print("result summary", acc*100, ECE_ori*100, ece)
    
    result_dict = {'accuracy':acc*100,'ece':ECE_ori*100, 'start_temp': start_temp, 'end_temp': end_temp, 'dataset_name':dataset_name, 'lang':dataset_split, 'source_lang': source_lang, 'model': model_name}
    result_file = 'extractive_result' + os.sep + dataset_name + os.sep + dataset_split + os.sep + model_name.split('/')[-1] + '_' + source_lang + '_' + str(start_temp) + '_' + str(end_temp) +'.json'
    
    with open(result_file, 'w') as json_file:
        json.dump({'result': result_dict }, json_file)

    return acc,ECE_ori

#dump the information needed for temperature scaling for generative model such as mT5, mBART
def generative_temperature_scaling(save_folder, dataset_name, dataset_split, model_name):
    answers = []
    
    logits_list = []
    labels_list = []
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    model.to("cuda")
    
    def add_eos_to_examples(example):
        example['input_text'] = 'question: %s context: %s ' % (example['question'], example['context'])
        example['target_text'] = '%s ' % example['answers']['text'][0]
        
        return example
    
    def convert_to_features(example_batch):
        input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'], padding="max_length", max_length=512, truncation=True)
        target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], padding="max_length", max_length=64, truncation=True)
        
        
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
    
    if dataset_name == "tydiqa":
        valid_dataset =  load_dataset('json', data_files={'train':'./data/tydiqa-processed-train.json','validation':'./data/tydiqa-processed-dev.json'}, field=dataset_split)['train']
    elif dataset_name == 'mlqa':
        valid_dataset = load_dataset(dataset_name, 'mlqa.'+dataset_split+'.'+dataset_split, split="validation")
    else:
        valid_dataset = load_dataset(dataset_name, dataset_split, split="validation")
    valid_dataset = valid_dataset.map(add_eos_to_examples, load_from_cache_file=False)
    valid_dataset = valid_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)
    
    if (dataset_name == "mlqa") or (dataset_name == "xquad") :
        valid_dataset_processed = valid_dataset.remove_columns(['id', 'context', 'question', 'answers', 'input_text', 'target_text']) #title
    elif (dataset_name == "tydiqa") or (dataset_name == "squad"):
        valid_dataset_processed = valid_dataset.remove_columns(['id', 'context', 'question', 'answers', 'input_text', 'target_text','title']) 
    else:
        #valid_dataset_processed = valid_dataset.remove_columns(['id', 'context', 'answers', 'question', 'title', 'input_text', 'target_text', 'input_ids', 'attention_mask', 'labels', 'decoder_attention_mask'])
        valid_dataset_processed = valid_dataset.remove_columns(['id', 'info', 'question', 'answerKey','input_text','target_text','title']) 
    #valid_dataset_processed = valid_dataset.remove_columns(['id', 'info', 'question', 'answerKey','input_text','target_text']) 
    valid_dataset_processed = valid_dataset_processed
    valid_dataset_processed.set_format('torch')
    
    references = []

    corr = 0
    total = 0
    
    for idx, item in tqdm(enumerate(valid_dataset_processed)):
        
        logits = []
        
        total += 1
        answers = []
        batch = {k: item[k].unsqueeze(0).to("cuda:0") for k in item.keys()}
        outputs = model.generate(input_ids=batch['input_ids'], 
                                attention_mask=batch['attention_mask'],
                                max_length=64,
                                length_penalty = 0.0,
                                num_beams=20,
                                num_return_sequences=20,
                                early_stopping=True,
                                return_dict_in_generate=True,
                                output_scores=True)

        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True
        )  
        reconstructed_scores = transition_scores.sum(axis=1)
        prob_scores = reconstructed_scores

        #outs = [tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)]
        
        for seq, prob_score in zip(outputs.sequences, prob_scores):
            text = tokenizer.decode(seq, skip_special_tokens=True)
            answers.append({'text':text,'prob':prob_score.detach().cpu().item()})
            #print(text,prob_score.detach().cpu().item())

        
        example = valid_dataset[idx]
        example_id = example['id']
        references.append({"id":example_id,"answers":example["answers"]})
            #print(example['answers'], best_answer['text'])
            
        correct = False
        gold_label = -1
        #filter sentences
        for answer_idx, post_answer in enumerate(answers):
            logits.append(post_answer['prob'])
            if not correct:
                #check whether any of the sentence is in the golden answer
                for gold_answer in example['answers']['text']:
                    if normalize_answer(gold_answer) == normalize_answer(post_answer['text']):
                        print('correct answers: {}, predicted: {} '.format(gold_answer, post_answer))
                        correct = True
                        gold_label = answer_idx
                        corr += 1
                        break
        
        if correct:
            logits_list.append(logits)
            labels_list.append(gold_label)
    
    save_file = save_folder + os.sep + model_name.split('/')[-1] + '_' + dataset_name + '_' + dataset_split + '_temperature_scaling_logits.json'
        
    with open(save_file, 'w') as json_file:
        json.dump({'logits': logits_list , 'labels': labels_list}, json_file)
    
    print("Logits for temperature scaling saved to ", save_file)

# set the temperature using temperature scaling method 
def set_generative_temperature(generative_logit_file):
    
    with open(generative_logit_file, 'r') as json_file:
        data = json.load(json_file)
        
    logits_list = data['logits']
    labels_list = data['labels']
    
    logits_list = torch.FloatTensor(logits_list)
    labels_list = torch.LongTensor(labels_list)
    temp = set_temperature(logits_list, labels_list)
    
    print("tempature is set to ", temp)
    result_file = 'generative_result' + os.sep + 'temperature_'+generative_logit_file.split('/')[-1].replace('_temperature_scaling_logits','')
    
    with open(result_file, 'w') as json_file:
        json.dump({'temperature': temp , 'generative_logit_file':generative_logit_file}, json_file)
    
    return temp


# set the temperature using temperature scaling method for merged dataset
def set_generative_temperature_merge(save_folder, model_name, dataset_name):
    
    if dataset_name == 'mlqa':
        lang_list = ['en', 'ar', 'de','es','hi','vi','zh']
    elif dataset_name == 'tydiqa':
        lang_list = ['english','arabic','indonesian','russian','telugu']
    else:
        print("Dataset is not valid for pooled temperature scaling")
        return 0
    
    logits_list = []
    labels_list = []
    
    for lang in lang_list:
        generative_logit_file = save_folder + os.sep + model_name.split('/')[-1] + '_' +dataset_name + '_' + lang + '_temperature_scaling_logits.json'
        with open(generative_logit_file, 'r') as json_file:
            data = json.load(json_file)
            
        logits_list = logits_list+data['logits']
        labels_list = labels_list+data['labels']
    
    logits_list = torch.FloatTensor(logits_list)
    labels_list = torch.LongTensor(labels_list)
    temp = set_temperature(logits_list, labels_list)
    
    print("tempature is set to ", temp)
    result_file = 'generative_result' + os.sep + 'temperature_'+model_name.split('/')[-1] + '_' +dataset_name + '_' + 'merge' + '.json'
    
    with open(result_file, 'w') as json_file:
        json.dump({'temperature': temp , 'dataset_name':dataset_name+'_merge'}, json_file)
    
    return temp

# dump logits information of generative model 
def dump_generative_logits_info(save_folder, dataset_name, dataset_split, model_name):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    model.to("cuda")
    
    def add_eos_to_examples(example):
        example['input_text'] = 'question: %s context: %s ' % (example['question'], example['context'])
        example['target_text'] = '%s ' % example['answers']['text'][0]
        
        return example
    
    def convert_to_features(example_batch):
        input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'], padding="max_length", max_length=512, truncation=True)
        target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], padding="max_length", max_length=64, truncation=True)
        
        
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
    
    if dataset_name == "tydiqa":
        valid_dataset =  load_dataset('json', data_files={'train':'./data/tydiqa-processed-train.json','validation':'./data/tydiqa-processed-dev.json'}, field=dataset_split)['validation']
    elif dataset_name == 'mlqa':
        valid_dataset = load_dataset(dataset_name, 'mlqa.'+dataset_split+'.'+dataset_split, split="test")
    elif dataset_name == 'xquad':
        valid_dataset = load_dataset(dataset_name, 'xquad.'+dataset_split, split="validation")
    elif dataset_name == 'squad':
        valid_dataset = load_dataset(dataset_name, dataset_split, split="validation")
    else:
        valid_dataset = load_dataset(dataset_name, dataset_split, split="validation")
    #valid_dataset = load_dataset(dataset_name, split, split="validation")
    valid_dataset_saved = valid_dataset.map(add_eos_to_examples, load_from_cache_file=False)
    valid_dataset_saved = valid_dataset_saved.map(convert_to_features, batched=True, load_from_cache_file=False)
    
    if (dataset_name == "mlqa") or (dataset_name == "xquad") :
        valid_dataset_processed = valid_dataset_saved.remove_columns(['id', 'context', 'question', 'answers', 'input_text', 'target_text']) #title
    elif (dataset_name == "tydiqa") or (dataset_name == "squad"):
        valid_dataset_processed = valid_dataset_saved.remove_columns(['id', 'context', 'question', 'answers', 'input_text', 'target_text','title']) 
    else:
        #valid_dataset_processed = valid_dataset.remove_columns(['id', 'context', 'answers', 'question', 'title', 'input_text', 'target_text', 'input_ids', 'attention_mask', 'labels', 'decoder_attention_mask'])
        valid_dataset_processed = valid_dataset_saved.remove_columns(['id', 'info', 'question', 'answerKey','input_text','target_text','title']) 
    #valid_dataset_processed = valid_dataset.remove_columns(['id', 'info', 'question', 'answerKey','input_text','target_text']) 
    valid_dataset_processed = valid_dataset_processed
    valid_dataset_processed.set_format('torch')
    

    saved_processed_answers=[]
    stored_inputs = []
    references = []

    total = 0
    
    for idx, item in tqdm(enumerate(valid_dataset_processed)):
        total += 1
        answers = []
        batch = {k: item[k].unsqueeze(0).to("cuda:0") for k in item.keys()}
        print(batch['input_ids'].shape)
        outputs = model.generate(input_ids=batch['input_ids'], 
                                #attention_mask=batch['attention_mask'],
                                length_penalty = 0.0,
                                max_length=64,
                                num_beams=20,
                                num_return_sequences=20,
                                early_stopping=True,
                                return_dict_in_generate=True,
                                output_scores=True)

        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True
        )  
        reconstructed_scores = transition_scores.sum(axis=1)
        prob_scores = reconstructed_scores
        #print(prob_scores,torch.exp(transition_scores))
        #outs = [tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)]
        
        for seq, prob_score in zip(outputs.sequences, prob_scores):
            text = tokenizer.decode(seq, skip_special_tokens=True)
            answers.append({'text':text,'logit_score':prob_score.detach().cpu().item()})
            #print(text,prob_score.detach().cpu().item())
        saved_processed_answers.append(answers)
        example = valid_dataset[idx]
        example_id = example['id']
        stored_inputs.append(example)
        references.append({"id":example_id,"answers":example["answers"]})
    
    
    # dump the raw answers, logits and gold references into file 
    save_file = save_folder + os.sep + model_name.split('/')[-1] + '_' + dataset_name + '_' + dataset_split + '.json'
    with open(save_file, 'w') as json_file:
        json.dump({'references': references , 'answers': saved_processed_answers, 'inputs':stored_inputs }, json_file)
        
        
# compute the ECE with dumped logits information (generative model)
def compute_generative_ECE(save_folder, dataset_name, dataset_split, source_lang, model_name, temperature=1.0):
    
    if not os.path.exists('generative_result' + os.sep + dataset_name):
        os.mkdir('generative_result' + os.sep + dataset_name)
    
    if not os.path.exists('generative_result' + os.sep + dataset_name + os.sep + dataset_split):
        os.mkdir('generative_result' + os.sep + dataset_name + os.sep + dataset_split)
    
    #load the logits from saved file
    save_file = save_folder + os.sep + model_name.split('/')[-1] + '_' + dataset_name + '_' + dataset_split + '.json'
    if os.path.exists(save_file):
        with open(save_file, 'r') as json_file:
            data = json.load(json_file)
    else:
        print("Please generate outputs first")
        return 0
    
    answers = data['answers']
    references = data['references']

    confidence_list = []
    acuracy_list = []
    
    bin_gap = 0.1
    total_bin = 10
    
    orig_acc_bins = { i*bin_gap:[] for i in range(total_bin)}
    orig_conf_bins = { i*bin_gap:[] for i in range(total_bin)}
    
    corr = 0
    total = 0

    
    for answer, reference in zip(answers, references): 
        
        total += 1
        
        processed_answers = compute_softmax_score_generative(answer, temperature)
        best_answer = max(processed_answers, key=lambda x: x['prob'])
        conf = best_answer['prob']
        
        correct = False
        for i in orig_conf_bins.keys():
            if (conf >= float(i)) and (conf < (float(i) + bin_gap)):
                orig_conf_bins[i].append(conf)
                confidence_list.append(conf)
                for gold_answer in reference['answers']['text']:
                    if normalize_answer(gold_answer) == normalize_answer(best_answer['text']):
                        print('correct answers: {}, predicted: {}, conf :{} '.format(gold_answer, best_answer['text'], conf))
                        correct = True
                        orig_acc_bins[i].append(1)
                        acuracy_list.append(1)
                        corr += 1
                        break
                if not correct:
                    print('wrong answers: {}, predicted: {}, conf :{} '.format(gold_answer, best_answer['text'], conf))
                    orig_acc_bins[i].append(0)
                    acuracy_list.append(0)

    acc = corr/total
    ECE_ori = 0
    ece = 0
    for i in orig_conf_bins.keys():
        conf_list = orig_conf_bins[i]
        acc_list = orig_acc_bins[i]
        if len(conf_list) >  0:
            ECE_ori += len(conf_list)/total * abs(sum(conf_list)/len(conf_list) - sum(acc_list)/len(acc_list))
            
    assert (len(acuracy_list) == len(confidence_list))           
    #ece = compute_ece(evaluate_path,acuracy_list,confidence_list, True, dataset_name = dataset_name, split=split)
    print("result summary", acc*100, ECE_ori*100, ece)
    
    result_dict = {'accuracy':acc*100,'ece':ECE_ori*100, 'temp': temperature, 'dataset': dataset_name, 'lang':dataset_split, 'model': model_name, 'source_lang': source_lang}
    
    result_file = 'generative_result' + os.sep + dataset_name + os.sep + dataset_split + os.sep + model_name.split('/')[-1] + '_' + source_lang + '_'+ str(temperature) +'.json'
    
    with open(result_file, 'w') as json_file:
        json.dump({'result': result_dict }, json_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess qamr file into huggingface format')
    parser.add_argument('--evaluate_model_name',type=str, default="roberta-base", help='path for reading pretrained model')
    parser.add_argument('--dataset_name',type=str, default='xquad', help='the name of the dataset')
    parser.add_argument('--lang',type=str, default='en', help='the language we are interested')
    parser.add_argument('--source_lang',type=str, default='en', help='source language for running temperature scaling')
    parser.add_argument('--split',type=str, default='validation', help='whether to train a new question answering model')
    parser.add_argument('--save_folder',type=str, default='./results', help='folder for dump logits for validation set')
    parser.add_argument('--output_folder',type=str, default='./results', help='folder for dump outputs')
    
    # dump results for generative model and compute ECE for extractive models 
    parser.add_argument('--dump_validation_extractive',type=bool, default=False, help='dump logits for temperature scaling (extractive model)')
    parser.add_argument('--run_ts_extractive',type=bool, default=False, help='run temperature scaling for extractive model')
    parser.add_argument('--dump_extractive_results',type=bool, default=False, help='dump outputs for test data')
    parser.add_argument('--compute_extractive_ECE',type=bool, default=False, help='compute ECE for extractive model')
    parser.add_argument('--start_temp',type=float, default=1.0, help='temperature for start index')
    parser.add_argument('--end_temp',type=float, default=1.0, help='temperature for end index')
    
    # dump results for generative model and compute ECE for generative models 
    parser.add_argument('--dump_validation_generative',type=bool, default=False, help='dump logits for temperature scaling (generative model)')
    parser.add_argument('--run_ts_generative',type=bool, default=False, help='run temperature scaling for generative model')
    parser.add_argument('--dump_generative_results',type=bool, default=False, help='dump outputs for test data')
    parser.add_argument('--compute_generative_ECE',type=bool, default=False, help='compute ECE for generative model')
    parser.add_argument('--temp',type=float, default=1.0, help='temperature')
    
    parser.add_argument('--ts_enabled',action='store_true', help='compute ECE for generative model')
    
    # run baseline temperatures 
    parser.add_argument('--run_ts_extractive_merge',type=bool, default=False, help='run temperature scaling for extractive model on merged data')
    parser.add_argument('--run_ts_generative_merge',type=bool, default=False, help='run temperature scaling for generative model on merged data')
    
    args = parser.parse_args()
    evaluate_path = args.evaluate_model_name
    
    # code for running temperature scaling for extractive models
    if args.dump_validation_extractive:
        extractive_temperature_scaling(args.save_folder, args.dataset_name, args.lang, args.evaluate_model_name)
        
    if args.run_ts_extractive:
        extractive_logit_file = args.save_folder + os.sep + args.evaluate_model_name.split('/')[-1] + '_' +args.dataset_name + '_' +args.lang + '_temperature_scaling_logits.json'
        set_extractive_temperature(extractive_logit_file)
    
    if args.run_ts_extractive_merge:
        set_extractive_temperature_merge(args.save_folder, args.evaluate_model_name, args.dataset_name)
        
    if args.dump_extractive_results:
        if args.ts_enabled:
            #load the temperature from  a file
            if args.source_lang == 'squad':
                extractive_logit_file = args.save_folder + os.sep + args.evaluate_model_name.split('/')[-1] + '_' + 'squad' + '_' + 'plain_text' + '.json'
            elif args.source_lang == 'merge':
                if (args.dataset_name == 'xquad') or (args.dataset_name == 'tydiqa'):
                    extractive_logit_file = args.save_folder + os.sep + args.evaluate_model_name.split('/')[-1] + '_' + 'mlqa' + '_' + 'merge' + '.json'
                else:
                    extractive_logit_file = args.save_folder + os.sep + args.evaluate_model_name.split('/')[-1] + '_' + args.dataset_name + '_' + 'merge' + '.json'
            else:    
                if (args.dataset_name == 'xquad'):
                    extractive_logit_file = args.save_folder + os.sep + args.evaluate_model_name.split('/')[-1] + '_' + 'mlqa' + '_' +args.source_lang + '.json'
                else:
                    extractive_logit_file = args.save_folder + os.sep + args.evaluate_model_name.split('/')[-1] + '_' + args.dataset_name + '_' +args.source_lang + '.json'
            extractive_temperature_file = 'extractive_result' + os.sep + 'temperature_'+extractive_logit_file.split('/')[-1]
            with open(extractive_temperature_file, 'r') as json_file:
                temp_data = json.load(json_file)
            start_temp = round(temp_data['start_temp'],4)
            end_temp = round(temp_data['end_temp'],4)
            json_file.close()
            dump_extractive_logits_info(args.output_folder, args.dataset_name,  args.lang, args.source_lang, args.evaluate_model_name, n_best=20,max_answer_length=30, start_temp = start_temp, end_temp=end_temp)
        else:
            dump_extractive_logits_info(args.output_folder, args.dataset_name,  args.lang, args.source_lang, args.evaluate_model_name, n_best=20,max_answer_length=30, start_temp = 1.0, end_temp=1.0)
    
    if args.compute_extractive_ECE:
        if args.source_lang == 'squad':
            extractive_logit_file = args.save_folder + os.sep + args.evaluate_model_name.split('/')[-1] + '_' + 'squad' + '_' + 'plain_text' + '.json'
        elif args.source_lang == 'merge':
            if (args.dataset_name == 'xquad') or (args.dataset_name == 'tydiqa'):
                extractive_logit_file = args.save_folder + os.sep + args.evaluate_model_name.split('/')[-1] + '_' + 'mlqa' + '_' + 'merge' + '.json'
            else:
                extractive_logit_file = args.save_folder + os.sep + args.evaluate_model_name.split('/')[-1] + '_' + args.dataset_name + '_' + 'merge' + '.json'
        else:
            if (args.dataset_name == 'xquad'):
                extractive_logit_file = args.save_folder + os.sep + args.evaluate_model_name.split('/')[-1] + '_' + 'mlqa' + '_' +args.source_lang + '.json'
            else:
                extractive_logit_file = args.save_folder + os.sep + args.evaluate_model_name.split('/')[-1] + '_' + args.dataset_name + '_' +args.source_lang + '.json'
        
        if args.ts_enabled:
            #load the temperature from  a file
            extractive_temperature_file = 'extractive_result' + os.sep + 'temperature_'+extractive_logit_file.split('/')[-1]
            if os.path.exists(extractive_temperature_file):
                with open(extractive_temperature_file, 'r') as json_file:
                    temp_data = json.load(json_file)
                start_temp = round(temp_data['start_temp'],4)
                end_temp = round(temp_data['end_temp'],4)
                json_file.close()
                #dump_extractive_logits_info(args.output_folder, args.dataset_name,  args.lang, args.evaluate_model_name, n_best=20,max_answer_length=30, start_temp = start_temp, end_temp=end_temp)
                compute_extractive_ECE(args.output_folder, args.dataset_name,  args.lang, args.source_lang, args.evaluate_model_name,  start_temp = start_temp, end_temp=end_temp)
            else:
                print("Please run temperature scaling first")
        else:
            #dump_extractive_logits_info(args.output_folder, args.dataset_name,  args.lang, args.evaluate_model_name, n_best=20,max_answer_length=30, start_temp = 1.0, end_temp=1.0)
            compute_extractive_ECE(args.output_folder, args.dataset_name,  args.lang, args.source_lang, args.evaluate_model_name,  start_temp = 1.0, end_temp=1.0)
    
    # code for running temperature scaling for generative models
    if args.dump_validation_generative:
        generative_temperature_scaling(args.save_folder, args.dataset_name, args.lang, args.evaluate_model_name)
        
    if args.run_ts_generative:
        generative_logit_file = args.save_folder + os.sep + args.evaluate_model_name.split('/')[-1] + '_' + args.dataset_name + '_' + args.lang + '_temperature_scaling_logits.json'
        set_generative_temperature(generative_logit_file)
    
    if args.run_ts_generative_merge:
        set_generative_temperature_merge(args.save_folder, args.evaluate_model_name, args.dataset_name)
        
    if args.dump_generative_results:
        dump_generative_logits_info(args.output_folder, args.dataset_name, args.lang, args.evaluate_model_name)
    
    if args.compute_generative_ECE:
        if not args.ts_enabled:
            compute_generative_ECE(args.output_folder, args.dataset_name, args.lang, args.source_lang, args.evaluate_model_name, temperature=1.0)
        else:
            
            if args.source_lang == 'squad':
                generative_temperature_file = 'generative_result' + os.sep + 'temperature_'+ args.evaluate_model_name.split('/')[-1] + '_' + 'squad' + '_' + 'plain_text' + '.json'
            elif args.source_lang == 'merge':
                if (args.dataset_name == 'xquad') or (args.dataset_name == 'tydiqa'):
                    generative_temperature_file = 'generative_result' + os.sep + 'temperature_'+ args.evaluate_model_name.split('/')[-1] + '_' +  'mlqa' + '_' + 'merge' + '.json'
                else:
                    generative_temperature_file = 'generative_result' + os.sep + 'temperature_'+ args.evaluate_model_name.split('/')[-1] + '_' +  args.dataset_name + '_' + 'merge' + '.json'
            else:
                if (args.dataset_name == 'xquad'):
                    generative_temperature_file = 'generative_result' + os.sep + 'temperature_'+ args.evaluate_model_name.split('/')[-1] + '_' + 'mlqa' + '_' + args.source_lang + '.json'
                else:
                    generative_temperature_file = 'generative_result' + os.sep + 'temperature_'+ args.evaluate_model_name.split('/')[-1] + '_' + args.dataset_name + '_' + args.source_lang + '.json'
            
            if os.path.exists(generative_temperature_file):
                with open(generative_temperature_file, 'r') as json_file:
                    temp = json.load(json_file)["temperature"]
                json_file.close()
                temp = round(temp,4)
                compute_generative_ECE(args.output_folder, args.dataset_name, args.lang, args.source_lang, args.evaluate_model_name, temperature=temp) 
            else:
                print("Please run temperature scaling first")
