# Improving Calibration for Multilingual Question Answering Model

Here we provide the instructions to run our experiments for extractive QA and generative QA as described in the paper.

## Extractive Models

## To train the model
Please specify dataset, pretrained model path, folder for saving finetuned model, number of epochs

-  ``python3 train_extractive_mlqa.py --train_model True --path xlm-roberta-base --save_path xlm-squad-0.0 --num_of_epochs 3 ``

To train a model with label smoothing, add ``--ls_factor``

### To run data augmentation via translation (need preprocessed MLQA translated dataset)
Please specify pretrained model path, folder for saving finetuned model, number of epochs, number of augmented data for each language and data augmentation strategy. For example, set path to xlm-roberta-base,  save_path prefix to be xlm-mix, mix_strategy=select_split_mix (which is corresponded to Mixed in Table 4 of section 5.1) min_number=9929, number of epochs to 3. More avaiable mix strategy options are 'original_mix','original_no_mix','select_same_mix','select_split_mix','select_no_mix'

``python3 train_extractive_mlqa.py  --mix_training True --path xlm-roberta-base --save_path xlm-mix --mix_strategy select_split_mix --min_number 9929 --num_of_epochs 3  ``


The model will be saved into *$save_path-mix-training-$mix_strategy_$min_number*

### To measure the calibration of the model
To get the calibation performance for the model, please specify the dataset, language, and the model path

> ``python3 calibration_temp_multilingual.py --dump_extractive_results True --output_folder output_logits --dataset_name xquad --lang en --evaluate_model_name ${model_path} ``

> ``python3 calibration_temp_multilingual.py --compute_extractive_ECE True --output_folder output_logits --dataset_name xquad --lang en --evaluate_model_name ${model_path}``


To run the temperate scaling (note that we have two choices for temperature scaling: *squad* and *merge*): 

For squad:

> ``python3 calibration_temp_multilingual.py --dump_validation_extractive True --save_folder temp_logits --dataset_name squad --lang plain_text --evaluate_model_name ${model_path}``

> ``python3 calibration_temp_multilingual.py --run_ts_extractive True --save_folder temp_logits --dataset_name squad --lang plain_text --evaluate_model_name ${model_path}``

For merge:

need to run dump the validation samples from all MLQA validation samples (en, vi, zh, es, de, hi)

> ``python3 calibration_temp_multilingual.py --dump_validation_extractive True --save_folder temp_logits --dataset_name mlqa --lang ${lang} --evaluate_model_name ${model_path}``

> ``python3 calibration_temp_multilingual.py --run_ts_extractive_merge True --save_folder temp_logits --dataset_name mlqa --evaluate_model_name ${model_path}``


To get the calibration performance after TS :

please specify the dataset, language, the model path, the TS choice (*squad* or *merge*)

- Extractive
> ``python3 calibration_temp_multilingual.py --dump_extractive_results True --output_folder output_logits --dataset_name xquad --lang en --evaluate_model_name ${model_name} --ts_enabled --source_lang ${ts_choice} ``

> ``python3 calibration_temp_multilingual.py --compute_extractive_ECE True --output_folder output_logits --dataset_name xquad --lang en --evaluate_model_name ${model_name} --ts_enabled --source_lang ${ts_choice} ``



## Generative Models

### To train the model

Please specify dataset, pretrained model path, folder for saving finetuned model, number of epochs

- ``python3 train_generative_mlqa.py --path google/mt5-base --save_path mt5-base-squad --num_of_epochs 5 --train_model True``


### To run data augmentation via translation (need preprocessed MLQA translated dataset)
Please specify pretrained model path, folder for saving finetuned model, number of epochs, number of augmented data for each language and data augmentation strategy. For example, set path to xlm-roberta-base,  save_path prefix to be xlm-mix, mix_strategy=select_split_mix (which is corresponded to Mixed in Table 4 of section 5.1) min_number=9929, number of epochs to 3. More avaiable mix strategy options are 'original_mix','original_no_mix','select_same_mix','select_split_mix','select_no_mix'

``python3 train_generative_mlqa.py  --mix_training True --path google/mt5-base --save_path mt5 --mix_strategy select_split_mix --min_number 9929 --num_of_epochs 5 ``


The model will be saved into *$save_path-mix-training-$mix_strategy_$min_number*

### To measure the calibration of the model
   > ``python3 calibration_temp_multilingual.py --dump_generative_results True --output_folder output_logits --dataset_name xquad --lang en --evaluate_model_name ${model_path}``

   >`` python3 calibration_temp_multilingual.py --compute_generative_ECE True --output_folder output_logits --dataset_name xquad --lang en --evaluate_model_name ${model_path} ``

To run the temperate scaling (note that we have two choices for temperature scaling: *squad* and *merge*): 

> ``python3 calibration_temp_multilingual.py --dump_validation_generative True --save_folder temp_logits --dataset_name squad --lang plain_text --evaluate_model_name ${model_path}``

> ``python3 calibration_temp_multilingual.py --run_ts_generative True --save_folder temp_logits --dataset_name squad --lang plain_text --evaluate_model_name ${model_path}``

For merge:

need to run dump the validation samples from all MLQA validation samples (en, vi, zh, es, de, hi)

> ``python3 calibration_temp_multilingual.py --dump_validation_generative True --save_folder temp_logits --dataset_name mlqa --lang ${lang} --evaluate_model_name ${model_path}``

> ``python3 calibration_temp_multilingual.py --run_ts_generative_merge True --save_folder temp_logits --dataset_name mlqa --evaluate_model_name ${model_path}``

To get the calibration performance after TS :

please specify the dataset, language, the model path, the TS choice (*squad* or *merge*)

> ``python3 calibration_temp_multilingual.py --dump_generative_results True --output_folder output_logits --dataset_name xquad --lang en --evaluate_model_name ${model_path}``

> ``python3 calibration_temp_multilingual.py --compute_generative_ECE True --output_folder output_logits --dataset_name xquad --lang en --evaluate_model_name ${model_path} --ts_enabled --source_lang ${ts_choice}``