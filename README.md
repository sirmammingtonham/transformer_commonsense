# transformer_sound_similarity

## Creating environment
Run `conda env create -f environment.yml`, then `conda activate story_cloze` to set the env.

`convert_commonsense.py` and `convert_storycloze.py` are for pre-processing. To run, create a folder `baseline_texts` in the root directory with the base text files and run `python convert_commonsense.py cat` for the category task or replace `cat` with `imp` for the importance task.

## Running classification tasks
Run `python train_baselines_classification.py --model_name_or_path $MODEL --task_name $TASK --do_train --do_eval --do_predict --max_seq_length 128 --per_device_train_batch_size 16 --learning_rate 5e-6 --num_train_epochs 18 --output_dir $OUTPUT --evaluation_strategy epoch --logging_steps 10 --save_steps -1 --overwrite_output_dir --fp16`
Where $MODEL is a huggingface transformer model (we used bert-large-cased, roberta-large, and microsoft/deberta-large), $TASK is the task ("storycloze_prediction" for the baseline on the original storycloze, "commonsense_category_prediction" for the category classification task, and "commonsense_importance_prediction" for the importance multilabel classification), and $OUTPUT is the output directory where the weights get saved to.

Even though these didn't make it into the paper, if you want to run the generation that I experimented with you can run
`python train_baselines_generation.py --model_name_or_path $MODEL --task_name $TASK --do_train --do_eval --do_predict --output_dir $OUTPUT --num_train_epochs 12 --per_device_train_batch_size 8 --block_size 128 --learning_rate 5e-5 --evaluation_strategy epoch --logging_steps 10 --save_steps -1 --overwrite_output_dir --fp16` where $MODEL is the same as before and $TASK is either "storycloze_ending_generation" for generating the endings of the original storycloze dataset, or "category_ending_generation" for generating endings of the commonsense dataset.

For the multitask that I experimented with, run
`python train_baselines_multitask.py --model_name_or_path $MODEL --task_name $TASK --do_train --do_eval --do_predict --per_device_train_batch_size 8 --learning_rate 5e-6 --num_train_epochs 24 --output_dir $OUTPUT --evaluation_strategy no --logging_steps 5 --save_steps -1 --overwrite_output_dir --fp16 --max_seq_length 128` where model is the same as before (or you can use gpt2 as well), and $TASK is either commonsense_category_prediction or commonsense_importance_prediction.
