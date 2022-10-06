# A Corpus for Commonsense Inference in the Story Cloze Test

This repo contains code for [A Corpus for Commonsense Inference in the Story Cloze Test](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.375.pdf) (Bingsheng Yao, Ethan Joseph, Julian Lioanag, Mei Si; In Proceedings of the 13th Conference on Language Resources and Evaluation (LREC 2022))

## Dataset
Dataset is accessible in arrow format in [baseline_data/](baseline_data) and raw text format in [baseline_texts/](baseline_texts). 

## Baselines

### Creating environment
Run `conda env create -f environment.yml`, then `conda activate story_cloze` to set the env.

`convert_commonsense.py` and `convert_storycloze.py` are for pre-processing. Run `python convert_commonsense.py cat` to convert the mturk collected txt files to a feather dataset for the category task or replace `cat` with `imp` for the importance task. (The processed datasets are already in the `baseline_data/` folder.)

### Running classification tasks
Run `python train_baselines_classification.py --model_name_or_path $MODEL --task_name $TASK --do_train --do_eval --do_predict --max_seq_length 128 --per_device_train_batch_size 16 --learning_rate 5e-6 --num_train_epochs 18 --output_dir $OUTPUT --evaluation_strategy epoch --logging_steps 10 --save_steps -1 --overwrite_output_dir --fp16`
Where $MODEL is a huggingface transformer model (we used bert-large-cased, roberta-large, and microsoft/deberta-large), $TASK is the task ("storycloze_prediction" for the baseline on the original storycloze, "commonsense_category_prediction" for the category classification task, and "commonsense_importance_prediction" for the importance multilabel classification), and $OUTPUT is the output directory where the weights get saved to.

Even though these didn't make it into the paper, if you want to run the generation that I experimented with you can run
`python train_baselines_generation.py --model_name_or_path $MODEL --task_name $TASK --do_train --do_eval --do_predict --output_dir $OUTPUT --num_train_epochs 12 --per_device_train_batch_size 8 --block_size 128 --learning_rate 5e-5 --evaluation_strategy epoch --logging_steps 10 --save_steps -1 --overwrite_output_dir --fp16` where $MODEL is the same as before and $TASK is either "storycloze_ending_generation" for generating the endings of the original storycloze dataset, or "category_ending_generation" for generating endings of the commonsense dataset.

For the multitask that I experimented with, run
`python train_baselines_multitask.py --model_name_or_path $MODEL --task_name $TASK --do_train --do_eval --do_predict --per_device_train_batch_size 8 --learning_rate 5e-6 --num_train_epochs 24 --output_dir $OUTPUT --evaluation_strategy no --logging_steps 5 --save_steps -1 --overwrite_output_dir --fp16 --max_seq_length 128` where model is the same as before (or you can use gpt2 as well), and $TASK is either commonsense_category_prediction or commonsense_importance_prediction.

## Citation
```
@inproceedings{yao-etal-2022-corpus,
    title = "A Corpus for Commonsense Inference in Story Cloze Test",
    author = "Yao, Bingsheng  and
      Joseph, Ethan  and
      Lioanag, Julian  and
      Si, Mei",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.375",
    pages = "3500--3508",
    abstract = "The Story Cloze Test (SCT) is designed for training and evaluating machine learning algorithms for narrative understanding and inferences. The SOTA models can achieve over 90{\%} accuracy on predicting the last sentence. However, it has been shown that high accuracy can be achieved by merely using surface-level features. We suspect these models may not \textit{truly} understand the story. Based on the SCT dataset, we constructed a human-labeled and human-verified commonsense knowledge inference dataset. Given the first four sentences of a story, we asked crowd-source workers to choose from four types of narrative inference for deciding the ending sentence and which sentence contributes most to the inference. We accumulated data on 1871 stories, and three human workers labeled each story. Analysis of the intra-category and inter-category agreements show a high level of consensus. We present two new tasks for predicting the narrative inference categories and contributing sentences. Our results show that transformer-based models can reach SOTA performance on the original SCT task using transfer learning but don{'}t perform well on these new and more challenging tasks.",
}
```
