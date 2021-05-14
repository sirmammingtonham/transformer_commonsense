#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_from_disk, DatasetDict

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process, set_seed
from transformers.utils import check_min_version
from src.trainer import CommonSenseTrainer, default_data_collator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, hamming_loss

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.6.0.dev0")

logger = logging.getLogger(__name__)
SEED = 690
THRESHOLD = 0.5
set_seed(SEED)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: storycloze_prediction', 'commonsense_category_prediction', 'commonsense_importance_prediction"},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={
                                     "help": "A csv or a json file containing the test data."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.seed = SEED
    is_multilabel = data_args.task_name == 'commonsense_importance_prediction'

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(
        training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    class_weights = None
    if data_args.task_name == 'storycloze_prediction':
        loaded_data = load_from_disk('./storycloze/storycloze_valid')
        train_valid = loaded_data.train_test_split(test_size=0.1, seed=SEED)
        test = load_from_disk('./storycloze/storycloze_test')
        datasets = DatasetDict({
            'train': train_valid['train'],
            'test': test,
            'validation': train_valid['test']}
        )
    elif data_args.task_name == 'commonsense_category_prediction':
        loaded_data = load_from_disk('./baseline_data/category')
        train_testvalid = loaded_data.train_test_split(
            test_size=0.2, seed=SEED)

        test_valid = train_testvalid['test'].train_test_split(
            test_size=0.5, seed=SEED)
        datasets = DatasetDict({
            'train': train_testvalid['train'],
            'test': test_valid['test'],
            'validation': test_valid['train']}
        )
        class_weights = compute_class_weight('balanced', classes=np.unique(
            datasets["train"]["label"]), y=datasets["train"]["label"])
    elif data_args.task_name == 'commonsense_importance_prediction':
        loaded_data = load_from_disk('./baseline_data/importance')
        train_testvalid = loaded_data.train_test_split(
            test_size=0.2, seed=SEED)
        test_valid = train_testvalid['test'].train_test_split(
            test_size=0.5, seed=SEED)
        datasets = DatasetDict({
            'train': train_testvalid['train'],
            'test': test_valid['test'],
            'validation': test_valid['train']}
        )
    else:
        raise Exception('need valid task')
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    label_list = datasets["train"].features["label"].names if not is_multilabel else [
        'sentence1', 'sentence2', 'sentence3', 'sentence4']
    num_labels = 2 if data_args.task_name == 'storycloze_prediction' else 4

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    sep_token = tokenizer.special_tokens_map['sep_token']

    def preprocess_function(examples):
        # Tokenize the texts
        if data_args.task_name == 'storycloze_prediction':
            args = [' '.join((examples['InputSentence1'][i], examples['InputSentence2'][i],
                              examples['InputSentence3'][i], examples['InputSentence4'][i], sep_token,
                              examples['RandomFifthSentenceQuiz1'][i], examples['RandomFifthSentenceQuiz2'][i]))
                    for i in range(len(examples['InputSentence1']))
                    ]
            result = tokenizer(args, padding=padding,
                               max_length=max_seq_length, truncation=True)
        else:
            args = [' '.join((examples['first_sentence'][i], examples['second_sentence'][i],
                              examples['third_sentence'][i], examples['fourth_sentence'][i], sep_token,
                              examples['fifth_sentence'][i]))
                    for i in range(len(examples['first_sentence']))
                    ]
            result = tokenizer(args, padding=padding,
                               max_length=max_seq_length, truncation=True)

        # result["labels"] = examples["label"]

        return result

    column_names = datasets["train"].column_names
    # we want to remove extra columns from dataset except for label
    column_names.remove("label")

    datasets = datasets.map(preprocess_function, batched=True, remove_columns=column_names,
                            load_from_cache_file=not data_args.overwrite_cache)
    if training_args.do_train:
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(
                range(data_args.max_train_samples))

    if training_args.do_eval:
        eval_dataset = datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(
                range(data_args.max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        predict_dataset = datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(
                range(data_args.max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(
                f"Sample {index} of the training set: {train_dataset[index]}.")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(
            p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1) if not is_multilabel else np.array(
            [p > THRESHOLD for p in preds])
        precision, recall, fscore, _ = precision_recall_fscore_support(
            p.label_ids, preds, average='weighted', zero_division=0)
        accuracy = accuracy_score(p.label_ids, preds)
        if not is_multilabel:
            balanced_accuracy = balanced_accuracy_score(p.label_ids, preds)
            result = {'accuracy': accuracy, 'balanced_accuracy': balanced_accuracy,
                      'precision': precision, 'recall': recall, 'f1': fscore}
        else:
            hamming_score = hamming_loss(p.label_ids, preds)
            result = {'accuracy': accuracy, 'hamming_score': hamming_score,
                      'precision': precision, 'recall': recall, 'f1': fscore}
        return result

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = CommonSenseTrainer(
        class_weights=class_weights,
        multi_label=is_multilabel,
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(
                train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
                eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        metrics = trainer.evaluate(eval_dataset=predict_dataset)
        metrics["eval_samples"] = len(predict_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        # Removing the `label` columns because it contains -1 and Trainer won't like that.
        # predict_dataset.remove_columns_("label")
        predictions = trainer.predict(
            predict_dataset, metric_key_prefix="predict").predictions
        predictions = np.argmax(predictions, axis=1) if not is_multilabel else np.array(
            [p > THRESHOLD for p in predictions])

        output_predict_file = os.path.join(
            training_args.output_dir, f"predict_results.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info(
                    f"***** Predict results {data_args.task_name} *****")
                writer.write("index\tprediction\n")
                for index, (item, label) in enumerate(zip(predictions, predict_dataset['label'])):
                    item = label_list[item] if not is_multilabel else [
                        label_list[i] for i, x in enumerate(item) if x]
                    label = label_list[label] if not is_multilabel else [
                        label_list[i] for i, x in enumerate(label) if x]
                    writer.write(f"{index}: {item} / {label}\n")

    if training_args.push_to_hub:
        trainer.push_to_hub()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
