import numpy as np
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from transformers import Trainer
from transformers.tokenization_utils_base import BatchEncoding

from typing import Dict


class CommonSenseTrainer(Trainer):
    def __init__(self, class_weights=None, multi_label=False, **kwargs):
        super().__init__(**kwargs)
        if class_weights is not None:
            assert(len(class_weights) == self.model.num_labels)
            self.class_weights = torch.as_tensor(
                class_weights).float().to(self.args.device)
            print(self.class_weights)
        else:
            self.class_weights = None
        self.multi_label = multi_label

    def compute_loss(self, model, inputs, return_outputs=False):
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        loss = None
        logits = outputs["logits"]

        if labels is not None:
            if self.label_smoother is not None:
                loss = self.label_smoother(outputs, labels)
            elif self.multi_label:  # multilabel, use binary cross entropy
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            elif not self.multi_label:  # multiclass not multilabel, use normal cross entropy
                loss_fct = CrossEntropyLoss(weight=self.class_weights)
                loss = loss_fct(
                    logits.view(-1, self.model.num_labels), labels.view(-1))

        # Save past state if it exists (from original implementation)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return (loss, outputs) if return_outputs else loss


class MultitaskTrainer(Trainer):
    def __init__(self, class_weights=None, multi_label=False, **kwargs):
        super().__init__(**kwargs)
        if class_weights is not None:
            assert(len(class_weights) == self.model.num_labels)
            self.class_weights = torch.as_tensor(
                class_weights).float().to(self.args.device)
            print(self.class_weights)
        else:
            self.class_weights = None
        self.multi_label = multi_label

    def compute_loss(self, model, inputs, return_outputs=False):
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        lm_labels, cls_labels = labels if labels is not None else [None, None]

        outputs = model(**inputs)

        lm_logits = outputs["logits"]
        cls_logits = outputs["cls_logits"]

        lm_loss = None
        if lm_labels is not None:
            lm_labels = lm_labels.to(self.args.device)
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_logits = lm_logits[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()
            lm_loss_fct = CrossEntropyLoss()
            lm_loss = lm_loss_fct(
                shifted_logits.view(-1, self.model.config.vocab_size), lm_labels.view(-1))

        cls_loss = None
        if cls_labels is not None:
            cls_labels = cls_labels.to(self.args.device)
            if not self.multi_label:
                # multiclass not multilabel, use normal cross entropy
                cls_loss_fct = CrossEntropyLoss(weight=self.class_weights)
                cls_loss = cls_loss_fct(
                    cls_logits.view(-1, self.model.num_labels), cls_labels.view(-1))
            else:
                # multilabel, use binary cross entropy
                cls_loss_fct = BCEWithLogitsLoss()
                cls_loss = cls_loss_fct(cls_logits, cls_labels)

        losses = [loss for loss in [lm_loss, cls_loss] if loss is not None]
        loss = sum(losses) if losses else None

        # Save past state if it exists (from original implementation)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return (loss, outputs) if return_outputs else loss


def default_data_collator(features) -> Dict[str, torch.Tensor]:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:

        - ``label``: handles a single value (int or float) per object
        - ``label_ids``: handles a list of values per object

    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """
    IGNORE_LIST = ("label", "label_ids", "first_sentence", "second_sentence",
                   "third_sentence", "fourth_sentence", "fifth_sentence",)

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(
            first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(
            label, (int, np.integer)) else torch.float
        batch["labels"] = torch.tensor(
            [f["label"] for f in features], dtype=dtype)

    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(
                first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor(
                [f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in IGNORE_LIST and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


def multitask_data_collator(features) -> Dict[str, torch.Tensor]:
    """
    data collator for the multitask objective
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    if "label" in first and first["label"] is not None:
        dtype = torch.long
        batch["labels"] = (torch.tensor([f["input_ids"].copy() for f in features], dtype=dtype), torch.tensor(
            [f["label"] for f in features], dtype=dtype))

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch
