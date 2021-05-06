import torch
from torch.nn import CrossEntropyLoss
from transformers import Trainer


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        
        if class_weights is not None:
            assert(len(class_weights) == kwargs['model'].num_labels)
            self.class_weights = torch.as_tensor(
                class_weights).float().to(self.args.device)
            print(self.class_weights)

    def compute_loss(self, model, inputs, return_outputs=False):
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        loss = None
        logits = outputs["logits"]

        if self.label_smoother is not None and labels is not None:
            loss = self.label_smoother(outputs, labels)
        elif labels is not None:
            # calculate weighted loss
            label_index = (labels >= 0).nonzero()
            labels = labels.long()
            labeled_logits = torch.gather(logits, 0, label_index.expand(
                label_index.size(0), logits.size(1)))
            labels = torch.gather(labels, 0, label_index.view(-1))
            loss_fct = CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(
                labeled_logits.view(-1, self.model.num_labels).float(), labels.view(-1))

        # Save past state if it exists (from original implementation)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return (loss, outputs) if return_outputs else loss
