import torch
from torch import nn
from transformers import DebertaModel, DebertaPreTrainedModel
from transformers.models.deberta.modeling_deberta import ContextPooler, StableDropout

from .output import MultitaskOutput


class DebertaMultitaskModel(DebertaPreTrainedModel):
    """
    The Deberta Model transformer with a language modeling head and  on top (linear layer with weights tied to the input
    embeddings).
    """

    def __init__(self, config):
        super().__init__(config)
        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaModel(config)
        # classification head
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        # lm head
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=True)

        self.init_weights()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # lm head
        pooled_output = outputs[0]

        lm_logits = self.lm_head(pooled_output)

        pooled_output = self.pooler(pooled_output)
        pooled_output = self.dropout(pooled_output)
        cls_logits = self.classifier(pooled_output)

        loss = None
        if not return_dict:
            output = (lm_logits, cls_logits) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        else:
            return MultitaskOutput(
                loss=loss,
                logits=lm_logits,
                cls_logits=cls_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past}

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx)
                               for past_state in layer_past),)
        return reordered_past
