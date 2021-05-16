from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.modeling_outputs import ModelOutput

@dataclass
class MultitaskOutput(ModelOutput):
    """
    Output for multitask models (classification + language modeling)
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None # lm_logits (named this way so model.generate can use them)
    cls_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class MultitaskOutputWithCrossAttentions(ModelOutput):
    """
    Output for multitask models (classification + language modeling)
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None # lm_logits (named this way so model.generate can use them)
    cls_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None