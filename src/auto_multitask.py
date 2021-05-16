from collections import OrderedDict
from transformers.models.auto.configuration_auto import (
    BertConfig,
    RobertaConfig,
    DebertaConfig,
    GPT2Config
)
from .bert_multitask import BertMultitaskModel
from .roberta_multitask import RobertaMultitaskModel
from .deberta_multitask import DebertaMultitaskModel
from .gpt2_multitask import GPT2MultitaskModel

from transformers.models.auto.auto_factory import auto_class_factory

MULTITASK_MODEL_MAPPING = OrderedDict(
    [
        (BertConfig, BertMultitaskModel),
        (RobertaConfig, RobertaMultitaskModel),
        (DebertaConfig, DebertaMultitaskModel),
        (GPT2Config, GPT2MultitaskModel),
    ]
)

AutoModelForMultitasking = auto_class_factory(
    "AutoModelForMultitasking", MULTITASK_MODEL_MAPPING, head_doc="multitask classification + generation"
)