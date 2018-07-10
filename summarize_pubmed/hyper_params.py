from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import re
# import io
# import os
# import string
# import six
# import json
# from tensor2tensor.data_generators import generator_utils
# from tensor2tensor.data_generators import problem
# from tensor2tensor.data_generators import text_encoder
# from tensor2tensor.data_generators import text_problems
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
# # from tensor2tensor.data_generators import tokenizer
# from tensor2tensor.utils import metrics

import tensorflow as tf

@registry.register_hparams
def transformer_few_sample():
    """Set of hyperparameters."""
    hparams = transformer.transformer_base()
    hparams.num_hidden_layers = 2
    hparams.hidden_size = 16
    hparams.batch_size = 4
    hparams.dropout = 0.1
    hparams.learning_rate = 0.01
    hparams.filter_size = 8
    hparams.dropout = 0.5
    return hparams
