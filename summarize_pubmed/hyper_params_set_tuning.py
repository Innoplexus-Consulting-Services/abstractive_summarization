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
# from tensor2tensor.data_generators import tokenizer
# from tensor2tensor.utils import metrics

import tensorflow as tf


# hyperparameter tuning ranges
@registry.register_ranged_hparams
def transformer_few_sample_range(rhp):
  rhp.set_float("learning_rate", 0.05, 2.0, scale=rhp.LOG_SCALE)
  # rhp.set_discrete("learning_rate_warmup_steps",
  #                  [1000, 2000, 4000, 8000, 16000])
  rhp.set_float("initializer_gain", 0.5, 2.0)
  rhp.set_float("optimizer_adam_beta1", 0.85, 0.95)
  rhp.set_float("optimizer_adam_beta2", 0.97, 0.99)
  rhp.set_float("weight_decay", 0.0, 1e-4)
  rhp.set_int("num_hidden_layers", 2, 3, 4)
  rhp.set_discrete("hidden_size", [128, 256, 512])
  rhp.set_float("dropout", 0.2, 0.4, 0.6)
  rhp.set_float("attention_dropout", 0.1, 0.2, 0.3)
