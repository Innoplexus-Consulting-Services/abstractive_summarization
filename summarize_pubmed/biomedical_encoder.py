# Biomedical Encoder Developed
# for summarize_pubmed problem of
# tensor2tensor module
# for queries contact Mainak chain

"""
Encoders for biomedical text data

* BioMedicalEncoder: with user-supplied biomed vocabulary file
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import collections
# from itertools import chain
# import math
# import re
# import tempfile
# import numpy as np
# import six
# from six.moves import range  # pylint: disable=redefined-builtin
# from tensor2tensor.data_generators import tokenizer
from tensor2tensor.data_generators import text_encoder

try:
    from . import biomedical_tokenizer
except:
    import biomedical_tokenizer

import tensorflow as tf

vocab_filename = "vocab.%s.%s" % ('summarize_pubmed', 'biotokens')

UNK = "<UNK>"
RESERVED_TOKENS = text_encoder.RESERVED_TOKENS.append(UNK)
UNK_ID = text_encoder.RESERVED_TOKENS.index(UNK)
text_encoder.NUM_RESERVED_TOKENS = len(text_encoder.RESERVED_TOKENS)
print(text_encoder.RESERVED_TOKENS)

class BioMedicalEncoder(text_encoder.TokenTextEncoder):
    def __init__(self,
                 vocab_filename,
                 reverse=False,
                 vocab_list=None,
                 replace_oov=None,
                 num_reserved_ids=text_encoder.NUM_RESERVED_TOKENS):

        print('*************************')
        print(vocab_filename)
        print(num_reserved_ids)
        super(BioMedicalEncoder, self).__init__(vocab_filename=vocab_filename,
                                                num_reserved_ids=num_reserved_ids)
        self._reverse = reverse
        self._replace_oov = replace_oov
        if vocab_filename:
          self._init_vocab_from_file(vocab_filename)
        else:
          assert vocab_list is not None
          self._init_vocab_from_list(vocab_list)


    def encode(self, s):
        """Converts a space-separated string of tokens to a list of ids."""
        sentence = s
        tokens = list(biomedical_tokenizer.biomedical_tokenizer(sentence))
        print(len(tokens))
        if self._replace_oov is not None:
          tokens = [t if t in self._token_to_id else self._replace_oov
                    for t in tokens]
        ret = [self._token_to_id[tok] for tok in tokens]
        return ret[::-1] if self._reverse else ret
