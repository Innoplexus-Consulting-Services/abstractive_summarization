"""
Returns a collections.Counter object given a biomedical text
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import string
import os
import six
import nltk
import re
import json
import argparse
from collections import Counter

try:
    from . import iplexus_analyzer
except:
    import iplexus_analyzer

def sequence_check(index, tokens, match_biomedical_tokens):
    token_span = 1
    token = tokens[index]
    while(True):
        if (' '.join(tokens[index:index+token_span+1])) in match_biomedical_tokens:
            token_span += 1
        else:
            return ' '.join(tokens[index:index+token_span]).strip(), (token_span-1)

def text_tokenizer(text, biomedical_tokens):
    tokens = nltk.word_tokenize(text)
    vocab_tokens = []
    tokens_iter = iter(tokens)
    index = -1
    for token in tokens_iter:
        index += 1
        if token in [bio_token.split()[0] for bio_token in biomedical_tokens]:
            match_biomedical_tokens = [bio_token for bio_token in biomedical_tokens
                                       if bio_token.split()[0] == token ]
            vocab_token, token_span = sequence_check(index, tokens, match_biomedical_tokens)
            vocab_tokens.append(vocab_token)
            for s in range(token_span):
                next(tokens_iter)
                index = index + 1
        else:
            vocab_tokens.append(token)
    return vocab_tokens

def get_biomedical_tokens(text):
    biomedical_tokens = []
    for ent in iplexus_analyzer.tag_entities(text)['ents']:
        biomedical_tokens.append(text[ent['start']:ent['end']])
    return list(set(biomedical_tokens))

def biomedical_tokenizer(text):
    """
    Given a text, returns a collections.Counter() object
    of biomedical tokens of the same
    """
    #initializing the counter
    vocab_counter = Counter()
    text = text.strip()
    biomedical_tokens = get_biomedical_tokens(text)
    biomedical_tokens = [token.strip() for token in biomedical_tokens]
    vocab_tokens = text_tokenizer(text, biomedical_tokens)
    vocab_counter.update(vocab_tokens)
    return vocab_counter
