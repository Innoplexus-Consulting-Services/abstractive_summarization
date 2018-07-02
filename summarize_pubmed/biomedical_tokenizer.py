##################################
#Developed by Mainak Chain       #
#during internship project of    #
#Document Summarization at       #
#Innoplexus, 2018. Copyright 2018#
##################################
"""
Vocabulary generations script for life science domain.
Needs to have the respective tagger module for extracting the entities.
Can be done with any biomedical texts to preserve the unique
biomedical tokens for embedding.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import string
import six
import nltk
import re
# from highlighter import HighLighter
import argparse
from collections import Counter

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--doc_path',
                    help='path to the biomedical document')
parser.add_argument('--vocab_path',
                    help='path to write the biomedical vocab')
parser.add_argument('--vocab_size',
                    help='max size of vocabulary to keep', type=int)
parser.add_argument('--preserve_biomedical_tokens', action='store_true',
                    help='to preserve biomed tokens irrespective of vocab size (True/False)')


args = parser.parse_args()

def sequence_check(index, tokens, match_biomedical_tokens):
    token_span = 1
    token = tokens[index]
    while(True):
        if (' '.join(tokens[index:index+token_span+1])) in match_biomedical_tokens:
            token_span += 1
        else:
            return ' '.join(tokens[index:index+token_span]).strip(), (token_span-1)

def biomedical_tokenizer(text):
    # h = HighLighter()
    # tagged_text = h.highlight({'input_text':text},'publications')['input_text']
    # biomedical_tokens = re.findall(r'<span .*?>(.*?)</span>',tagged_text)
    # biomedical_tokens = list(set(biomedical_tokens))
    # return biomedical_tokens
    return ['lung cancer','spiral fracture','social phobia']


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

def read_file_data(file):
    lines = []
    with open(file, 'r') as f:
        lines = f.readlines()
        return lines

def write_vocab_file(vocab_counter, vocab_path, vocab_size):
    with open(vocab_path,'w') as vocab_file:
        for (token, count) in vocab_counter.most_common(vocab_size):
            vocab_file.write(token + '\n')
    print( "Vocab File generated!")

# def biomedical_tokenizer(text):
if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    vocab_path = args.vocab_path
    vocab_size = args.vocab_size
    doc_path = args.doc_path
    biomed_cond = args.preserve_biomedical_tokens
    vocab_counter = Counter()

    lines = read_file_data(doc_path)
    if lines:
        print("Document read successfully!")
    else:
        print('Error reading document!... Exiting. See ya!')
        exit(-1)
    for line in lines:
        line = line.strip()
        biomedical_tokens = biomedical_tokenizer(line)
        # biomedical_tokens = [token.strip() for token in biomedical_tokens]
        vocab_tokens = text_tokenizer(line, biomedical_tokens)
        vocab_counter.update(vocab_tokens)
        if biomed_cond:
            vocab_counter.update(biomedical_tokens)
    write_vocab_file(vocab_counter, vocab_path, vocab_size)
