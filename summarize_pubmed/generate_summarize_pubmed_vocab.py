##################################
#Developed by Mainak Chain       #
#during internship project of    #
#Document Summarization at       #
#Innoplexus, 2018. Copyright 2018#
##################################
"""
Vocabulary file generation for our problem - 'summarize_pubmed'.
Needs to have the respective tagger module for extracting the entities.
Can be done with any biomedical texts to preserve the unique
biomedical tokens for embedding.
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
from progressbar import progressbar

import biomedical_tokenizer

parser = argparse.ArgumentParser()

parser.add_argument('--tmp_dir',
                    help='folder to read pubmed documents from')
parser.add_argument('--data_dir',
                    help='folder to write the biomedical vocab')
parser.add_argument('--vocab_size',
                    help='max size of vocabulary to keep', type=int)
parser.add_argument('--preserve_biomedical_tokens', action='store_true',
                    help='to preserve biomed tokens irrespective of vocab size (True/False)')

args = parser.parse_args()

vocab_filename = "vocab.%s.%s" % ('summarize_pubmed', 'biotokens')

def preprocess_text(text):
    text = text.lower().strip()
    re.sub(' +',' ',text.replace('\n',' '))
    char_set = ['\n','\t']
    for ch in char_set:
        text = text.replace(ch, ' ')
    text = ''.join([i if ord(i) < 128 else ' ' for i in text]) #only ascii chars
    text = re.sub(r"[\(\[].*?[\)\]]", '',text) #removes the bracket texts
    #removing brackets
    for brack in ["(",")","[","]","{","}"]:
        text = text.replace(brack, ' ')
    #removes the hyper-links if any
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text)
    punct_set = [" ,", " .", " !", " ?", " '"," \""]
    for punct in punct_set:
        text = text.replace(punct, punct[-1])
    text = re.sub(' +',' ',text.strip())
    # Space around punctuation
    text = re.sub("[%s]" % re.escape(string.punctuation), r" \g<0> ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def _get_data(file):
    """ reads the json file and returns the loaded data"""
    loaded_data = []
    with open(file, mode='r') as file_json:
        loaded_data = json.load(file_json)
        for document in loaded_data:
            body = preprocess_text(document['body'])
            abstract = preprocess_text(document['abstract'])
            return abstract, body

def write_vocab_file(vocab_counter, data_dir, vocab_size):
    with open(os.path.join(data_dir, vocab_filename),'w') as vocab_file:
        for (token, count) in vocab_counter.most_common(vocab_size):
            vocab_file.write(token + '\n')
    print( "Vocab File: " + vocab_filename + " generated!")

def _get_file_list(input_folderpath):
    path = input_folderpath
    input_files = os.listdir(path)
    return input_files

def _get_file_data(tmp_dir, file):
    #data stats (used to exclude outliers)
    avg_b = 28000   #body length average
    avg_a = 1400    #abstract length average
    sig_b = 19000   #deviation of body length
    sig_a = 600     #deviation of abstract length

    #generates only the vocab for train and dev files
    if os.path.isfile(os.path.join(tmp_dir, file)):
        abstract, body = _get_data(os.path.join(tmp_dir, file))

        if ((avg_b - 2*sig_b) <= len(body) <= (avg_b + 2*sig_b)) \
            and ((avg_a - 2*sig_a) <= len(abstract) <= (avg_a + 2*sig_a)):
            return(body + " " + abstract )
        else:
            return None


# def get_biomedical_tokens(text):
if __name__ == "__main__":
    tmp_dir = args.tmp_dir
    data_dir = args.data_dir
    vocab_size = args.vocab_size
    biomed_cond = args.preserve_biomedical_tokens

    file_counter = 0
    input_files = _get_file_list(tmp_dir)
    print(len(input_files))

    print('Generating Vocabulary file...')
    for file in progressbar(input_files):
        text = _get_file_data(tmp_dir, file)
        if text:
            file_counter += 1
        else:
            continue
        vocab_counter = biomedical_tokenizer.biomedical_tokenizer(text)
        if biomed_cond:
            vocab_counter.update(biomedical_tokens)

    print(str(file_counter) + " files used for developing vocabulary.")
    write_vocab_file(vocab_counter, data_dir, vocab_size)
