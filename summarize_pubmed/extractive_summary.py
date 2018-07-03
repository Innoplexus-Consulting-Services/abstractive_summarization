"""Creating an extractive summary of pubmed text"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import io
import os
import string
from nltk import sent_tokenize
from gensim.summarization.summarizer import summarize
#Needs files2rouge installed for ROUGE calculation

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string("text_file", None, "file to read the biomedical text from")
flags.DEFINE_string("summary_file", None, "file to output the summary file")
flags.DEFINE_string("preprocess_file", None, "file to output the preprocessed text")
flags.DEFINE_float("extractive_summarize_ratio", 0.15,
                     "Ratio of output text to input text for extractive summarization")
flags.DEFINE_bool("split", True,
                  "To split the output or not")
###############################
# CAUTION: Do not add '\n' in #
# a single text sample. It is #
# used for seperation of      #
# multi-text documents.       #
###############################
"""For multi-text summarization, add them to text_file seperated by \n """
"""The summary output will be in order as the text file, seperated by \n"""

def preprocess_text(text):
    char_set = ['\n','\t']
    for ch in char_set:
        text = text.replace(ch, ' ')
    text = ''.join([i if ord(i) < 128 else ' ' for i in text]) #only ascii chars
    text = re.sub(r'\([^)]*\)', '',text) #removes the bracket texts
    #removing brackets and numbers
    for brack in ["(",")","[","]","{","}"]:
        text = text.replace(brack, ' ')
    # text = ''.join([i for i in text if not i.isdigit()])
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text)
    punct_set = [" ,", " .", " !", " ?", " '"," \""]
    for punct in punct_set:
        text = text.replace(punct, punct[-1])
    text = re.sub(' +',' ',text.strip().replace('\n',' '))
    return re.sub(' +',' ',text.strip())

def read_file(text_file):
    with io.open(text_file,'r') as file:
        lines = file.readlines()
    return lines

def extractive_summarize(text, ratio, split):
    summarized_text = summarize(text, ratio=ratio, split=split)
    return summarized_text

def write_text_to_file(text, file):
    with io.open(file,'w') as f:
        f.write(text)

def rouge_score(summary_file, text_file):
    os.system("files2rouge "+summary_file+" "+text_file)

if __name__ == "__main__":
    text_file = FLAGS.text_file
    summary_file = FLAGS.summary_file
    preprocess_file = FLAGS.preprocess_file
    ratio = FLAGS.extractive_summarize_ratio
    split = FLAGS.split

    lines = read_file(text_file)
    if lines:
        print('Text file read!')
    else:
        print('Text file reading unsuccessful!.. Exiting. see ya!')
        exit(-1)
    text = ''.join(lines)
    text = preprocess_text(text)
    summarized_text = extractive_summarize(text, ratio, split)
    if summarized_text:
        print('Summarization done!')
    else:
        print("Summarization failed!... Exiting. See ya!")
        exit(-1)

    write_text_to_file(text, preprocess_file)
    print("Preprocessed text file generated!")

    write_text_to_file(preprocess_text(''.join(summarized_text)), summary_file)
    print('Summary file generated!')

    rouge_score(summary_file, text_file)
