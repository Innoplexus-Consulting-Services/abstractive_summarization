"""Creating an extractive summary of pubmed text"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from flask import Flask, request
from flask_cors import CORS
import re
import json
import io
import os
import string
from nltk import sent_tokenize
from gensim.summarization.summarizer import summarize
#Needs files2rouge installed for ROUGE calculation

import tensorflow as tf

app = Flask(__name__)
CORS(app)
PASSWORD = "asdhBBBXyAH#$%^"
USER = "user"

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
    return open(text_file,'r',encoding='utf-8').read().splitlines()

def extractive_summarize(text, ratio, split):
    summarized_text = summarize(text, ratio=ratio, split=split)
    return summarized_text

def write_text_to_file(text, file):
    with io.open(file,'w') as f:
        f.write(text)

def rouge_score(summary_file, text_file):
    os.system("files2rouge "+summary_file+" "+text_file)

def summarize_batch(batch):
    """it gets similarity for batches."""
    results = []

    for record in batch:
        if record['password'] == PASSWORD and record['username'] == USER:
            text = record["text"]
            text = text.replace("\n","")
            ratio = record["ratio"]
            split = record["split"]
            text = preprocess_text(text)
            summarized_text = extractive_summarize(text, ratio, split)
            del record["text"]
            del record["username"]
            del record["password"]
            record["summarized_text"] = summarized_text
            results.append(record)
        else:
            results.append({"response": "Authentication Failed"})

    return results

@app.route('/summarize', methods=['POST'])
def summarizeAPI():
    batch = json.loads(request.form["batch"])
    results = summarize_batch(batch)
    return json.dumps({"results" : results})




if __name__ == "__main__":
    host = "localhost"
    # host = "0.0.0.0"
    app.run(
        host=host,
        port=int(4567)
    )
