import os
import json
import io
import re
import string
import six
import hashlib
import argparse
import tensorflow as tf
from nltk import sent_tokenize

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',
                    help='directory with all json files to convert.')
parser.add_argument('--output_dir',
                    help='directory with all .story files to store.')
parser.add_argument('--mode',
                    help='train, test or val.')

args = parser.parse_args()

def get_file_list(input_folderpath):
    path = input_folderpath + '/*'
    input_files = tf.gfile.Glob(path)
    return input_files

def get_file_data(file):
    """ reads the json file and returns the loaded data"""
    loaded_data = []
    with open(file, mode='r') as file_json:
        loaded_data = json.load(file_json)
        for document in loaded_data:
            return document['abstract'], document['body']

def generate_hash(inp):
    """Generate a sha1 hash to match the raw url to the filename extracted."""
    h = hashlib.sha1()
    h.update(inp)
    return h.hexdigest()

def preprocess_text(text):
    char_set = ['\n','\t']
    for ch in char_set:
        text = text.replace(ch, ' ')
    text = ''.join([i if ord(i) < 128 else ' ' for i in text]) #only ascii chars
    text = re.sub(r'\([^)]*\)', '',text) #removes the bracket texts
    #removing brackets and numbers
    for brack in ["(",")","[","]","{","}"]:
        text = text.replace(brack, ' ')
    text = ''.join([i for i in text if not i.isdigit()])
    punct_set = [" ,", " .", " !", " ?", " '"," \""]
    for punct in punct_set:
        text = text.replace(punct, punct[-1])
    return re.sub(' +',' ',text.strip())

def write_files(mode, data_dir, output_dir):
    """Writes the files and corresponding hash codes in a file"""
    input_files = get_file_list(data_dir)
    hash_names = []
    for file in input_files:
        hash_name = generate_hash(file.encode('utf-8'))
        hash_names.append(hash_name)

    dup = len(set(input_files)) - len(set(hash_names))
    if dup != 0:
        tf.logging.info("{} duplicated hash codes. Exiting....See ya!".format(dup))
        exit(-1)

    with open(output_dir + '/all_' + mode + '.txt', 'w') as allf:
        for i, hash_name in enumerate(hash_names):
            with open(output_dir + '/' + hash_name + '.story', 'w') as hashf:
                abstract, body = get_file_data(input_files[i])
                abstract = preprocess_text(abstract)
                body = preprocess_text(body)
                if abstract and body:
                    allf.write(input_files[i]+ '\n')
                    body_sentences = sent_tokenize(body)
                    abs_sentences = sent_tokenize(abstract)
                    for sent in body_sentences[:min(len(body_sentences),20)]:
                        hashf.write(sent + '\n')
                    hashf.write("@highlight\n")
                    for sent in abs_sentences[:min(len(abs_sentences),5)]:
                        hashf.write(sent + '\n')

if __name__ == '__main__':
    data_dir = args.data_dir
    output_dir = args.output_dir
    mode = args.mode
    write_files(mode, data_dir, output_dir)
