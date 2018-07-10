import os
import sys
import json
# from rouge import Rouge
import argparse
from pprint import pprint
import tensorflow as tf
import re

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',
                    help='input folder path')
parser.add_argument('--output_dir',
                    help='directory with all preprocessed files to store.')

args = parser.parse_args()

def get_file_list(input_folderpath):
    if input_folderpath.endswith('/'):
        path = input_folderpath + '*'
    else:
        path = input_folderpath + '/*'
    input_files = tf.gfile.Glob(path)
    return input_files

def get_file_data(file):
    """ reads the json file and returns the loaded data"""
    loaded_data = []
    with open(file, mode='r') as file_json:
        loaded_data = json.load(file_json)
        for document in loaded_data:
            return document

def write_preprocessed_text_to_files(body, abstract, out_dir, filename):
    if out_dir.endswith('/'):
        path = out_dir
    else:
        path = out_dir + '/'
    filename_parts = filename.split('/')
    filename_ = filename_parts[-1]
    file = filename_.split('.')[0]
    with open(path + '/' + file, 'w') as out_file:
        out_file.write('\n---------------------------Body--------------------------\n')
        out_file.write(body)
        out_file.write('\n--------------------------Abstract-----------------------\n')
        out_file.write(abstract)

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
    return re.sub(' +',' ',text.strip())

if __name__ == '__main__':
    in_dir = args.input_dir
    out_dir = args.output_dir
    file_list = get_file_list(in_dir)
    for file in file_list:
        document = get_file_data(file)
        prep_body = preprocess_text(document['body'])
        prep_abstract = preprocess_text(document['abstract'])
        write_preprocessed_text_to_files(prep_body, prep_abstract, out_dir, file)
