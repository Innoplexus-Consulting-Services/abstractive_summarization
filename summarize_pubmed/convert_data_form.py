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
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',
                    help='directory with all json files to convert.')
parser.add_argument('--output_dir',
                    help='directory with all .story files to store.')
parser.add_argument('--mode',
                    help='train, test or val.')
parser.add_argument('--sentdiv',
                    help='start_sent,end_sent,abs_sent')

args = parser.parse_args()

sent_div = args.sentdiv.split(',')
assert len(sent_div) == 3
first_sent = int(sent_div[0])
last_sent = int(sent_div[1])
abs_sent = int(sent_div[2])

def get_file_list(input_folderpath):
    path = input_folderpath + '/*'
    input_files = tf.gfile.Glob(path)
    return input_files

def get_file_data(file):
    """ reads the json file and returns the loaded data"""
    loaded_data = []
    with open(file, mode='r') as file_json:
        try:
            loaded_data = json.load(file_json)
            for document in loaded_data:
                return document['abstract'], document['body']
        except ValueError:
            print('File name: '+ str(file))
            return False, False


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
    # text = ''.join([i for i in text if not i.isdigit()])
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text)
    punct_set = [" ,", " .", " !", " ?", " '"," \""]
    for punct in punct_set:
        text = text.replace(punct, punct[-1])
    return re.sub(' +',' ',text.strip())

def check(abs, bod):
    #data stats
    avg_b = 28000   #body length average
    avg_a = 1400    #abstract length average
    sig_b = 19000   #deviation of body length
    sig_a = 600 #deviation of abstract length

    if ((avg_b - 2*sig_b) <= len(bod) <= (avg_b + 2*sig_b)) \
        and ((avg_a - 2*sig_a) <= len(abs) <= (avg_a + 2*sig_a)):
        return True
    else:
        return False

def write_files(mode, data_dir, output_dir):
    """Writes the files and corresponding hash codes in a file"""
    files_list = get_file_list(data_dir)
    hash_names = []
    delete_instance = False
    write_or_append = 'w'

    completed_file_list = []

    if os.path.exists(output_dir + '/all_' + mode + '.txt'):
        print('Found file: ' + output_dir + '/all_' + mode + '.txt')
        with open(output_dir + '/all_' + mode + '.txt', 'r') as allf:
            completed_file_list = allf.readlines()
            print('Number of documents completed: ' + str(len(completed_file_list)))
            print('Conversion resuming from this checkpoint...')
            write_or_append = 'a'

    input_files = list(set(files_list) - set(completed_file_list))

    for file in input_files:
        hash_name = generate_hash(file.encode('utf-8'))
        hash_names.append(hash_name)

    dup = len(set(input_files)) - len(set(hash_names))
    if dup != 0:
        tf.logging.info("{} duplicated hash codes. Exiting....See ya!".format(dup))
        exit(-1)

    with open(output_dir + '/all_' + mode + '.txt', write_or_append) as allf:
        for i, hash_name in tqdm(enumerate(hash_names)):
            with open(output_dir + '/' + hash_name + '.story', 'w') as hashf:
                abstract, body = get_file_data(input_files[i])
                if (not abstract) or (not body):
                    continue
                abstract = preprocess_text(abstract)
                body = preprocess_text(body)

                cond = check(abstract, body) #to check if it's not an outlier

                if cond:
                    allf.write(input_files[i]+ '\n')
                    body_sentences = sent_tokenize(body)
                    abs_sentences = sent_tokenize(abstract)
                    for sent in body_sentences[:min(len(body_sentences), first_sent)]:
                        sent = sent.strip()
                        if sent.startswith('introduction'):
                            hashf.write(sent[13:] + '\n\n')
                        elif sent.startswith('background'):
                            hashf.write(sent[11:] + '\n\n')
                        elif sent.startswith('. '):
                            hashf.write(sent[2:] + '\n\n')
                        else:
                            hashf.write(sent + '\n\n')

                    if (len(body_sentences) > first_sent) and (last_sent > 0):
                        for sent in body_sentences[-1*min(len(body_sentences)-first_sent, last_sent):]:
                            sent = sent.strip()
                            if sent.startswith('introduction'):
                                hashf.write(sent[13:] + '\n\n')
                            elif sent.startswith('background'):
                                hashf.write(sent[11:] + '\n\n')
                            elif sent.startswith('. '):
                                hashf.write(sent[2:] + '\n\n')
                            else:
                                hashf.write(sent + '\n\n')

                    for sent in abs_sentences[:min(len(abs_sentences), abs_sent)]:
                        sent = sent.strip()
                        if sent.startswith('aim'):
                            hashf.write("@highlight\n\n")
                            hashf.write(sent[4:] + '\n\n')
                        elif sent.startswith('objective'):
                            hashf.write("@highlight\n\n")
                            hashf.write(sent[10:] + '\n\n')
                        elif sent.startswith('introduction'):
                            hashf.write("@highlight\n\n")
                            hashf.write(sent[13:] + '\n\n')
                        elif sent.startswith('background'):
                            hashf.write("@highlight\n\n")
                            hashf.write(sent[11:] + '\n\n')
                        else:
                            hashf.write("@highlight\n\n")
                            hashf.write(sent + '\n\n')
                    delete_instance = False
                else:
                    delete_instance = True

            if delete_instance == True:
                os.remove(output_dir + '/' + hash_name + '.story')


if __name__ == '__main__':
    data_dir = args.data_dir
    output_dir = args.output_dir
    mode = args.mode
    write_files(mode, data_dir, output_dir)
