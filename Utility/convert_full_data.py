"""converting the json data to tensorflow model feeding data
    Used when the samples are present individually in files in a in_folder
    Goes with: 'dump_full_data_from_server.py'"""
#import dependencies
import collections
import struct
import sys
import json

import os
from os.path import isfile, join

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

# from progressbar import progressbar

import tensorflow as tf
from tensorflow.core.example import example_pb2

from numpy.random import seed as random_seed
from numpy.random import shuffle as random_shuffle

random_seed(21) #Reproducibil

#Define the Global Variables
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('command', 'text_to_binary',
                           'Either text_to_vocabulary or text_to_binary.'
                           'Specify FLAGS.in_directories accordingly.')
tf.app.flags.DEFINE_string('in_folder','','path to input json data file')
tf.app.flags.DEFINE_string('out_files', '','comma seperated paths to files') #specify the outfile during command line interface
tf.app.flags.DEFINE_string('split', '', 'comma separated fractions of data') #specify during terminal command call
# tf.app.flags.DEFINE_string('vocab_file','data/vocabulary','path to output the vocab of the training corpus')
# tf.app.flags.DEFINE_integer('body_len', 30000, 'Define the length of body to consider')
# tf.app.flags.DEFINE_integer('abs_len', 1500, 'Define the length of abstract to consider')
tf.app.flags.DEFINE_integer('max_words', 500000000, 'Define the max number of words to consider in vocab')
#
# UNKNOWN_TOKEN = '<UNK>'
# PAD_TOKEN = '<PAD>'
#

def _get_json_file_data(file):
    """ reads the json file and returns the loaded data"""
    loaded_data = []
    with open(file, mode='r') as file_json:
        loaded_data = json.load(file_json)
    for document in loaded_data:
        return document

def _text_to_vocabulary(input_folder, output_file):
    """Converts the training corpus text to vocabulary
       and stores it in vocab file"""

    counter = collections.Counter()
    count_doc = 0

    for filename in os.listdir(input_folder):
        count_doc += 1
        document = _get_json_file_data(input_folder + filename)
        body = document['body']
        abstract = document['abstract']
        # getting words.
        words_body = word_tokenize(body)
        words_abstract = word_tokenize(abstract)
        # updating the counter.
        counter.update(words_body)
        counter.update(words_abstract)

        #progress marker
        if count_doc % 5000 == 0:
            print(count_doc)

        for word in list(counter):
            if word.isdigit():
                del counter[word]
    # write the vocab in the file.

    with open(output_file, mode='w') as file_vocab:
        for word, count in counter.most_common(FLAGS.max_words):
            file_vocab.write(word + ' ' + str(count) + '\n')
            # Output the number of UNK tokens
        unk_words = len(counter) - FLAGS.max_words
        unk_count = 0
        for word, count in counter.most_common()[:-unk_words-1:-1]:
            unk_count += count
        s_count = int(counter['.'] * .7)

        file_vocab.write('<s> '+str(s_count)+'\n')
        file_vocab.write('</s> '+str(s_count)+'\n')
        file_vocab.write('<UNK> '+ str(unk_count)+'\n')
        file_vocab.write('<PAD> '+str(5)+'\n')
    print('Vocabulary file generated!!')

def _text_to_binary(input_folder, output_files, split_fractions):
    """ Splitting the input data by split fractions
        for training, testing and validation and
        passing the output file accordingly to _convert_json_to_binary"""

    count_doc = 0
    file_list = os.listdir(input_folder)
    length_file_list = len(file_list)
    random_shuffle(file_list)
    start = 0

    for index, file_out in enumerate(output_files):
        sample_count = int(length_file_list * split_fractions[index])
        end = min(start+ sample_count, len(file_list)-1)
        for filename in file_list[start:end]:
            count_doc += 1
            document = _get_json_file_data(input_folder + filename)
            _convert_json_to_binary(document, file_out)
            if count_doc + 1 == sample_count:
                start = end
                count_doc = 0

        print(file_out + ": " + str(sample_count))


def _convert_json_to_binary(document, output_filename):
    """ converting the split sections of json filenames to
        binary for training and storing in output file """
    with open(output_filename, 'ab') as out_file:
        # extracting body and abstract out of document.
        body = document['body']
        abstract = document['abstract']

        body = body.replace('\n',' ').replace('\t',' ')
        abstract = abstract.replace('\n',' ').replace('\t',' ')

        sentences = sent_tokenize(body)
        body = '<d> <p>' + ' '.join([' <s> ' + (' ').join(word_tokenize(sentence)) + ' </s> ' for sentence\
                                    in sentences]) + '</p> </d>'
        # encoding in unicode.
        body = body.encode('utf8')
        # encoding abstract.
        sentences = sent_tokenize(abstract)
        abstract = '<d> <p>' + ' '.join([' <s> ' + (' ').join(word_tokenize(sentence)) + ' </s> ' for sentence\
                                    in sentences]) + '</p> </d>'
        abstract = abstract.encode('utf8')
        # converting abstract and body to tf example(binary format!).
        tf_example = example_pb2.Example()
        tf_example.features.feature['article'].bytes_list.value.extend([body])
        tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
        tf_example_str = tf_example.SerializeToString()
        str_len = len(tf_example_str)
        out_file.write(struct.pack('q', str_len))
        out_file.write(struct.pack('%ds' % str_len, tf_example_str))

def main(unused_argv):
    """ the main function """
    assert FLAGS.command and FLAGS.in_folder and FLAGS.out_files
    # parsing command line variables.
    output_filenames = FLAGS.out_files.split(',')
    input_folder = FLAGS.in_folder

    if FLAGS.command == 'text_to_binary':
        assert FLAGS.split
        split_fractions = [float(s) for s in FLAGS.split.split(',')]
        assert len(output_filenames) == len(split_fractions)
        _text_to_binary(input_folder, output_filenames, split_fractions)
    elif FLAGS.command == 'text_to_vocabulary':
        assert len(output_filenames) == 1
        _text_to_vocabulary(input_folder, output_filenames[0])


if __name__ == '__main__':
    tf.app.run()
