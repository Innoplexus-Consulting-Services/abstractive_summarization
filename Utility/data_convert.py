"""converting the json data to tensorflow model feeding data"""
""" Used when input files are singular i.e. one json file with
    all bodies and abstracts of the documents"""
#import dependencies
import collections
import struct
import sys
import json

from os import listdir
from os.path import isfile, join

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

import tensorflow as tf
from tensorflow.core.example import example_pb2

from numpy.random import seed as random_seed
from numpy.random import shuffle as random_shuffle

random_seed(21) #Reproducibility

#Define the Global Variables
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('command', 'text_to_binary',
                           'Either text_to_vocabulary or text_to_binary.'
                           'Specify FLAGS.in_directories accordingly.')
tf.app.flags.DEFINE_string('in_file','','path to input json data file')
tf.app.flags.DEFINE_string('out_files', '','comma seperated paths to files') #specify the outfile during command line interface
tf.app.flags.DEFINE_string('split', '', 'comma separated fractions of data') #specify during terminal command call
# tf.app.flags.DEFINE_string('vocab_file','data/vocabulary','path to output the vocab of the training corpus')
# tf.app.flags.DEFINE_integer('body_len', 30000, 'Define the length of body to consider')
# tf.app.flags.DEFINE_integer('abs_len', 1500, 'Define the length of abstract to consider')
tf.app.flags.DEFINE_integer('max_words', 5000000, 'Define the max number of words to consider in vocab')
#
# UNKNOWN_TOKEN = '<UNK>'
# PAD_TOKEN = '<PAD>'
#

def _get_json_file_data():
    """ reads the json file and returns the loaded data"""
    loaded_data = []
    with open(FLAGS.in_file, mode='r') as file_json:
        loaded_data = json.load(file_json)
    return loaded_data

def _text_to_vocabulary(output_file):
    """Converts the training corpus text to vocabulary
       and stores it in vocab file"""
    # loading data.
    loaded_data = _get_json_file_data()
    counts = {}
    # get all the words in the counter object.
    counter = collections.Counter()
    for document in loaded_data:
        body = document['body']
        abstract = document['abstract']
        # getting words.
        words_body = word_tokenize(body)
        words_abstract = word_tokenize(abstract)
        # updating the counter.
        counter.update(words_body)
        counter.update(words_abstract)

    # write the vocab in the file.
    with open(output_file, mode='w') as file_vocab:
        for word, count in counter.most_common(FLAGS.max_words):
            file_vocab.write(word + ' ' + str(count) + '\n')
        file_vocab.write('<s> 0\n')
        file_vocab.write('</s> 0\n')
        file_vocab.write('<UNK> 0\n')
        file_vocab.write('<PAD> 0\n')

def _text_to_binary(output_files, split_fractions):
    """ Splitting the input data by split fractions
        for training, testing and validation and
        passing the output file accordingly to _convert_json_to_binary"""
    loaded_data = _get_json_file_data()

    random_shuffle(loaded_data)

    start_from_index = 0
    for index, file_out in enumerate(output_files):
        # calculating no of examples in given file.
        sample_count = int(len(loaded_data) * split_fractions[index])
        print(file_out + ": " + str(sample_count))
        # computing corresponding index.
        end_index = min(start_from_index + sample_count, len(loaded_data))
        # converting the block to binary.
        _convert_json_to_binary(loaded_data[start_from_index:end_index], file_out)
        # updating index.
        start_from_index = end_index

def _convert_json_to_binary(input_text_files, output_filename):
    """ converting the split sections of json filenames to
        binary for training and storing in output file """
    with open(output_filename, 'wb') as out_file:
        for document in input_text_files:
            # extracting body and abstract out of document.
            body = document['body']
            abstract = document['abstract']

            body = body.replace('\n',' ').replace('\t',' ')
            abstract = abstract.replace('\n',' ').replace('\t',' ')

            sentences = sent_tokenize(body)
            body = '<d><p>' + ' '.join(['<s>' + sentence + '</s>' for sentence\
                                        in sentences]) + '</p></d>'
            # encoding in unicode.
            body = body.encode('utf8')
            # encoding abstract.
            abstract = '<d><p><s>' + abstract + ' </s></p></d>'
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
    assert FLAGS.command and FLAGS.in_file and FLAGS.out_files
    # parsing command line variables.
    output_filenames = FLAGS.out_files.split(',')

    if FLAGS.command == 'text_to_binary':
        assert FLAGS.split
        split_fractions = [float(s) for s in FLAGS.split.split(',')]
        assert len(output_filenames) == len(split_fractions)
        _text_to_binary(output_filenames, split_fractions)
    elif FLAGS.command == 'text_to_vocabulary':
        assert len(output_filenames) == 1
        _text_to_vocabulary(output_filenames[0])


if __name__ == '__main__':
    tf.app.run()
