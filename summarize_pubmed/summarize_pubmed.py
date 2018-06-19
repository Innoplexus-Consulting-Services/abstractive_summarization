"""Data generators for pubmed datasets"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import io
import os
import string
import six
import json
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
# from tensor2tensor.data_generators import tokenizer
from tensor2tensor.utils import metrics

import tensorflow as tf

# flags = tf.flags
# FLAGS = flags.FLAGS

# flags.DEFINE_string("vocab_path", '/home/mainak.chain/Wikisum_dev/data/vocab/', "Directory to read vocab from.")

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

# Techniques for data prep from See et al. (2017)
dm_single_close_quote = u"\u2019"  # unicode
dm_double_close_quote = u"\u201d"
# Acceptable ways to end a sentence.
END_TOKENS = [
    u".", u"!", u"?", u"...", u"'", u"`", u"\"", dm_single_close_quote,
    dm_double_close_quote, u")"
]

def example_generator(file):
    """ reads the json file and returns the loaded data"""
    loaded_data = []
    with open(file, mode='r') as file_json:
        loaded_data = json.load(file_json)
        for document in loaded_data:
            body = re.sub(' +',' ',document['body'].strip().replace('\n',' '))
            abstract = re.sub(' +',' ',document['abstract'].strip().replace('\n',' '))
            return abstract, body

def _get_file_list(input_folderpath):
    path = input_folderpath
    input_files = os.listdir(path)
    return input_files

def write_raw_text_to_files(input_files, data_dir, tmp_dir, is_training):
    """Write text to files."""

    def write_to_file(input_files, data_dir, tmp_dir, filename):
        with io.open(os.path.join(data_dir, filename + ".source"), "w") as fbody:
            with io.open(os.path.join(data_dir, filename + ".target"),
                       "w") as fabstract:
                for file in input_files:
                    if os.path.isfile(os.path.join(tmp_dir, file)):
                        abstract, body = _get_data(os.path.join(tmp_dir, file))

                        fbody.write(body + "\n")
                        fabstract.write(abstract + "\n")

    filename = "pmc.train" if is_training else "pmc.dev"
    tf.logging.info("Writing %s" % filename)
    write_to_file(input_files, data_dir, tmp_dir, filename)

    if not is_training:
        test_input_files = _get_file_list(os.path.join(tmp_dir, 'test'))
        filename = "pmc.test"
        tf.logging.info("Writing %s" % filename)
        write_to_file(test_input_files, data_dir, os.path.join(tmp_dir, 'test'), filename)

@registry.register_problem
class SummarizePubmed(text_problems.Text2TextProblem):
    """Summarize Pub Med articles to their abstracts"""

    # def hparams(self, defaults, unused_model_hparams):
    #     return super(SummarizePubmed, self).hparams(defaults, unused_model_hparams)

    def example_reading_spec(self):
        return super(SummarizePubmed, self).example_reading_spec()

    def preprocess_example(self, example, mode, hparams):
        return problem.Problem.preprocess_example(self, example, mode, hparams)

    # @property
    # def approx_vocab_size(self):
    #     return 2**15  # ~32k

    # @property
    # def vocab_filename(self):
    #     return "vocab.%s.%d" % ('cnnpubmed', self.approx_vocab_size)

    @property
    def is_generate_per_split(self):
        return True

    def generate_text_for_vocab(self, data_dir, tmp_dir):
        del data_dir
        input_files = _get_file_list(tmp_dir)
        for file in input_files:
            if os.path.isfile(os.path.join(tmp_dir, file)):
                abstract, body = _get_data(os.path.join(tmp_dir, file))
                yield abstract + " " + body

    # def feature_encoders(self, data_dir):
    #     vocab_filename = os.path.join(data_dir, self.vocab_filename)
    #     encoder = text_encoder.SubwordTextEncoder(vocab_filename)
    #     # Shared encoder for inputs and targets
    #     return {"inputs": encoder, "targets": encoder}

    def eval_metrics(self):
        return super(WikisumBase, self).eval_metrics() + [
          metrics.Metrics.ROUGE_2_F, metrics.Metrics.ROUGE_L_F
        ]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir

        #data stats
        avg_b = 28000   #body length average
        avg_a = 1400    #abstract length average
        sig_b = 19000   #deviation of body length
        sig_a = 600 #deviation of abstract length

        is_training = dataset_split == problem.DatasetSplit.TRAIN
        input_files = _get_file_list(tmp_dir)
        write_raw_text_to_files(input_files, data_dir, tmp_dir, is_training)
        for file in input_files:
            if os.path.isfile(os.path.join(tmp_dir, file)):
                abstract, body = _get_data(os.path.join(tmp_dir, file))

            if ((avg_b - 2*sig_b) <= len(body) <= (avg_b + 2*sig_b)) \
                and ((avg_a - 2*sig_a) <= len(abstract) <= (avg_a + 2*sig_a)):
                yield {"inputs": body, "targets": abstract}
