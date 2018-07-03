"""Data generators for pubmed dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import io
import os
import string
import json
from nltk import sent_tokenize

from gensim.summarization.summarizer import summarize

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
# from tensor2tensor.data_generators import tokenizer
from tensor2tensor.utils import metrics
from progressbar import progressbar

try:
    from . import biomedical_encoder
except:
    import biomedical_encoder

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_float("extractive_summarize_ratio", 0.15,
                     "Ratio of output text to input text for extractive summarization")
flags.DEFINE_integer("vocab_size",2**15,"Size of vocabulary to use")
flags.DEFINE_string("encode_method", 'biotokentext', "Encoding method (subword/tokentext)")
# End-of-sentence marker.
EOS = text_encoder.EOS_ID

UNKNOWN_TOKEN = "<UNK>"
RESERVED_TOKENS = text_encoder.RESERVED_TOKENS.append(UNKNOWN_TOKEN)
UNK_ID = text_encoder.RESERVED_TOKENS.index(UNKNOWN_TOKEN)
text_encoder.NUM_RESERVED_TOKENS = len(text_encoder.RESERVED_TOKENS)

# Techniques for data prep from See et al. (2017)
dm_single_close_quote = u"\u2019"  # unicode
dm_double_close_quote = u"\u201d"
# Acceptable ways to end a sentence.
END_TOKENS = [
    u".", u"!", u"?", u"...", u"'", u"`", u"\"", dm_single_close_quote,
    dm_double_close_quote, u")"
]
def preprocess_text(text):
    text = text.lower()
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

def extractive_summarize(text, ratio, split):
    summarized_text = ' '.join(summarize(text, ratio=ratio, split=split))
    return summarized_text

def _get_data(file):
    """ reads the json file and returns the loaded data"""
    loaded_data = []
    with open(file, mode='r') as file_json:
        loaded_data = json.load(file_json)
        for document in loaded_data:
            body = re.sub(' +',' ',document['body'].strip().replace('\n',' '))
            abstract = re.sub(' +',' ',document['abstract'].strip().replace('\n',' '))
            #extractive Summarization
            body = preprocess_text(body)
            extractive_summary = extractive_summarize(body,
                                        ratio=FLAGS.extractive_summarize_ratio, split=True)
            extractive_summary = preprocess_text(extractive_summary)
            abstract = preprocess_text(abstract)
            return abstract, extractive_summary, body

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
                with io.open(os.path.join(data_dir, filename + ".extractive_summ"),
                           "w") as fextract:

                    for file in progressbar(input_files):
                        if os.path.isfile(os.path.join(tmp_dir, file)):
                            abstract, extractive_summary, body = _get_data(os.path.join(tmp_dir, file))

                            fbody.write(body + "\n")
                            fabstract.write(abstract + "\n")
                            fextract.write(extractive_summary + "\n")


    filename = "summ_pubmed.train" if is_training else "summ_pubmed.dev"
    tf.logging.info("Writing %s" % filename)
    write_to_file(input_files, data_dir, tmp_dir, filename)

    if not is_training:
        test_input_files = _get_file_list(os.path.join(tmp_dir, 'test'))
        filename = "summ_pubmed.test"
        tf.logging.info("Writing %s" % filename)
        write_to_file(test_input_files, data_dir, os.path.join(tmp_dir, 'test'), filename)

class AdditionalVocabType(text_problems.VocabType):
  """Available text vocabularies."""
  SUBWORD = "subwords"
  BIOTOKEN = "biotokens"

@registry.register_problem
class SummarizePubmed(text_problems.Text2TextProblem):
    """Summarize Pub Med articles to their abstracts"""

    # def hparams(self, defaults, unused_model_hparams):
    #     return super(SummarizePubmed, self).hparams(defaults, unused_model_hparams)

    def example_reading_spec(self):
        return super(SummarizePubmed, self).example_reading_spec()

    def preprocess_example(self, example, mode, hparams):
        return problem.Problem.preprocess_example(self, example, mode, hparams)

    @property
    def approx_vocab_size(self):
        return FLAGS.vocab_size

    @property
    def additional_reserved_tokens(self):
        return [UNKNOWN_TOKEN]

    @property
    def oov_token(self):
        """Out of vocabulary token. Only for VocabType.TOKEN."""
        if FLAGS.encode_method == 'subword':
            return None
        else:
            #Needs to be handled. Using None for the time being
            print( UNKNOWN_TOKEN)
            return UNKNOWN_TOKEN

    @property
    def vocab_type(self):
        if FLAGS.encode_method == 'subword':
            return text_problems.VocabType.SUBWORD
        elif FLAGS.encode_method == 'biotokentext':
            return AdditionalVocabType.BIOTOKEN
        else:
            raise ValueError("Unrecognized VocabType")

    @property
    def is_generate_per_split(self):
        return True

    @property
    def vocab_filename(self):
        if self.vocab_type == AdditionalVocabType.SUBWORD:
            return "vocab.%s.%d.%s" % (self.dataset_filename(),
                                    self.approx_vocab_size,
                                    VocabType.SUBWORD)
        elif self.vocab_type == AdditionalVocabType.BIOTOKEN:
            return "vocab.%s.%s" % (self.dataset_filename(), AdditionalVocabType.BIOTOKEN)


    def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
        if self.vocab_type == AdditionalVocabType.SUBWORD:
            if force_get:
                vocab_filepath = os.path.join(data_dir, self.vocab_filename)
                encoder = text_encoder.SubwordTextEncoder(vocab_filepath)
            else:
                encoder = generator_utils.get_or_generate_vocab_inner(
                    data_dir, self.vocab_filename, self.approx_vocab_size,
                    self.generate_text_for_vocab(data_dir, tmp_dir),
                    max_subtoken_length=self.max_subtoken_length,
                    reserved_tokens=(
                        text_encoder.RESERVED_TOKENS + self.additional_reserved_tokens))
        elif self.vocab_type == AdditionalVocabType.BIOTOKEN:
            vocab_filename = os.path.join(data_dir, self.vocab_filename)
            encoder = biomedical_encoder.BioMedicalEncoder(vocab_filename,
                                                  replace_oov=self.oov_token)
        else:
            raise ValueError("Unrecognized VocabType")
        return encoder

    # def generate_text_for_vocab(self, data_dir, tmp_dir):
    #     input_files = _get_file_list(tmp_dir)
    #     for file in input_files:
    #         if os.path.isfile(os.path.join(tmp_dir, file)):
    #             abstract, extractive_summary, body = _get_data(os.path.join(tmp_dir, file))
    #             yield abstract + " " + extractive_summary


    # def feature_encoders(self, data_dir):
    #     vocab_filename = os.path.join(data_dir, self.vocab_filename)
    #     encoder = text_encoder.SubwordTextEncoder(vocab_filename)
    #     # Shared encoder for inputs and targets
    #     return {"inputs": encoder, "targets": encoder}

    def eval_metrics(self):
        return [metrics.Metrics.ROUGE_2_F, metrics.Metrics.ROUGE_L_F]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):

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
                abstract, extractive_summary, body = _get_data(os.path.join(tmp_dir, file))

            if ((avg_b - 2*sig_b) <= len(body) <= (avg_b + 2*sig_b)) \
                and ((avg_a - 2*sig_a) <= len(abstract) <= (avg_a + 2*sig_a)):
                yield {"inputs": extractive_summary, "targets": abstract}
