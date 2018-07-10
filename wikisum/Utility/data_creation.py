import re
import hashlib
import io
import os
import tarfile
import six
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow as tf

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

# # Techniques for data prep from See et al. (2017)
# dm_single_close_quote = u"\u2019"  # unicode
# dm_double_close_quote = u"\u201d"
# # Acceptable ways to end a sentence.
# END_TOKENS = [
#     u".", u"!", u"?", u"...", u"'", u"`", u"\"", dm_single_close_quote,
#     dm_double_close_quote, u")"
# ]


def _get_json_file_data(file):
    """ reads the json file and returns the loaded data"""
    loaded_data = []
    with open(file, mode='r') as file_json:
        loaded_data = json.load(file_json)
    for document in loaded_data:
        return document

def _get_file_list(input_folderpath):
    input_files = tf.gfile.Glob(input_folderpath + '*')
    return input_files

def example_generator():
    pass

def write_text_to_files(input_files, out_dir, is_training):
  """Write text to files."""

    def write_to_file(input_files, out_dir, filename):
    with io.open(os.path.join(tmp_dir, filename + ".source"), "w") as fstory:
      with io.open(os.path.join(tmp_dir, filename + ".target"),
                   "w") as fsummary:
        for example in example_generator(input_files, urls_path, sum_token=True):
          story, summary = _story_summary_split(example)
          fstory.write(story + "\n")
          fsummary.write(summary + "\n")

    filename = "pubmed.train" if is_training else "pubmed.dev"
    tf.logging.info("Writing %s" % filename)
    write_to_file(input_files, out_dir, filename)

    if not is_training:


    filename = "pubmed.test"
    tf.logging.info("Writing %s" % filename)
    write_to_file(input_files, test_urls_path, tmp_dir, filename)

@registry.register_problem
class SummarizePubMedArticle(text_problems.Text2TextProblem):
  """Summarize Pub Med articles to their abstracts"""

    @property
    def approx_vocab_size(self):
        return 2**16  # ~65k

    @property
    def vocab_filename(self):
        return "vocab.pubmed.%d" % self.approx_vocab_size

    @property
    def is_generate_per_split(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
        return False

    @property
    def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    # 10% evaluation data
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 9,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

    def generate_samples(self, input_dir, out_dir, dataset_split):
        del input_dir
        is_training = dataset_split == problem.DatasetSplit.TRAIN
        input_files = _get_file_list(input_dir)
        write_text_to_files(all_files)
