import unittest
from Utility import convert_full_data
import seq2seq_attention
import tensorflow as tf
import batch_reader
import data
import seq2seq_attention_decode
import seq2seq_attention_model


def test_text_to_vocab_function(input_folder, output_file):
    return convert_full_data._text_to_vocabulary(input_folder, output_file)

def test_text_to_binary_function(input_folder, output_files, split_fractions):
    return convert_full_data._text_to_binary(input_folder, output_files, split_fractions)

def dummy_model_run(vocab_path, data_path):
    vocab = data.Vocab(vocab_path, 1000)
    assert vocab.CheckVocab(data.PAD_TOKEN) > 0
    assert vocab.CheckVocab(data.UNKNOWN_TOKEN) >= 0
    assert vocab.CheckVocab(data.SENTENCE_START) > 0
    assert vocab.CheckVocab(data.SENTENCE_END) > 0
    hps = seq2seq_attention_model.HParams(
        mode="train",  # train, eval, decode
        min_lr=0.01,  # min learning rate.
        lr=0.15,  # learning rate
        batch_size=1,
        enc_layers=4,
        enc_timesteps=1,
        dec_timesteps=1,
        min_input_len=1,  # discard articles/summaries < than this
        num_hidden=10,  # for rnn cell
        emb_dim=10,  # If 0, don't use embedding
        max_grad_norm=2,
        num_softmax_samples=20)  # If 0, no sampled softmax.

    batcher = batch_reader.Batcher(
        data_path, vocab, hps, 'article',
        'abstract', 100,
        10, bucketing=False,
        truncate_input=False)
    tf.set_random_seed(111)
    model = seq2seq_attention_model.Seq2SeqAttentionModel(hps, vocab, num_gpus=0)
    seq2seq_attention._Train(model, batcher)
    return True

class MyTest(unittest.TestCase):
    def test(self):
        self.assertTrue(test_text_to_vocab_function('.test_files/raw_files/', ".test_files/vocab"))
        self.assertTrue(test_text_to_binary_function('.test_files/raw_files/', ".test_files/splits", 0.8))
        self.assertTrue(dummy_model_run(".test_files/vocab", ".test_files/splits/train.bin"))
