import unittest
from utility import convert_full_data
import os


def test_text_to_vocab_function(input_folder, output_file):
    """
    test for vocab generation
    """
    return convert_full_data._text_to_vocabulary(input_folder, output_file)

def test_text_to_binary_function(input_folder, output_files, split_fractions):
    """
    test for binarization
    """
    return convert_full_data._text_to_binary(input_folder, output_files, split_fractions)

def cleanup():
    """
    cleaning up file generated by test
    """
    os.remove("test_files/splits/train.bin")
    os.remove("test_files/splits/test.bin")
    os.remove("test_files/vocab/vocab")



class MyTest(unittest.TestCase):
    def test(self):
        self.assertTrue(test_text_to_vocab_function('test_files/raw_files/', "test_files/vocab/vocab"))
        self.assertTrue(test_text_to_binary_function('test_files/raw_files/', ["test_files/splits/train.bin","test_files/splits/test.bin"], [0.8,0.2]))
        cleanup()
if __name__ == '__main__':
        test = MyTest()
        test.test()
