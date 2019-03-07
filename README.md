# Textsum for Abstractive Summarization Documentation (Life-Science Domain)

## Introduction
The core model is the traditional sequence-to-sequence model with attention. It is customized (mostly inputs/outputs) for the text summarization task. Here we use [textsum model](https://github.com/tensorflow/models/tree/master/research/textsum), one of the research models of tensorflow, and tune it to our requirements. The model was trained on **PMC data**, with the bodies and the corresponding abstracts of publications.

The initial training of the model (on 10000 samples max.) was done on machine with specifications:

> 16 GB GPU RAM (Single GPU)
> 16 GB CPU RAM (8 cores)
> Ubuntu 16.04 LTS OS

_Caution:_ The training on very large data (with large vocab) samples might need multiple-gpu and multiple-machines.

## Dataset Discription

The original textsum model was trained on Gigaword dataset as described in [ Rush et al. A Neural Attention Model for Sentence Summarization](https://arxiv.org/abs/1509.00685). Gigaword contains around 9.5 million news articles sourced from various domestic and international news services over the last two decades. We tried to train on dataset of similar size (7 lacs documents of the 16 lacs documents present in testpmc database). The articles with non-null bodies and abstracts in the database stats are as follows:

| Stats-Criteria 	|     Abstract     	|       Body       	|
|:--------------:	|:----------------:	|:----------------:	|
|       Min      	|         3        	|         3        	|
|       Max      	|      122394      	|      2822028     	|
|      Mean      	|       1444       	|       28182      	|
|    Variance    	|     340269.89    	|     3.52e+08     	|
|  1st-Quartile  	|      1102.0      	|      16986.0     	|
|     Median     	|      1450.0      	|      25933.0     	|
|  3rd-Quartile  	|      1796.0      	|      36467.0     	|
|    Skewness    	|        13        	|        14        	|
|    Kurtosis    	|       2494       	|       1102       	|

## Environment Setup

Prerequisite: Install [tensorflow](http://www.python36.com/install-tensorflow-using-official-pip-pacakage/) and [bazel](https://docs.bazel.build/versions/master/install-ubuntu.html). The environment setup of running the textsum model is as follows (Install the corresponding dependencies if necessary) :

```sh
$ python --version
Python 3.5.2

$ pip --version
pip 10.0.1 from .../lib/python3.5/site-packages/pip (python 3.5)

#important packages
$ pip freeze
numpy==1.14.3
protobuf==3.5.2.post1
python-utils==2.3.0
six==1.11.0
tensorboard==1.8.0
tensorflow-gpu==1.8.0

$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2017 NVIDIA Corporation
Built on Fri_Sep__1_21:08:03_CDT_2017
Cuda compilation tools, release 9.0, V9.0.176

$ bazel version
Build label: 0.13.1
```
## Reference Materials
Here are some materials of study which can give us a deeper understanding to the problem:
* Official google's blog post [link](https://ai.googleblog.com/2016/08/text-summarization-with-tensorflow.html)
* Sequence to sequence learning paper - [link](https://arxiv.org/pdf/1409.3215.pdf)
* Textsum documentation - [link](http://textsum-document.readthedocs.io/en/latest/textsum.html)
* Pavel Surmenok blog - [link](http://pavel.surmenok.com/2016/10/15/how-to-run-text-summarization-with-tensorflow/)

## Hyperparameter Configuration:

Following is the configuration for the best trained model on Gigaword:

- **batch_size** : 4
- **bidirectional encoding layer** : 4
- **article length** : first 2 sentences, total words within 200
- **summary length** : total words within 35.
- **word embedding size** : 256
- **LSTM hidden units** : 256
- **Sampled softmax** : 8192
- **vocabulary size** : Most frequent 1000k words from dataset's article and summaries.

## How To Run

```shell
# cd to your workspace
# 1. Clone the textsum code to your workspace 'textsum' directory.
# 2. Create an empty 'WORKSPACE' file in your workspace.
# 3. Move the train/eval/test data to your workspace 'data' directory.
#    In the following example, I named the data training-*, test-*, etc.
#    If your data files have different names, update the --data_path.
#    If you don't have data but want to try out the model, copy the toy
#    data from the textsum/data/data to the data/ directory in the workspace.
$ ls -R
.:
data  textsum  WORKSPACE

./data:
vocab   train.bin   test.bin   validation.bin

./textsum:
batch_reader.py       beam_search.py       BUILD    README.md            seq2seq_attention_model.py  data
data.py  seq2seq_attention_decode.py  seq2seq_attention.py        seq2seq_lib.py

./textsum/data:
data  vocab

./textsum/Utility:
analyse_full_data.py        convert_full_data.py                         dump_data_from_server.py       full_db_stats.txt       tableDisplay.py
arrange.py      data_convert.py     dump_full_data_from_server.py         multi_convert_full_data.py

#creating vocab, train.bin, test.bin, validation.bin
$ python convert_full_data.py --command text_to_vocabulary \
    --in_folder ~/folder/sample_files \
    --out_files ~/dest_folder/vocab
    --max_words number_of_words
    
$ python convert_full_data.py --command text_to_binary \
    --in_folder ~/folder/sample_files \
    --out_files train.bin,validation.bin,test.bin
    --split 0.8,0.15,0.05

#to build using bazel
$ bazel build -c opt --config=cuda textsum/...

# Run the training.
$ bazel-bin/textsum/seq2seq_attention \
    --mode=train \
    --article_key=article \
    --abstract_key=abstract \
    --data_path=data/train.bin \
    --vocab_path=data/vocab \
    --log_root=textsum/log_root \
    --train_dir=textsum/log_root/train
    --truncate_input=True

# Run the eval. Try to avoid running on the same machine as training.
$ bazel-bin/textsum/seq2seq_attention \
    --mode=eval \
    --article_key=article \
    --abstract_key=abstract \
    --data_path=data/test.bin \
    --vocab_path=data/vocab \
    --log_root=textsum/log_root \
    --eval_dir=textsum/log_root/eval
    --truncate_input=True

# Run the decode. Run it when the model is mostly converged.
$ bazel-bin/textsum/seq2seq_attention \
    --mode=decode \
    --article_key=article \
    --abstract_key=abstract \
    --data_path=data/test.bin \
    --vocab_path=data/vocab \
    --log_root=textsum/log_root \
    --decode_dir=textsum/log_root/decode \
    --truncate_input=True
    --beam_size=8
```
## Results
##### version 1.1:
The output of every article were ***very small*** and ***repetitive***, replete with ```<UNK>``` tokens. Most of the article abstracts were dropped as `--truncate_input = False`. The training overfitted badly. Training was done on ```default parameters``` for the first round of results: 
###### specs:
- samples = 10000
- truncated_input  = False 
- training time = 1 day(approx.)
- avg. training loss = 1.2e-4
- avg. validation loss = 12.5
###### aim:
- to reduce the number of ``<UNK>`` tokens
- prevent overfitting 

##### version 1.2:
To prevent overfitting, we changed `--truncate_input = True`, so that it does not drop the samples and rather truncates it. The output for every article were very ***small*** and ***repetitive***, replete with `<UNK>` tokens. this time, it did overfit the data but with less magnitude. Training was done on ```default parameters``` to see the effect of changed parameter: 
###### specs:
- samples = 10000
- truncated_input  = True
- training time = 1 day (approx.)
- avg. training loss = 2.5
- avg. validation loss = 9.5

###### aim:
- First, try removing the `<UNK>` tokens on  smaller data
- Then, scale up the model by feeding it with larger data corpus to prevent overfitting

##### version 2.1:
We successfully ***removed all the `<UNK>`*** tokens by changing the binarisation script a little. The results are still **repetitive and small** and is still overfitting. 

###### specs:
- samples = 1100
- truncated_input  = True
- training time = 12 hours (approx.)
- avg. training loss = .8
- avg. validation loss = 6.9

###### aim:
- Having the `<UNK>` tokens removed, we just need to train it on larger dataset to prevent overfitting.





 



