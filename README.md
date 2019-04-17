# Abstractive Summarization

![Quality Gate Status](http://35.185.78.184/api/project_badges/measure?project=DocumentClassificationAPI&metric=alert_status)  

## Some background
 - ** Extractive summarization : ** systems form summaries by copying parts of the source text
through some measure of importance and then combine those part/sentences together to
render a summary. Importance of sentence is based on linguistic and statistical features.

- ** Abstractive summarization : ** systems generate new phrases, possibly rephrasing or using
words that were not in the original text. Naturally abstractive approaches are harder. For
perfect abstractive summary, the model has to first truly understand the document and then
try to express that understanding in short possibly using new words and phrases. Much
harder than extractive. Has complex capabilities like generalization, paraphrasing and incorporating real-world knowledge

## Introduction
The core model is the traditional sequence-to-sequence model with attention. It is customized (mostly inputs/outputs) for the text summarization task. Here we use [textsum model](https://github.com/tensorflow/models/tree/master/research/textsum), one of the research models of tensorflow, and tune it to our requirements. The model was trained on **PMC Data**, with the bodies and the corresponding abstracts of publications.

The initial training of the model (on 10000 samples max.) was done on machine with specifications:

> 16 GB GPU RAM (Single GPU)
> 16 GB CPU RAM (8 cores)
> Ubuntu 16.04 LTS OS

_Caution:_ The training on very large data (with large vocab) samples might need multiple-gpu and multiple-machines.

## Train Dataset Discription
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

1. Language :  Python 3.5.2 and higher
2. Install Required Packages By : `pip install -r requirements.txt`
3. Nvidia CUDA compiler : nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2017 NVIDIA Corporation
Built on Fri_Sep__1_21:08:03_CDT_2017
Cuda compilation tools, release 9.0, V9.0.176

## Reference Links
Here are some materials of study which can give us a deeper understanding to the problem:
* Official google's blog post [link](https://ai.googleblog.com/2016/08/text-summarization-with-tensorflow.html)
* Sequence to sequence learning paper - [link](https://arxiv.org/pdf/1409.3215.pdf)
* Textsum documentation - [link](http://textsum-document.readthedocs.io/en/latest/textsum.html)
* Pavel Surmenok blog - [link](http://pavel.surmenok.com/2016/10/15/how-to-run-text-summarization-with-tensorflow/)

## Hyperparameter Configuration:

Following is the configuration for the best trained model on Gigaword dataset:

- **batch_size** : 4
- **bidirectional encoding layer** : 4
- **article length** : first 2 sentences, total words within 200
- **summary length** : total words within 35.
- **word embedding size** : 256
- **LSTM hidden units** : 256
- **Sampled softmax** : 8192
- **vocabulary size** : Most frequent 1000k words from dataset's article and summaries.

> You may chnage the configuration by passing parameters to preprocessing and training file

## Data Format
The folder contain many such files having json format as give below:
```json
[{"body": "\nionospheric sounding is significant for studying the ionosphere which is an important part of the earth\u2019s upper atmosphere1. characteristic parameters like electron density, electron temperature, total electron content (tec) etc. can be obtained by equipment such as ionosonde, sounding rocket and radar as well as global navigation satellite system (gnss)234567. gnss provides an opportunity for sounding the ionosphere with high accuracy, temporal and spatial resolution. ,....................", "abstract": "\nglobal ionospheric products of vertical total electron content (vtec) derived from gnss measurements may have low accuracy over oceans and southern latitudes where there are not rich observations. project loon provides a great opportunity to enhance the measurements over those areas............."}]
```
The body will be a full size document or multiple document combined togather. The abstract is smaller concise form of the body.
some sample files files , vocab and train-test split are provided at `test_files`.

## Package Structure
  ```js
  ├── AUTHORS
  ├── batch_reader.py
  ├── beam_search.py
  ├── BUILD
  ├── data_convert_example.py
  ├── data.py
  ├── LICENSE
  ├── README.md
  ├── README.md~
  ├── requirements.txt
  ├── sample_data
  │   ├── sample_1041.json
  │   ├── sample_1043.json
  │   ├── sample_1110.json
  │   ├── sample_1116.json
  │   ├── sample_1143.json
  |
  ├── sample_processed_data
  │   ├── train.bin
  |   ├── test.bin
  |   ├── validation.bin
  │   └── vocab
  ├── seq2seq_attention_decode.py
  ├── seq2seq_attention_model.py
  ├── seq2seq_attention.py
  ├── seq2seq_lib.py
  └── utility
      ├── analyse_full_data.py
      ├── arrange.py
      ├── convert_full_data.py
      ├── data_convert.py
      ├── dump_data_from_server.py
      ├── dump_full_data_from_server.py
      ├── full_db_stats.txt
      ├── multi_convert_full_data.py
      └── tableDisplay.py
  ```

## How To Run
### Making vocabulary
Creating vocabulary
```python
python utility/convert_full_data.py --command text_to_vocabulary \
    --in_folder ~/folder/sample_files \
    --out_files ~/dest_folder/vocab
    --max_words number_of_words

python ./utility/convert_full_data.py --command text_to_vocabulary --in_folder ./test_files/raw_files/ --out_files  test_files/raw_files/vocab
```
> ### Other options
```python
utility/convert_full_data.py:
    --command: Either text_to_vocabulary or text_to_binary.Specify FLAGS.in_directories accordingly.
    --in_folder: path to input json data file
    --max_words: Define the max number of words to consider in vocab
    --out_files: comma seperated paths to files
    --split: comma separated fractions of data
```
### Splitting data into train test and validate
Splittng data into train.bin, test.bin, validation.bin

```python
python utility/convert_full_data.py --command text_to_binary  --in_folder ~/folder/sample_files  --out_files train.bin,validation.bin,test.bin --split 0.8,0.15,0.05

Example : convert_full_data.py --command text_to_binary     --in_folder test_files/raw_files  --out_files test_files/splits/train.bin,test_files/splits/validation.bin,test_files/splits/test.bin --split 0.8,0.15,0.05

```
### Run the training
```python
python seq2seq_attention.py --mode=train --article_key article --abstract_key abstract --data_path test_files/splits/train.bin --vocab_path test_files/vocab/vocab --log_root textsum/log_root --train_dir textsum/log_root/train --truncate_input True batch_size 8
```
> ### Other options
``` bash
seq2seq_attention.py:
      --abstract_key: tf.Example feature key for abstract.
      --article_key: tf.Example feature key for article.
      --beam_size: beam size for beam search decoding.
      --checkpoint_secs: How often to checkpoint.
      --data_path: Path expression to tf.Example.
      --decode_dir: Directory for decode summaries.
      --eval_dir: Directory for eval.
      --eval_interval_secs: How often to run eval.
      --log_root: Directory for model root.
      --max_abstract_sentences: Max number of first sentences to use from the abstract
      --max_article_sentences: Max number of first sentences to use from the article
      --max_run_steps: Maximum number of run steps.
      --max_words: Maximum number of words to consider in vocabulary.
      --mode: train/eval/decode mode
      --num_gpus: Number of gpus used.
      --random_seed: A seed value for randomness.
      --train_dir: Directory for train.
      --[no]truncate_input: Truncate inputs that are too long. If False, examples that are too long are discarded.
      --[no]use_bucketing: Whether bucket articles of similar length.
      --vocab_path: Path expression to text vocabulary file
```

### Run the eval
```python
$ python seq2seq_attention.py --mode=eval --article_key article --abstract_key abstract --data_path data/test.bin --vocab_path data/vocab --log_root=textsum/log_root --eval_dir textsum/log_root/eval --truncate_input True
```
### Decoding - Summarize unknown docs
```python
# Run the decode. Run it when the model is mostly converged.
python seq2seq_attention.py  --mode=decode --article_key article --abstract_key abstract  --data_path data/test.bin --vocab_path data/vocab --log_root=textsum/log_root --decode_dir textsum/log_root/decode --truncate_input True --beam_size 8
```

## Results
### version 1.1:
The output of every article were ***very small*** and ***repetitive***, replete with ```<UNK>``` tokens. Most of the article abstracts were dropped as `--truncate_input = False`. The training overfitted badly. Training was done on ```default parameters``` for the first round of results:
#### specs:
- samples = 10000
- truncated_input  = False
- training time = 1 day(approx.)
- avg. training loss = 1.2e-4
- avg. validation loss = 12.5
#### aim:
- to reduce the number of ``<UNK>`` tokens
- prevent overfitting

### version 1.2:
To prevent overfitting, we changed `--truncate_input = True`, so that it does not drop the samples and rather truncates it. The output for every article were very ***small*** and ***repetitive***, replete with `<UNK>` tokens. this time, it did overfit the data but with less magnitude. Training was done on ```default parameters``` to see the effect of changed parameter:
#### specs:
- samples = 10000
- truncated_input  = True
- training time = 1 day (approx.)
- avg. training loss = 2.5
- avg. validation loss = 9.5

#### aim:
- First, try removing the `<UNK>` tokens on  smaller data
- Then, scale up the model by feeding it with larger data corpus to prevent overfitting

### version 2.1:
We successfully ***removed all the `<UNK>`*** tokens by changing the binarisation script a little. The results are still **repetitive and small** and is still overfitting.

#### specs:
- samples = 1100
- truncated_input  = True
- training time = 12 hours (approx.)
- avg. training loss = .8
- avg. validation loss = 6.9

#### aim:
- Having the `<UNK>` tokens removed, we just need to train it on larger dataset to prevent overfitting.

#### Evaluating Results
For calculating ROUGE scores, we use *files2rouge* module. For installing files2rouge, please follow the instruction line-by-line as mentioned in the [repository](https://github.com/pltrdy/files2rouge).
For rouge score calculation, we can just run
``` bash
files2rouge summ_pubmed.test.target decode_pubmed.txt
```
### Results
After having the encoder and tokenizer integrated, I could only train on 200000 samples. The model did not overfit a lot. Here are the results,
```
ROUGE score calculated between gold summary and predictions
---------------------------------------------
1 ROUGE-1 Average_R: 0.09928 (95%-conf.int. 0.08813 - 0.11111)
1 ROUGE-1 Average_P: 0.89567 (95%-conf.int. 0.88064 - 0.90830)
1 ROUGE-1 Average_F: 0.16730 (95%-conf.int. 0.15267 - 0.18251)
---------------------------------------------
1 ROUGE-2 Average_R: 0.05326 (95%-conf.int. 0.04799 - 0.05877)
1 ROUGE-2 Average_P: 0.51645 (95%-conf.int. 0.49365 - 0.53864)
1 ROUGE-2 Average_F: 0.09159 (95%-conf.int. 0.08390 - 0.09969)
---------------------------------------------
1 ROUGE-L Average_R: 0.06384 (95%-conf.int. 0.05825 - 0.06994)
1 ROUGE-L Average_P: 0.62875 (95%-conf.int. 0.60653 - 0.65057)
1 ROUGE-L Average_F: 0.10991 (95%-conf.int. 0.10157 - 0.11832)
```
**Note** : As expected, give the body text is long, will have a very high accuracy but low recall.

##  Future Implementations

 - [ ] To include the **memory compressed attention** in our model like that implemented in wikisum problem of tensor2tensor, that can handle long sequences. [Understanding-handling-long-sequences-wikisum](To%20include%20the%20memory%20compressed%20attention%20in%20our%20model%20like%20that%20%20%20%20%20%20%20%20implemented%20in%20wikisum%20problem,%20that%20can%20handle%20long%20sequences.).
 - [ ] Training on **larger training data** set for longer time. As seen for the summarization model, it needs larger training data and longer training time to work comendably well. With this, the repetitions are reduced (less overfitting) and the sequences generated are **more accurate**. Unfortunately I was not able to tackle the issue by training on ml-engine or on our GPU due to time constraint.
 - [ ] The biomedical vocabulary generation takes **a lot of time**. Typically, with normal internet speed, for a vocab generation of 2 lacs samples, it takes around 2.5 days. It can be made faster using multiple CPU cores or even better, if we could use GPU for the same.   
