# Text Summarization Documentation

Text summarization is one of the crucial problem in the recent time and I am really fortunate to have worked on this project, which happens to be right at the cutting edge.  [Abstractive Text Summarization Using Sequence-to-Sequence RNNs and Beyond](https://arxiv.org/abs/1602.06023) is one of the pioneering work in the field. We can get a brief idea about the approaches and related work from [Text Summarization Techniques](https://arxiv.org/pdf/1707.02268.pdf) and [Github-repository](https://github.com/icoxfog417/awesome-text-summarization).  Various papers and there implements can be found [here](https://paperswithcode.com/search?q=summarization).  

## Data Description
As per our use case, I used PubMed data available with us. For our task, the body of the document was the text to summarize and it's abstract was the summarized text. The complete data can be found [here](https://drive.google.com/open?id=1KIKrgDS3jQm8scYi0TGmQXbQGWubCLQ2). 

The data can be dumped using the scripts in [gitlab-repo](https://gitlab.innoplexus.de/Innoplexus-Consulting-Services/DocumentClassificationAPI/tree/transformer_model/dump_data). The data analysing scripts and stats are also pushed there. 

The full dataset contains of 7 lacs document with body-text and it's corresponding abstracts. On average number of words in body and abstract is around 3000 and 200 respectively. Our database characters number can be summarized as :
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

## Problem Statement

To create a text summarizer based on the life science domain.  The main challenges in the project were:
* Summarizing long documents into multiple lines summary
* Preserving the sense of sentence and the **biomedical terms** in the summary generated
* The summary should imply the meaning of the whole document i.e. it should be able to provide information from various parts of the document and not just the *introduction part*.
* Tackling **unkwown words** in new summarization text.  

## Environment Specifications
The specifications are as follows:
```
$ python -V
Python 3.4.3

$ pip --version
pip 10.0.1 from /home/mainak.chain/env_wikisum/lib/python3.4/site-packages/pip (python 3.4)

#major packages
$ pip freeze
tqdm==4.23.4
pyrouge==0.1.3
tensorflow==1.8.0 //or tensorflow-gpu==1.8.0
gensim==3.4.0
six==1.11.0
tensor2tensor==1.6.3
nltk==3.3
pymongo==3.7.0

$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2017 NVIDIA Corporation	
Built on Fri_Sep__1_21:08:03_CDT_2017
Cuda compilation tools, release 9.0, V9.0.176

```
The environment used has been uploaded [here](https://drive.google.com/open?id=1oIpmX7jAbEJXliEf0EWGxG29OYLnL3bm). 

## Approaches adopted
We mainly adopted two approaches throughout the project. 
### Abstrative Summarization
- tensorflow-research model - **textsum** ([git-link](https://github.com/tensorflow/models/tree/master/research/textsum)) ([our-project-link](https://gitlab.innoplexus.de/Innoplexus-Consulting-Services/DocumentClassificationAPI/tree/textsum_dev))
- tensor2tensor **transformer model** ([git-link](https://github.com/tensorflow/tensor2tensor)) ([out-project-link](https://gitlab.innoplexus.de/Innoplexus-Consulting-Services/DocumentClassificationAPI/tree/transformer_model))
### Extractive Summarization
- Gensim Textrank summarization for summarizing our document body (of approx 270-300 sentences) to 100 sentences. The implementation is in [here](https://gitlab.innoplexus.de/Innoplexus-Consulting-Services/DocumentClassificationAPI/tree/transformer_model/summarize_pubmed).

## Walkthrough

The walkthrough various approaches has been provided here - 

 1. **Textsum** [walkthrough](https://gitlab.innoplexus.de/Innoplexus-Consulting-Services/DocumentClassificationAPI/blob/textsum_dev/README.md)   
 2.  **Tensor2tensor transformer model** 
 3. **Extractive summarisation**
 4. **BioMedicalTokenizer** - vocab generation with biomedical entities preserved.
 5.  **BioMedicalEncoder** - encoder designed to take the pregenerated vocab and encode the text accordingly.

## Problems Tackled

Some of the problems were tackled as follows 
 - [x]  Multiple line summarization . The summarization using extractive summarization generates a summary of around 100 lines. But the abstractive summarization is not able to generate such long summaries, may be because I could not train it successfully on large training sample.
 - [x] The vocab which was being generated using SubWordTokenizer, is now being generated using the BioMedicalTokenizer
 - [x] The encoding is also being done using the vocab generated using the BioMedicalTokenizer. Hence, the life-science tokens are preserved throughout the process.

# Tensor2tensor summarization Walkthrough
## Introduction
Tensorflow's module, Tensor2tensor, is a library of deep learning models and datasets designed to make deep learning more accessible. The official tensor2tensor module documentation is [here](https://github.com/tensorflow/tensor2tensor/blob/master/README.md). There are several applications to the inbuilt transformer model. For our use, I implemented our **summarize_pubmed** problem as a **Text2TextProblem** where we taking inputs are text (body of our document as input) and correponding outputs are also text (summary generated).

## Getting Started
### System and environment configuration
The system and environment config is almost same as that in textsum. The environment used has been uploaded and the link has been mentioned above. The major configurations are mentioned below:
```
$ pip freeze
redis==2.10.6
pythonrouge==0.2
numpy==1.14.4
oauth2client==4.1.2
pyrouge==0.1.3
tensorflow==1.8.0
gensim==3.4.0
tensor2tensor==1.6.3
nltk==3.3
gensim==3.4.0
files2rouge==2.0.0

$ pip -V && python -V
pip 10.0.1 from /home/mainak.chain/env_wikisum/lib/python3.4/site-packages/pip (python 3.4)
Python 3.4.3

```
### Installation
Install tensor2tensor as per the requirement
```
# Assumes tensorflow or tensorflow-gpu installed
pip install tensor2tensor

# Installs with tensorflow-gpu requirement
pip install tensor2tensor[tensorflow_gpu]

# Installs with tensorflow (cpu) requirement
pip install tensor2tensor[tensorflow]
```
### Training
For training a tensor2tensor model, first we need to convert our data in tf.Records format. The data conversion can be done by using t2t-datagen or by adding --generate_data command to t2t-trainer. If the tf.Records files and the vocab are found already in data_dir, it starts training by itself. The various directories and their uses are as follows :

 - **tmp_dir** :  stores all the data in *human-readable* form. This is where the data-conversion scripts takes the data as input to convert it into tf.Records files. This folder might have seperate structure for each problem.
 - **data_dir** : stores the tf.Records files and the vocabulary generated. This is the input for training.
 - **output_dir** : this is basically the train directory, which stores the graph, hyper-params used, model checkpoints and training curves. This is the directory to supply to tensorboard.
 - **usr_t2t_dir** : stores the user defined problems, hparams and hparams set for tuning. 

Training command with generate data samples:
```
# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
t2t-trainer \
  --data_dir=/home/user.name/data_dir \
  --tmp_dir=/home/user.name/tmp_dir \
  --problem=$PROBLEM \
  --model=transformer \
  --hparams_set=transformer_base \
  --output_dir=/home/user.name/train \
  --generate_data \
  --train_steps 10000 --eval_steps 300
```
Training the model can be done by defining two types of $PROBLEM 
 - Training with default *cnn_dailymail32k* problem
 - Training with our custom-made *summarize_pubmed*
#### Training with *cnn_dailymail32k* as problem
For training with the default cnn_dailymail32k problem, we need to convert our pubmed data in their format first and store it in tmp_dir.  First keep three folders seperately as train, test and val, each with the files you want to train, test and val on. Divide the number of files accordingly, e.g. 8:1:1 files in train,test and val folder respectively. Then run the script *convert_data_form.py* as 
```
$ python convert_data_form.py --help
optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   directory with all json files to convert.
  --output_dir OUTPUT_DIR
                        directory with all .story files to store.
  --mode MODE           train, test or val.
  --sentdiv SENTDIV     start_sent,end_sent,abs_sent

$ python convert_data_form.py --output_dir /home/hash --data_dir /home/train --mode train --sentdiv 70,30,10
```
Here, *mode* needs to be train, test or val depending on the data_dir you are supplying (/home/train, test or val). The *sentdiv* argument truncates 70 and 30 from the beginning and end of the body and 10 from the beginning of abstract. (vary and see if the system can take more of the sentences). This was done to match with the capability of the system we are training on. 

After we get the .story files and the three .txt files, move it into the tmp_dir folder with structure as, with .story files in any of the *stories* directory :
```
$ ls tmp_dir/ -R
tmp_dir/:
all_test.txt  all_train.txt  all_val.txt  cnn  dailymail

tmp_dir/cnn:
stories

tmp_dir/cnn/stories:

tmp_dir/dailymail:
stories

tmp_dir/dailymail/stories:
```
#### Training with *summarize_pubmed* problem
With summarize_pubmed problem, we can train on default *SubWord* tokenizer or our *BioMedicalTokenizer*. All we need to change is *encode_method* argument. This is specifically designed to integrate our biomedical domain in the problem. 

##### Biomedical Vocab Generation
This method of vocab generation, we keep the biomedical tokens intact. For example,
```
$ text = "Unfortunately, he is diagonised with lung cancer. The cancer is still in stage-1."
Counter({',': 1,
         '.': 2,
         'The': 1,
         'Unfortunately': 1,
         'cancer': 1,
         'diagonised': 1,
         'he': 1,
         'in': 1,
         'is': 2,
         'lung cancer': 1,
         'stage-1': 1,
         'still': 1,
         'with': 1})
```
For generating this, we can use biomedical_tokenizer, which returns a *collections.Counter()* object to a biomedical text supplied.  For generating the vocab we need to run *generate_summarize_pubmed_vocab.py* , which generates a *.biotokens* file in the data_dir. 
```
python generate_summarize_pubmed_vocab.py --tmp_dir /home/user.name/tmp/ --data_dir /home/user.name/data --vocab_size 30000
```
Note: It takes only the files in tmp_dir for vocab generation, and not the files in any folder inside it.

After the vocab is generated, we can train the model (just add our user defined arguments) :
```
--t2t-trainer \
  --data_dir=/home/user.name/data_dir \
  --tmp_dir=/home/user.name/tmp_dir \
  --problem=summarize_pubmed \
  --model=transformer \
  --hparams_set=transformer_few_sample \
  --output_dir=/home/user.name/train \
  --generate_data \
  --train_steps 10000 --eval_steps 300
  --t2t_usr_dir /home/user.name/summarize_pubmed
```
The ``--generate_data`` tag, generates the tf.Records files in data_dir and correspondingly outputs the 3 files *.source*, *.target*, *.extractive_summ* for each three category : train, test and dev categories which can be later used as a **reference** or for **calculating rouge scores**.

### Decoding
Training yields the model in *output_dir* with checkpoints and hyper-params used. 
We need to specify an empty file, where decoding can take place (here, *decode_pubmed.txt*). Decoding needs to be done using **t2t-decoder** , let's say with *beam_size* and *alpha* as 5 and .6 respectively.
```
t2t-decoder \
  --data_dir=/home/user.name/data \
  --problem=summarize_pubmed \
  --model=transformer \
  --hparams_set=transformer_few_sample \
  --output_dir=/home/user.name/train \
  --decode_hparams="beam_size=5,alpha=0.6" \
  --decode_from_file=summ_pubmed.test.extractive_summ \
  --decode_to_file=decode_pubmed.txt
```
#### Evaluating Results
For calculating ROUGE scores, we use *files2rouge* module. For installing files2rouge, please follow the instruction line-by-line as mentioned in the [repository](https://github.com/pltrdy/files2rouge). 
For rouge score calculation, we can just run
```
$ files2rouge summ_pubmed.test.target decode_pubmed.txt
```
### Results
After having the encoder and tokenizer integrated, I could only train on 200 samples. The model did overfit a lot. Here are the results,
```
ROUGE score calculated between gold summary and 

---- body ----
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

---- extractive-summarized-text -----
---------------------------------------------
1 ROUGE-1 Average_R: 0.24682 (95%-conf.int. 0.22895 - 0.26492)
1 ROUGE-1 Average_P: 0.69329 (95%-conf.int. 0.67466 - 0.71158)
1 ROUGE-1 Average_F: 0.34707 (95%-conf.int. 0.32895 - 0.36582)
---------------------------------------------
1 ROUGE-2 Average_R: 0.10620 (95%-conf.int. 0.09712 - 0.11568)
1 ROUGE-2 Average_P: 0.29895 (95%-conf.int. 0.28190 - 0.31632)
1 ROUGE-2 Average_F: 0.14970 (95%-conf.int. 0.13851 - 0.16119)
---------------------------------------------
1 ROUGE-L Average_R: 0.13671 (95%-conf.int. 0.12667 - 0.14709)
1 ROUGE-L Average_P: 0.39240 (95%-conf.int. 0.37705 - 0.40830)
1 ROUGE-L Average_F: 0.19322 (95%-conf.int. 0.18183 - 0.20507)
```
**Note** : As expected, give the body text is long, will have a very high accuracy but low recall.  While the extractve summarization will have relatively lower accuracy but higher recall.  Altogether the F1-score avg of ROUGE-L is better for extractive summarized part in comparison with the actual gold summary.

The summary generated using decoding was an overfit summary, with run just on 10000 steps. So, the ROUGE scores are not provided here.

##  Future Implementations

 - [ ] To include the **memory compressed attention** in our model like that
       implemented in wikisum problem of tensor2tensor, that can handle long sequences. 
       [Understanding-handling-long-sequences-wikisum](To%20include%20the%20memory%20compressed%20attention%20in%20our%20model%20like%20that%20%20%20%20%20%20%20%20implemented%20in%20wikisum%20problem,%20that%20can%20handle%20long%20sequences.).
 - [ ] Training on **larger training data** set for longer time. As seen for the summarization model, it needs larger training data and longer training time to work comendably well. With this, the repetitions are reduced (less overfitting) and the sequences generated are **more accurate**. Unfortunately I was not able to tackle the issue by training on ml-engine or on our GPU due to time constraint.
 - [ ] The biomedical vocabulary generation takes **a lot of time**. Typically, with normal internet speed, for a vocab generation of 2 lacs samples, it takes around 2.5 days. It can be made faster using multiple CPU cores or even better, if we could use GPU for the same.   

