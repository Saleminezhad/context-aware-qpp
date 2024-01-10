# <span style="color:blue"> weight difficulty </span>

In this work, we explore the sensitivity of retrieval methods to the choice and order of terms in a query, emphasizing its significant impact on retrieval effectiveness. We introduce a novel approach to predict query performance by learning difficulty weights for each query term within its specific context. These weights serve as indicators of a term's potential to enhance or diminish query effectiveness. Our method involves fine-tuning a language model to learn these difficulty weights, followed by their integration into a cross-encoder architecture for performance prediction. The proposed approach consistently demonstrates robust performance on the MSMARCO collection and the widely used Trec Deep Learning tracks query sets. Explore the repository for detailed implementation, experiments, and results.

<hr>

## 1) Term Weight Estimation

### 1.1 Trainig data (Query variation to term weight)

To create the training dataset for our model, we require pairs of queries (one easy and one hard) to assign labels to terms based on their retrieval effectiveness. The definition of easy and hard queries is determined by their performance metrics, such as MAP (Mean Average Precision). The necessary file can be located [here](https://drive.google.com/file/d/1GNbMB6vqet7xjpZ8P3mQUmNNAlD43g18/view?usp=sharing), download and put it under `/DeepCT2/data`  directory the with the following format:

` qid \t initial query \t Map (initial query) \t Target Query \t MAP (Target Query) `

To compute term weight labels, employ the ` term_label_generator.py ` script. Upon execution, each term in the query will be assigned a label, with a default of zero if not applicable. The output follows this format:

` {"query": "is sinusitis contagious", "term_recall": {"sinusitis": 1}, "doc": {"position": "1", "id": "43", "title": "is sinusitis contagious"}} `

The output files serves as the training data for the initial phase, where we estimate term difficulty weights.

<hr>

### 1.2 Train term weight estimation

environment setup:

* python 3.9+
* tensorflow 2.13.0
* nvidia-cudnn-cu11         8.6.0.163
  
trining code: we adapted [DeepCT](https://github.com/AdeDZY/DeepCT) methodology, However we transfer code into tensorflow 2 so it can be used with updated gpus. modified code can be found in the ` DeepCT2 ` folder.

for training run this code.
```
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH

export BERT_BASE_DIR=DeepCT2/uncased_L-12_H-768_A-12

export TRAIN_DATA_FILE=DeepCT2/data/data_label_neg_015.json
export OUTPUT_DIR=DeepCT2/outputs/marco_neg_max9_e12

python run_deepct.py \
  --task_name=marcodoc \
  --do_train=true \
  --do_eval=false \
  --do_predict=false \
  --data_dir=$TRAIN_DATA_FILE \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=9 \
  --train_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=12.0 \
  --recall_field=title \
  --output_dir=$OUTPUT_DIR 

export TRAIN_DATA_FILE=DeepCT2/data/data_label_pos_015.json
export OUTPUT_DIR=DeepCT2/outputs/marco_pos_max9_e12

python run_deepct.py \
  --task_name=marcodoc \
  --do_train=true \
  --do_eval=false \
  --do_predict=false \
  --data_dir=$TRAIN_DATA_FILE \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=9 \
  --train_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=12.0 \
  --recall_field=title \
  --output_dir=$OUTPUT_DIR 
```

BERT_BASE_DIR: The small, uncased BERT model released by Google.

TRAIN_DATA_FILE: a json file with the query term weight labels.

OUTPUT_DIR: output folder for training. It will store the tokenized training file (train.tf_record) and the checkpoints (model.ckpt).

If you don't want to train the model, download [trained model](https://drive.google.com/drive/folders/1Im6uCxkq5SaU0J2h_dyT-7MCuUPp1-iY?usp=sharing) and put it under ` DeepCT2/outputs `.

### 1.3 Inference term weight estimation


(You can skip this step. Alternatively, direct download our  estimate weights for the MS MARCO passage ranking corpus, from [here](https://drive.google.com/drive/folders/1R-Flu4dnP3aYXMLfqWYW6DnxKSJwm0Rv?usp=sharing) and copy them in the ` DeepCT2/predictions `)

```
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH

export BERT_BASE_DIR=/DeepCT2/uncased_L-12_H-768_A-12
export TEST_DATA_FILE=/DeepCT2/queries.train.small.tsv

export INIT_CKPT=/DeepCT2/outputs/marco_neg_max9_e12/model.ckpt-83689 
export OUTPUT_DIR=/DeepCT2/predictions/BertQPP_Query_neg_max9_e12

python run_deepct.py \
 --task_name=marcotsvdoc \
 --do_train=false \
 --do_eval=false \
 --do_predict=true \
 --data_dir=$TEST_DATA_FILE \
 --vocab_file=$BERT_BASE_DIR/vocab.txt \
 --bert_config_file=$BERT_BASE_DIR/bert_config.json \
 --init_checkpoint=$INIT_CKPT \
 --max_seq_length=15 \
 --train_batch_size=16 \
 --learning_rate=2e-5 \
 --num_train_epochs=12.0 \
 --output_dir=$OUTPUT_DIR

export INIT_CKPT=/DeepCT2/outputs/marco_pos_max9_e12/model.ckpt-83689 
export OUTPUT_DIR=/DeepCT2/predictions/BertQPP_Query_pos_max9_e12

python run_deepct.py \
 --task_name=marcotsvdoc \
 --do_train=false \
 --do_eval=false \
 --do_predict=true \
 --data_dir=$TEST_DATA_FILE \
 --vocab_file=$BERT_BASE_DIR/vocab.txt \
 --bert_config_file=$BERT_BASE_DIR/bert_config.json \
 --init_checkpoint=$INIT_CKPT \
 --max_seq_length=15 \
 --train_batch_size=16 \
 --learning_rate=2e-5 \
 --num_train_epochs=12.0 \
 --output_dir=$OUTPUT_DIR

```

TEST_DATA_FILE: a tsv file ` (docid \t doc_content) ` of the entire collection that you want ot compute weight. Here, we use the [MS MARCO queries](https://drive.google.com/file/d/1kiwbqlwQDSzO2BZFpcNs5Bsa1RgbAoPo/view?usp=sharing) to compute thier term weights, therefore, these term weights will be considered as the input for the query performance prediction part.

$OUTPUT_DIR: output folder for testing. computed term weights will be stored here.




## 2) Query Performance Prediction

After computing the term difficulty weights for every queries in our collection we need to expand queries and generate a term frequency version of them. alongside the modifed query we include the performance of the query and feed it into our model. the output file follows this format:

```
train_dic[qid]["qtext"]=query_soft_terms
train_dic[qid]["doc_text"]=query_hard_terms
train_dic[qid]["performance"]=query_performance_value
```
for creating training data you should download [queries.train.small.tsv](https://drive.google.com/file/d/1kiwbqlwQDSzO2BZFpcNs5Bsa1RgbAoPo/view?usp=sharing) and put it in `/DeepCT2/data`.

by running the ` train_weighted_term_frequemcy.py `, training pickle data will be created in ` BERTQPP/pklfiles/ `  directory.


### 2.1 train  query performance prediction
we adopt the methodology used in [BertQpp](https://github.com/Narabzad/BERTQPP). For more inofrmation about how to use this model refer to the github repository.


For those who prefer not to train the model, we provide the option to download our trained BERT-QPP model, available on [here](https://drive.google.com/file/d/1iU9W9DbKoMmbpuYRQJScFeEL1GQpf0AM/view?usp=sharing). This model is based on bert-based-uncased.

To train the model with your specific metric, use the `train_CE.py` script. This script facilitates learning the map@20 of BM25 retrieval on the MSMARCO training set. In our experiments, we utilized bert-base-uncased, and the trained model will be saved in the `BERTQPP/models/` directory.

### 2.2 Inference  query performance prediction

For testing, insert the trained model you wish to evaluate into the `test_CE.py` script and execute it. The results will be stored in the `results` directory, following the format: `QID\tPredicted_QPP_value`.

To evaluate the results, you can calculate the correlation between the actual performance of each query and the predicted QPP value.



