#!/bin/bash
spark-submit preprocess.py
spark-submit train.py hdfs:/user/sc6995/cf_train_index.parquet hdfs:/user/sc6995/model_index
spark-submit evaluate.py hdfs:/user/sc6995/cf_valid_index.parquet hdfs:/user/sc6995/model_index
