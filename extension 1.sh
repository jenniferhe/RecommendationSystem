#!/bin/bash
spark-submit train.py hdfs:/user/sc6995/cf_train_logtrans.parquet hdfs:/user/sc6995/model_log
spark-submit evaluate.py hdfs:/user/sc6995/cf_valid_logtrans.parquet hdfs:/user/sc6995/model_log

spark-submit train.py hdfs:/user/sc6995/cf_train_droptrans1.parquet hdfs:/user/sc6995/model_drop1
spark-submit evaluate.py hdfs:/user/sc6995/cf_valid_index.parquet hdfs:/user/sc6995/model_drop1
