#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 2: supervised model training

Usage:

    $ spark-submit supervised_train.py hdfs:/path/to/file.parquet hdfs:/path/to/save/model

spark-submit --conf spark.driver.memory=64g --conf spark.executor.memory=64g rs_train_f.py
    spark-submit --conf spark.driver.memory=64g rs_train_f.py
spark-submit --num-executors 64 --driver-memory 64g rs_preprocess.py
spark-submit --conf spark.driver.memory=50g --conf spark.executor.memory=50g --conf spark.executor.instances=64 rs_preprocess.py

'''

# We need sys to get the command line arguments
import sys
import time
import itertools
from pyspark import SparkConf
from pyspark.ml.recommendation import ALS
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, DoubleType, StructType
from pyspark.sql.functions import rank, col, collect_list, Column, first


def tmp_task(spark):
    now = time.time()
    lyrics = spark.read.parquet('hdfs:/user/yh2857/lyrics.parquet')
    lyrics = lyrics.orderBy('item_index', ascending=True)
    lyrics = lyrics.repartition('item_index')
    lyrics.write.mode('overwrite').parquet('lyrics_ordered.parquet')
    print("Writing lyrics parquet takes --- %s seconds ---" % (time.time() - now))
def main(spark):
    train_file = 'hdfs:/user/bm106/pub/project/cf_train.parquet'
    valid_file = 'hdfs:/user/bm106/pub/project/cf_validation.parquet'
    test_file = 'hdfs:/user/bm106/pub/project/cf_test.parquet'
    lyrics_file = 'hdfs:/user/bm106/pub/project/lyrics.parquet'
    features_file = 'hdfs:/user/bm106/pub/project/features.parquet'
    metadata_file = 'hdfs:/user/bm106/pub/project/metadata.parquet'
    tags_file = 'hdfs:/user/bm106/pub/project/tags.parquet'



    # train_file = 'hdfs:/user/sc6995/cf_train_index.parquet'
    # valid_file = 'hdfs:/user/sc6995/cf_valid_index.parquet'
    # test_file = 'hdfs:/user/sc6995/cf_test_index.parquet'

    now = time.time()
    cf_train = spark.read.parquet(train_file)
    indexer1 = StringIndexer(inputCol="user_id", outputCol="user_index", handleInvalid='skip')
    indexer2 = StringIndexer(inputCol="track_id", outputCol="item_index", handleInvalid='skip')
    pipeline = Pipeline(stages=[indexer1, indexer2])
    preprocessing = pipeline.fit(cf_train)
    cf_train = preprocessing.transform(cf_train)
    cf_train.write.mode('overwrite').parquet('cf_train_trans.parquet')
    print("Writing training parquet takes --- %s seconds ---" % (time.time() - now))

    now = time.time()
    cf_val = spark.read.parquet(valid_file)
    cf_val = preprocessing.transform(cf_val)
    cf_val.write.mode('overwrite').parquet('cf_val_trans.parquet')
    print("Writing validation parquet takes --- %s seconds ---" % (time.time() - now))

    now = time.time()
    cf_test = spark.read.parquet(test_file)
    cf_test = preprocessing.transform(cf_test)
    cf_test.write.mode('overwrite').parquet('cf_test_trans.parquet')
    print("Writing test parquet takes --- %s seconds ---" % (time.time() - now))

    now = time.time()
    lyrics = spark.read.parquet(lyrics_file)
    lyrics = preprocessing.transform(lyrics)
    lyrics = lyrics.orderBy('item_index', ascending=True)
    lyrics = lyrics.repartition('item_index')
    lyrics.write.mode('overwrite').parquet('lyrics.parquet')
    print("Writing lyrics parquet takes --- %s seconds ---" % (time.time() - now))

    now = time.time()
    features = spark.read.parquet(features_file)
    features = preprocessing.transform(features)
    features.write.mode('overwrite').parquet('features.parquet')
    print("Writing features parquet takes --- %s seconds ---" % (time.time() - now))

    now = time.time()
    metadata = spark.read.parquet(metadata_file)
    metadata = preprocessing.transform(metadata)
    metadata.write.mode('overwrite').parquet('metadata.parquet')
    print("Writing metadata parquet takes --- %s seconds ---" % (time.time() - now))

    now = time.time()
    tags = spark.read.parquet(tags_file)
    tags = preprocessing.transform(tags)
    tags.write.mode('overwrite').parquet('tags.parquet')
    print("Writing tags parquet takes --- %s seconds ---" % (time.time() - now))

# Only enter this block if we're in main
if __name__ == "__main__":
    conf = SparkConf()
    # conf.set("spark.driver.memory", '50G')
    # conf.set("spark.executor.memory", "50G")
    # conf.set("spark.executor.cores", "16")
    # conf.set('spark.executor.instances', '64')

    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set('spark.sql.broadcastTimeout', '36000')
    conf.set("spark.default.parallelism", "8")
    conf.set("spark.sql.shuffle.partitions", "8")
    conf.set("spark.io.compression.codec", "snappy")

    conf.set("spark.rdd.compress", "true")
    conf.set("spark.shuffle.service.enabled", "false")
    conf.set("spark.dynamicAllocation.enabled", "false")

    # Create the spark session object
    spark = SparkSession.builder.config(conf=conf).appName("preprocess").getOrCreate()
    main(spark)



