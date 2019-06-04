#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Usage:

    $ spark-submit --conf spark.driver.memory=64g --conf spark.executor.memory=64g evaluate.py data_path model_path
'''

# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

# TODO: you may need to add imports here
import time

from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.linalg import Vectors

from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml import Pipeline
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, DoubleType, StructType
from pyspark.sql.functions import rank, col, collect_list, Column, first
import itertools
from pyspark.ml.feature import StringIndexer
from pyspark import SparkConf
import gc


def main(spark, data_file, model_path):
    cf_valid = spark.read.parquet(data_file)
    cf_valid = cf_valid.withColumn('item_index', cf_valid['item_index'].cast('int'))
    # cf_train = spark.read.parquet(train_file)
    # print('Number of train data',cf_train.count())
    print('Number of validation data:', cf_valid.count())
    # user2pred = cf_valid.join(cf_train,'user_index','inner')
    # user2pred.show()
    true_label = cf_valid.groupby('user_index').agg(collect_list('item_index').alias('label'))
    '''

    '''
    start_time = time.time()
    train_frac = 0.02
    # Build y label and Logistics
    # Build pipeline and CV, fit data, and save

    ranks = [5, 10, 20, 30, 50]
    reg_params = [0.01, 0.1, 1]
    alpha = [0.1, 1, 5]

    max_score = -1
    # best_model = None
    best_rank = -1
    best_reg_param = -1
    best_alpha = -1
    print("Finished loading data --- %s seconds ---" % (time.time() - start_time))

    for i, j, k in itertools.product(ranks, reg_params, alpha):
        print('Training for rank {}, reg {}, alpha {}'.format(i, j, k))
        start_train = time.time()
        # model_file = 'model_index/model_reg_{}_rank{}_a_{}_{}'.format(j, i, k, train_frac)
        model_file = str(model_path)+'/model_reg_{}_rank{}_a_{}_{}'.format(j,i,k,train_frac)
        model = ALSModel.load(model_file)
        user_subset_recs = model.recommendForUserSubset(cf_valid, 500)
        user_subset_recs = user_subset_recs.select('user_index', 'recommendations.item_index')
        rankings = user_subset_recs.join(true_label, 'user_index', 'inner')
        # rankings.show()
        print('Loading model and predict {}'.format(time.time() - start_train))
        start_metrics = time.time()
        predictAndLabels = rankings.rdd.map(lambda p: (p[1], p[2]))
        metrics = RankingMetrics(predictAndLabels)
        score = metrics.meanAveragePrecision
        print('Metrics start takes {}'.format(time.time() - start_metrics))
        print('Metrics MAP:', score)
        print("Finished at --- %s seconds ---" % (time.time() - start_time))
        del metrics, model
        gc.collect()
        if score > max_score:
            max_score = score
            # best_model = model
            best_rank = i
            best_reg_param = j
            best_alpha = k
    print("Best params: rank={}, reg_param={}, alpha={}, Score: {}".format(best_rank, best_reg_param, best_alpha,
                                                                           max_score))
    print("Finished cross validation --- %s seconds ---" % (time.time() - start_time))


# Only enter this block if we're in main
if __name__ == "__main__":
    conf = SparkConf()
    conf.set("spark.executor.memory", "8G")
    conf.set("spark.driver.memory", '16G')
    conf.set("spark.executor.cores", "4")
    conf.set('spark.executor.instances', '40')
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.shuffle.service.enabled", "false")
    conf.set("spark.dynamicAllocation.enabled", "false")
    conf.set('spark.sql.broadcastTimeout', '36000')
    conf.set("spark.default.parallelism", "400")
    conf.set("spark.sql.shuffle.partitions", "400")
    conf.set("spark.io.compression.codec", "snappy")
    conf.set("spark.rdd.compress", "true")
    # Create the spark session object
    spark = SparkSession.builder.appName('evaluate').getOrCreate()
    data_file = sys.argv[1]
    model_file = sys.argv[2]
    main(spark, data_file, model_file)



