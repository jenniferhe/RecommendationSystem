#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Training the model

Usage:

    $ spark-submit train.py hdfs:/path/to/file.parquet hdfs:/path/to/save/

'''

import sys
from pyspark.sql import SparkSession
import time
from pyspark.ml.recommendation import ALS
import itertools
from pyspark import SparkConf

def main(spark, data_file, model_path):
    '''

    Parameters
    ----------
    spark : SparkSession object

    data_file : string, path to the training parquet file to load

    model_file : string, path to store the serialized model file
    '''
    start_time = time.time()
    train_frac = 0.02
    cf_train = spark.read.parquet(data_file).sample(False, train_frac, 1)
    print('training size : ' + str(cf_train.count()))
    ranks = [5,10,20,50,80]
    reg_params = [0.01, 0.1, 1]
    alpha = [0.1, 1, 5]
    print("Finished loading data --- %s seconds ---" % (time.time() - start_time))
    cf_train.cache()
    for i,j,k in itertools.product(ranks,reg_params,alpha):
        print('Training for rank {}, reg {}, alpha {}'.format(i,j,k))
        start_train = time.time()
        als = ALS(userCol="user_index", itemCol="item_index", ratingCol="count", implicitPrefs=True, rank=i,
                  regParam=j, alpha=k)
        model = als.fit(cf_train)
        model.write().overwrite().save(str(model_path)+'/model_reg_{}_rank{}_a_{}_{}'.format(j, i, k, train_frac))
        print('Training takes {}'.format(time.time() - start_train))



# Only enter this block if we're in main
if __name__ == "__main__":
    conf = SparkConf()
    conf.set("spark.executor.memory", "64G")
    conf.set("spark.driver.memory", '64G')
    conf.set("spark.executor.cores", "4")
    conf.set('spark.executor.instances', '10')
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.shuffle.service.enabled", "false")
    conf.set("spark.dynamicAllocation.enabled", "false")
    conf.set('spark.sql.broadcastTimeout', '36000')
    conf.set("spark.default.parallelism", "40")
    conf.set("spark.sql.shuffle.partitions", "40")
    conf.set("spark.io.compression.codec", "snappy")
    conf.set("spark.rdd.compress", "true")

    # Create the spark session object
    spark = SparkSession.builder.appName('train').getOrCreate()

    # Get the filename from the command line
    data_file = sys.argv[1]
    model_path = sys.argv[2]
    # Call our main routine
    main(spark, data_file, model_path)



