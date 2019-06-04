#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

1)Index the user and item identifiers
2)Index user and item, transform the count data into log(1+count)
3)Index user and item, drop low count value
4)save the data from three transformations separately

Usage:

    $ spark-submit preprocess.py
'''


from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.feature import StringIndexer
from pyspark import SparkConf

def main(spark):
    train_file = 'hdfs:/user/bm106/pub/project/cf_train.parquet'
    cf_train = spark.read.parquet(train_file)
    valid_file = 'hdfs:/user/bm106/pub/project/cf_validation.parquet'
    cf_valid = spark.read.parquet(valid_file)
    test_file = 'hdfs:/user/bm106/pub/project/cf_test.parquet'
    cf_test = spark.read.parquet(test_file)
    indexer1 = StringIndexer(inputCol="user_id", outputCol="user_index", handleInvalid='skip')
    indexer2 = StringIndexer(inputCol="track_id", outputCol="item_index", handleInvalid='skip')
    logTrans = SQLTransformer(statement='select user_index,item_index,log10(count+1) as count from __THIS__')
    dropTrans = SQLTransformer(statement='select * from __THIS__ where count>1')
    pipeline1 = Pipeline(stages=[indexer1, indexer2])
    pipeline2 = Pipeline(stages=[indexer1, indexer2, logTrans])
    pipeline3 = Pipeline(stages=[indexer1, indexer2, dropTrans])
    pre1 = pipeline1.fit(cf_train)
    pre2 = pipeline2.fit(cf_train)
    pre3 = pipeline3.fit(cf_train)

    pre1.transform(cf_train).write.parquet('hdfs:/user/sc6995/cf_train_index.parquet')
    pre1.transform(cf_valid).write.parquet('hdfs:/user/sc6995/cf_valid_index.parquet')
    pre1.transform(cf_test).write.parquet('hdfs:/user/sc6995/cf_test_index.parquet')

    pre2.transform(cf_train).write.parquet('hdfs:/user/sc6995/cf_train_logtrans.parquet')
    pre2.transform(cf_valid).write.parquet('hdfs:/user/sc6995/cf_valid_logtrans.parquet')
    pre2.transform(cf_test).write.parquet('hdfs:/user/sc6995/cf_test_logtrans.parquet')

    pre3.transform(cf_train).write.parquet('hdfs:/user/sc6995/cf_train_droptrans1.parquet')
    #pre3.transform(cf_valid).write.parquet('hdfs:/user/sc6995/cf_valid_droptrans1.parquet')
    #pre3.transform(cf_test).write.parquet('hdfs:/user/sc6995/cf_test_droptrans1.parquet')


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
    spark = SparkSession.builder.appName('preprocess').getOrCreate()
    main(spark)



