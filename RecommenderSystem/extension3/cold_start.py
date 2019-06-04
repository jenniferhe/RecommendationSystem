#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 2: Cold Start Implementation
Cold-start: using the supplementary data, build a model that can map observable feature data to the learned latent factor 
representation for items. To evaluate its accuracy, simulate a cold-start scenario by holding out a subset of items during 
training (of the recommender model), and compare its performance to a full collaborative filter model.
Usage:

    $ spark-submit --conf spark.driver.memory=16g --conf spark.executor.memory=16g --conf spark.executor.instances=8 cold_start.py tmp

'''

import sys,time,itertools,pickle,os
from pyspark import SparkConf
import pandas as pd
import numpy as np
from pyspark.ml.classification import RandomForestClassifier,RandomForestClassificationModel
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer,StringIndexer,VectorAssembler,CountVectorizer,StandardScaler
from pyspark.ml.recommendation import ALS,ALSModel
from pyspark.ml.linalg import Vectors,VectorUDT
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, DoubleType, StructType
from pyspark.sql.functions import rank, col, collect_list, Column, first,udf
import pyspark.sql.functions as F



def preprocess_ALS_data(spark):
    file_path = 'hdfs:/user/yh2857/model_frac_1/rank10_reg0.1_alpha0.01/itemFactors'
    lfs_df = spark.read.parquet(file_path)
    idx = lfs_df.rdd.map(lambda row: row[0])
    features = lfs_df.rdd.map(lambda row: row[1])
    lfs_df = idx.zip(features.map(lambda x: Vectors.dense(x))).toDF(schema=['id', 'features'])

    kmeans = KMeans(k=50, seed=1)
    model = kmeans.fit(lfs_df)
    transformed = model.transform(lfs_df)
    centers = model.clusterCenters()
    with open('kmean_centers.txt', 'wb') as f:
        pickle.dump(centers, f)
    return transformed,centers

def preprocess_files(spark):
    lyrics_file = 'hdfs:/user/yh2857/lyrics_processed.parquet'
    features_file = 'hdfs:/user/yh2857/features.parquet'
    metadata_file = 'hdfs:/user/yh2857/metadata.parquet'
    tags_file = 'hdfs:/user/yh2857/tags.parquet'

    tags = spark.read.parquet(tags_file)
    tags_agg = tags.groupby('item_index').agg(collect_list('tag').alias('tags'))
    cv = CountVectorizer(inputCol="tags", outputCol="keywords", vocabSize=5000, minDF=2.0)
    model = cv.fit(tags_agg)
    tags_agg = model.transform(tags_agg).withColumnRenamed('item_index', 'tags_item_index')
    tags_agg.limit(5).show()
    print("tags feature info ", tags_agg.count(), tags_agg)
    # tmp = tags.groupBy("tag").count().sort(desc("count"))
    # feature_words = tmp.where(col('count') >= 100).select('tag')
    # huge_df = tags.join(feature_words, feature_words.tag == tags.tag, "left_outer")

    metadata = spark.read.parquet(metadata_file)
    indexer = StringIndexer(inputCol="artist_id", outputCol="artist_idx")
    metadata = indexer.fit(metadata).transform(metadata).withColumnRenamed('item_index', 'meta_item_index')
    metadata.limit(5).show()
    print("metadata features info ", metadata.count(), metadata)

    features = spark.read.parquet(features_file)
    features.limit(5).show()
    print("features data info ", features.count(), features)

    lyrics = spark.read.parquet(lyrics_file).withColumnRenamed('item_index', 'lyrics_item_index')
    lyrics.limit(5).show()
    print("lyrics data info ", lyrics.count(), lyrics)

    df = tags_agg.join(metadata, tags_agg.tags_item_index == metadata.meta_item_index, 'inner')
    df = df.join(features, df.tags_item_index == features.item_index, 'inner')
    df = df.join(lyrics, df.tags_item_index == lyrics.lyrics_item_index, 'inner')
    df = df.select("item_index", 'keywords', 'artist_idx', 'year', 'artist_hotttnesss', 'artist_familiarity',
                   'duration','countvector',
                   'loudness_mean', 'loudness_std', 'timbre_00', 'timbre_01', 'timbre_02', 'timbre_03', 'timbre_04',
                   'timbre_05', 'timbre_06', 'timbre_07', 'timbre_08', 'timbre_09', 'timbre_10', 'timbre_11')

    filter_col = [col for col in features.columns if col.startswith('timbre')]
    filter_col += (['loudness_std', 'loudness_mean', 'duration', 'artist_hotttnesss', 'artist_familiarity', 'year'])

    df_assembler = VectorAssembler(inputCols=filter_col, outputCol="features_tmp").setHandleInvalid('skip')
    scaler = StandardScaler(inputCol="features_tmp", outputCol="features", withStd=True, withMean=False)
    tmp = df_assembler.transform(df)
    scalerModel = scaler.fit(tmp)
    fitted = scalerModel.transform(tmp)
    fitted = fitted.select("item_index", "features")
    fitted.limit(5).show()
    print(fitted.count(), fitted)

    filter_col += (['countvector', 'keywords'])
    df_assembler = VectorAssembler(inputCols=filter_col, outputCol="features_tmp").setHandleInvalid('skip')
    scaler = StandardScaler(inputCol="features_tmp", outputCol="features", withStd=True, withMean=False)
    tmp = df_assembler.transform(df)
    scalerModel = scaler.fit(tmp)
    fitted2 = scalerModel.transform(tmp)
    fitted2 = fitted2.select("item_index", "features")
    fitted2.limit(5).show()
    print(fitted2.count(), fitted2)

    return fitted,fitted2

def eval_logreg(new_df,filename):
    (train, test) = new_df.randomSplit([0.8, 0.2], 24)
    train = train.withColumnRenamed('prediction', 'label')
    test = test.withColumnRenamed('prediction', 'label')
    df = MLUtils.convertVectorColumnsFromML(train, "features")
    parsedData = df.select(col("label"), col("features")).rdd.map(lambda row: LabeledPoint(row.label, row.features))
    model = LogisticRegressionWithLBFGS.train(parsedData, numClasses=50)
    model.save(spark.sparkContext, filename)
    # sameModel = LogisticRegressionModel.load(spark.sparkContext, "LogRegLBFGSModel")
    labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
    trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(parsedData.count())
    print("LogReg Small Training Error = " + str(trainErr))
    df = MLUtils.convertVectorColumnsFromML(test, "features")
    parsed_test = df.select(col("label"), col("features")).rdd.map(lambda row: LabeledPoint(row.label, row.features))
    testErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(parsed_test.count())
    print("LogReg Small Test Error = " + str(testErr))

def eval_rf(train,test,name):

    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)
    model2 = rf.fit(train)
    model2.write().overwrite().save(name)
    predictions = model2.transform(train)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Train Error = %g" % (1.0 - accuracy))

    predictions = model2.transform(test)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

def tune_rf_param(train,test,name):
    now = time.time()
    best_score = 0
    best_model = None

    depths = [5,10,20]
    numtrees = [2,5,10,20,50]
    for i, j in itertools.product(depths, numtrees):
        print('Training for depth = {}, num_tree = {}'.format(i,j))
        rf = RandomForestClassifier(labelCol="label", featuresCol="features", maxDepth = i, numTrees=j)
        model2 = rf.fit(train)

        predictions = model2.transform(train)
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        print("Train Error = %g" % (1.0 - accuracy))

        predictions = model2.transform(test)
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        print("Test Error = %g" % (1.0 - accuracy))

        if accuracy > best_score:
            best_model = model2
            best_score = accuracy

    print('Overall time used ={}'.format(time.time() - now))
    best_model.write().overwrite().save(name)


def main(spark,output_name):
    sys.stdout = open(output_name, "w")

    # STEP 1: Buiding Dataset

    fitted, fitted2 = preprocess_files(spark) # Fitted2 contains word vectors
    transformed, centers = preprocess_ALS_data(spark)

    new_df = fitted.join(transformed.select('id', 'prediction'), fitted.item_index == transformed.id, 'inner').drop('id')
    new_df.write.mode('overwrite').parquet('coldstart_processed_short.parquet')
    new_df2 = fitted2.join(transformed.select('id', 'prediction'), fitted2.item_index == transformed.id, 'inner').drop('id')
    new_df2.write.mode('overwrite').parquet('coldstart_processed_full.parquet')

    df_path = 'hdfs:/user/yh2857/coldstart_processed_short.parquet'
    df2_path = 'hdfs:/user/yh2857/coldstart_processed_full.parquet'
    new_df = spark.read.parquet(df_path).withColumnRenamed('prediction', 'label')
    new_df2 = spark.read.parquet(df2_path).withColumnRenamed('prediction', 'label')

    (train, test) = new_df.randomSplit([0.8, 0.2], 24)
    (train2, test2) = new_df2.randomSplit([0.8, 0.2], 24)
    train = train.cache()
    test = test.cache()
    train2 = train2.cache()
    test2 = test2.cache()

    #STEP 2: Building Classifier

    eval_logreg(train, test,os.path.join(output_name,'short_logreg.model'))
    eval_logreg(train2, test2,os.path.join(output_name,'full_logreg.model'))
    eval_rf(train, test,os.path.join(output_name,'short_rf.model'))
    eval_rf(train2, test2,os.path.join(output_name,'full_rf.model'))

    tune_rf_param(train,test,os.path.join(output_name,'short_rf.model'))
    tune_rf_param(train2,test2,os.path.join(output_name,'full_rf.model'))

    sys.stdout.close()

def main2(spark,output):
    # STEP 3: Use Classifier to predict Latent Factor Vector and updated ALS Model

    model = RandomForestClassificationModel.load('hdfs:/user/yh2857/short_rf.model')
    lfsdf = spark.read.parquet('hdfs:/user/yh2857/model_frac_1/rank10_reg0.1_alpha0.01/itemFactors')
    idx = lfsdf.rdd.map(lambda row: row[0])
    features = lfsdf.rdd.map(lambda row: row[1])
    lfsdf = idx.zip(features.map(lambda x: Vectors.dense(x))).toDF(schema=['id', 'features'])

    # print(lfsdf.count(),lfsdf.select('id').distinct().count(),lfsdf)

    with open('kmean_centers.txt', 'rb') as f:
        centers = pickle.load(f)
    new_centers = []
    for i,c in enumerate(centers):
        new_centers.append([i, Vectors.dense(centers[0].tolist())])
    centerdf = spark.createDataFrame(
        pd.DataFrame(data=new_centers)).withColumnRenamed('0', 'center_idx').withColumnRenamed('1', 'center_features')
    # print(centerdf.count(), centerdf)

    df_path = 'hdfs:/user/yh2857/coldstart_processed_short.parquet'
    new_df = spark.read.parquet(df_path).withColumnRenamed('prediction', 'label')

    train, test= new_df.randomSplit([0.8, 0.2], 24)
    print(test.count(), test.select('item_index').distinct().count(),test)

    predictions = model.transform(test)
    # predictions.select("prediction").distinct().show()
    predicted = predictions.join(centerdf, predictions.prediction == centerdf.center_idx, 'left')
    # predicted.show()

    original_lfs = lfsdf.join(predicted, lfsdf.id == predicted.item_index, "leftanti")
    predicted = predicted.select('item_index', 'center_features').withColumnRenamed("center_features", 'features')

    print(original_lfs)
    print(predicted)
    updated_lfs = original_lfs.withColumnRenamed('id', 'item_index').union(predicted)
    # updated_lfs.show()
    output_file = 'hdfs:/user/yh2857/rank10_reg0.1_alpha0.1/itemFactors'
    updated_lfs.write.mode('overwrite').parquet(output_file)

# Only enter this block if we're in main
if __name__ == "__main__":
    output_name = sys.argv[1]
    conf = SparkConf()
    # conf.set("spark.driver.memory", '50G')
    # conf.set("spark.executor.memory", "50G")
    # conf.set("spark.executor.cores", "4")
    # conf.set('spark.executor.instances', '64')

    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set('spark.sql.broadcastTimeout', '36000')
    conf.set("spark.default.parallelism", "64")
    conf.set("spark.sql.shuffle.partitions", "64")
    conf.set("spark.io.compression.codec", "snappy")

    conf.set("spark.rdd.compress", "true")
    conf.set("spark.shuffle.service.enabled", "false")
    conf.set("spark.dynamicAllocation.enabled", "false")

    # Create the spark session object
    spark = SparkSession.builder.config(conf=conf).appName(output_name).getOrCreate()
    # main(spark, output_name)
    main2(spark,output_name)



