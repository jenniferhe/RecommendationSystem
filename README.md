# Recommeder System
## Overview

In this project, we built a recommendation model to solve a realistic, large-scale applied problem using Spark's alternating least squares (ALS) method and tuned the hyper-parameters to optimize performance on the validation set together. On top of the baseline collaborative filter model, we implemented three extensions:
alternative model formulations, fast search, and cold start strategy*.*

## The data set

The following dataset was provided on Dumbo's HDFS(NYU) `hdfs:/user/bm106/pub/project`:

  - `cf_train.parquet`
  - `cf_validation.parquet`
  - `cf_test.parquet`
  
  - `metadata.parquet`
  - `features.parquet`
  - `tags.parquet`
  - `lyrics.parquet`


The first three files contain training, validation, and testing data for the collaborative filter.  Specifically, each file contains a table of triples `(user_id, count, track_id)` which measure implicit feedback derived from listening behavior.  The first file `cf_train` contains full histories for approximately 1M users, and partial histories for 110,000 users, located at the end of the table.

`cf_validation` contains the remainder of histories for 10K users, and should be used as validation data to tune your model.

`cf_test` contains the remaining history for 100K users, which should be used for your final evaluation.

The four additional files consist of supplementary data for each track (item) in the dataset. 

## Basic recommender system 

#### Requirements

You should use Spark's alternating least squares (ALS) method to learn latent factor representations for users and items.  This model has some hyper-parameters that you should tune to optimize performance on the validation set, notably: 

  - the *rank* (dimension) of the latent factors,
  - the *regularization* parameter, and
  - *alpha*, the scaling parameter for handling implicit feedback (count) data.

The choice of evaluation criteria for hyper-parameter tuning is entirely up to you, as is the range of hyper-parameters you consider, but be sure to document your choices in the final report.

Once your model is trained, evaluate it on the test set using the ranking metrics provided by spark.  Evaluations should be based on predictions of the top 500 items for each user.

#### Hints

You may need to transform the user and item identifiers (strings) into numerical index representations for it to work properly with Spark's ALS model.  You might save some time by doing this once and saving the results to new files in HDFS.

Start small, and get the entire system working start-to-finish before investing time in hyper-parameter tuning!

You may consider downsampling the data to more rapidly prototype your model.  If you do this, be careful that your downsampled data includes enough users from the validation set to test your model.

## Extensions 

The choice of extension is up to you, but here are some ideas:

  - *Alternative model formualtions*: the `AlternatingLeastSquares` model in Spark implements a particular form of implicit-feedback modeling, but you could change its behavior by modifying the count data.  Conduct a thorough evaluation of different modification strategies (e.g., log compression, or dropping low count values) and their impact on overall accuracy.
  - *Fast search*: use a spatial data structure (e.g., LSH or partition trees) to implement accelerated search at query time.  For this, it is best to use an existing library such as `annoy` or `nmslib`, and you will need to export the model parameters from Spark to work in your chosen environment.  For full credit, you should provide a thorough evaluation of the efficiency gains provided by your spatial data structure over a brute-force search method.
  - *Cold-start*: using the supplementary data, build a model that can map observable feature data to the learned latent factor representation for items.  To evaluate its accuracy, simulate a cold-start scenario by holding out a subset of items during training (of the recommender model), and compare its performance to a full collaborative filter model.
