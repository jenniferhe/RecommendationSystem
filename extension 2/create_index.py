import annoy
import pandas as pd
import numpy as np
from annoy import AnnoyIndex
import random
import time
import nmslib

annoy_metrics = 'angular'
annoy_metrics = 'euclidean'

def build_tree(num_users, ranks, df, num_trees):
    f = ranks
    t = AnnoyIndex(f,metric = annoy_metrics)  # Length of item vector that will be indexed
    for i in range(num_users):
        v = df[i].tolist()
        t.add_item(i, v)

    t.build(num_trees)  # 10 trees
    return t


def brute_force(array, k, top_k=100):
    '''
    compute distance
    '''
    distance = np.sum((array - array[k]) ** 2, axis=1)

    return np.argsort(distance)[:top_k]


def get_recall(tree_res, brute_res):
    if type(tree_res) != list:
        tree_res = tree_res.tolist()
    brute_res = set(brute_res.tolist())
    tree_res = set(tree_res)
    total = len(brute_res)
    relevant = len(brute_res.intersection(tree_res))
    return relevant / total


def search_tree_and_time(df, t, num_query=100, top_k=100):
    num_samples, ranks = df.shape
    index2query = np.random.choice(range(num_samples), num_query)
    time_st = 0
    time_bf = 0
    recalls = []
    for idx in index2query:
        start_st = time.time()
        tree_res = t.get_nns_by_item(idx, top_k * 5)
        time_st += time.time() - start_st

        start_bf = time.time()
        brute_res = brute_force(df, idx, top_k)
        time_bf += time.time() - start_bf

        recall = get_recall(tree_res, brute_res)
        recalls.append(recall)

    return time_st, time_bf, recalls


def search_tree_and_time2(df, t, num_query=100, top_k=100, max_multiplier=10):
    num_samples, ranks = df.shape
    index2query = np.random.choice(range(num_samples), num_query)
    time_st = 0
    time_bf = 0
    recalls = []
    time_array_st = []
    time_array_bf = []
    for multiplier in range(1, max_multiplier):
        recall_ = []
        for idx in index2query:
            start_st = time.time()
            tree_res = t.get_nns_by_item(idx, top_k * multiplier)
            time_st += time.time() - start_st

            start_bf = time.time()
            brute_res = brute_force(df, idx, top_k)
            time_bf += time.time() - start_bf

            recall = get_recall(tree_res, brute_res)
            recall_.append(recall)
        recalls.append(np.mean(recall_))
        time_array_st.append(num_query / time_st)
        time_array_bf.append(num_query / time_bf)

    return time_array_st, time_array_bf, recalls

if __name__ == '__main__':
    # load data
    print('Loading data ...')
    df = pd.read_csv('user_factor.csv',header = None)
    df = df.values[:,1:]
    num_users, ranks = df.shape
    t = build_tree(num_users, ranks, df, 10)
    t.save('tree_10')
    t = build_tree(num_users, ranks, df, 50)
    t.save('tree_50')
    # hnsw initialize
    space = 'l2'
    index = nmslib.init(method='hnsw', space=space)
    index.addDataPointBatch(df)
    M = 40
    efC = 100
    num_threads = 4
    index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post': 0}
    index.createIndex(index_time_params, print_progress=True)
    index.saveIndex('hnsw_index40.bin')

    index = nmslib.init(method='hnsw', space=space)
    index.addDataPointBatch(df)
    M = 80
    efC = 100
    num_threads = 4
    index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post': 0}
    index.createIndex(index_time_params, print_progress=True)
    index.saveIndex('hnsw_index80.bin')


    # Set index parameters
    # These are the most important onese
    # SW graph
    NN = 40
    efC = 100

    num_threads = 4
    index_time_params = {'NN': NN, 'indexThreadQty': num_threads, 'efConstruction': efC}
    # space_name='l2'
    print('Initialized ...')
    index2 = nmslib.init(method='sw-graph', space=space, data_type=nmslib.DataType.DENSE_VECTOR)
    index2.addDataPointBatch(df)
    # Create an index
    start = time.time()
    index2.createIndex(index_time_params, print_progress=True)
    end = time.time()
    print('Index-time parameters', index_time_params)
    print('Indexing time = %f' % (end - start))
    index2.saveIndex('sw_graph_index40.bin')

    NN = 100
    efC = 100

    num_threads = 4
    index_time_params = {'NN': NN, 'indexThreadQty': num_threads, 'efConstruction': efC}
    # space_name='l2'
    print('Initialized ...')
    index2 = nmslib.init(method='sw-graph', space=space, data_type=nmslib.DataType.DENSE_VECTOR)
    index2.addDataPointBatch(df)
    # Create an index
    start = time.time()
    index2.createIndex(index_time_params, print_progress=True)
    end = time.time()
    print('Index-time parameters', index_time_params)
    print('Indexing time = %f' % (end - start))
    index2.saveIndex('sw_graph_index100.bin')