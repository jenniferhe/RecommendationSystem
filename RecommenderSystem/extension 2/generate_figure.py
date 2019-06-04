import annoy
import pandas as pd
import numpy as np
from annoy import AnnoyIndex
import random

import matplotlib.pyplot as plt
import time
import nmslib


def brute_force(array, k, top_k=100):
    '''
    compute distance
    '''
    distance = np.sum((array - array[k]) ** 2, axis=1)

    return np.argsort(distance)[:top_k]

def search_tree_and_time2(df, t, index, index2, num_query=100, top_k=100, max_multiplier=10):
    num_samples, ranks = df.shape
    index2query = np.random.choice(range(num_samples), num_query)
    time_st = 0
    time_bf = 0
    recall_array_hnsw = []
    recall_array_st = []
    recall_array_sw = []
    time_array_st = []
    time_array_bf = []
    time_array_hnsw = []
    time_array_sw = []
    for multiplier in [1,5]:
        print('Iteration {}'.format(multiplier))
        recall_st_ = []
        recall_hnsw_ = []
        recall_sw_ = []

        for idx in index2query:
            start_st = time.time()
            tree_res = t.get_nns_by_item(idx, top_k * multiplier)
            time_st += time.time() - start_st

            start_hnsw = time.time()
            hnsw_ids, _ = index.knnQuery(df[idx], k=top_k * multiplier)
            time_hnsw = time.time() - start_hnsw

            start_sw = time.time()
            sw_ids, _ = index2.knnQuery(df[idx], k=top_k * multiplier)
            time_sw = time.time() - start_sw

            start_bf = time.time()
            brute_res = brute_force(df, idx, top_k * multiplier)
            time_bf += time.time() - start_bf

            recall_st = get_recall(tree_res, brute_res)
            recall_hnsw = get_recall(hnsw_ids, brute_res)
            recall_sw = get_recall(sw_ids, brute_res)
            recall_hnsw_.append(recall_hnsw)
            recall_st_.append(recall_st)
            recall_sw_.append(recall_sw)

        recall_array_st.append(np.mean(recall_st_))
        recall_array_hnsw.append(np.mean(recall_hnsw_))
        recall_array_sw.append(np.mean(recall_sw_))
        time_array_st.append(num_query / time_st)
        time_array_bf.append(num_query / time_bf)
        time_array_hnsw.append(num_query / time_hnsw)
        time_array_sw.append(num_query / time_sw)

    return time_array_st, time_array_bf, time_array_hnsw, time_array_sw, recall_array_st, recall_array_hnsw, recall_array_sw

def get_recall(tree_res, brute_res):
    if type(tree_res) != list:
        tree_res = tree_res.tolist()
    brute_res = set(brute_res.tolist())
    tree_res = set(tree_res)
    total = len(brute_res)
    relevant = len(brute_res.intersection(tree_res))
    return relevant / total

if __name__ =='__main__':
    # load data
    annoy_metrics = 'angular'
    annoy_metrics = 'euclidean'
    start_scratch = True
    if start_scratch:
        df = pd.read_csv('user_factor.csv', header=None)
        df = df.values[:, 1:]
        num_users, ranks = df.shape
        t = AnnoyIndex(ranks, metric = annoy_metrics)
        t.load('tree_50')
        space_name = 'l2'
        index = nmslib.init(method='hnsw', space=space_name)
        index.addDataPointBatch(df)
        index.loadIndex('hnsw_index80.bin')

        # Set index parameters
        # These are the most important onese
        NN = 50
        efC = 100

        num_threads = 4
        index_time_params = {'NN': NN, 'indexThreadQty': num_threads, 'efConstruction': efC}

        print('Initialized ...')
        index2 = nmslib.init(method='sw-graph', space=space_name, data_type=nmslib.DataType.DENSE_VECTOR)
        index2.addDataPointBatch(df)
        # Create an index
        start = time.time()
        # index2.createIndex(index_time_params)
        index2.loadIndex('sw_graph_index100.bin')
        end = time.time()
        print('Index-time parameters', index_time_params)
        print('Indexing time = %f' % (end - start))

        num_query = 50
        max_multiplier = 15
        time_st, time_bf, time_hnsw, time_sw, recall_array_st, recall_array_hnsw, recall_array_sw = search_tree_and_time2(
            df, t, index,index2, num_query=num_query, max_multiplier=max_multiplier)
        print('######## Finish ###########')
        result = pd.DataFrame(
            [time_st, time_bf, time_hnsw, time_sw, recall_array_st, recall_array_hnsw, recall_array_sw]).transpose()
        result.columns = ['st', 'bf', 'hnsw', 'sw', 'recall_st', 'recall_hnsw', 'recall_sw']
        result.to_csv('fig_data.csv',index=False)
    else:
        result = pd.read_csv('fig_data.csv')
    print(result)
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(result.recall_st, result.st, '-x')
    ax.plot(result.recall_hnsw, result.hnsw, '-o')
    ax.plot(result.recall_sw,result.sw,'-*')
    ax.plot(1.0, 6.2, 'ro')
    ax.set_yscale('log')
    plt.legend(['annoy','hnsw(nmslib)', 'sw-graph(nmslib)','brute force'], loc='best')
    plt.grid()
    plt.xlabel('Recall')
    plt.ylabel('Query per second 1/s')
    plt.savefig('ext2.png',dpi=300)
    plt.show()






