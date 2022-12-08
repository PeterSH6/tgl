import gen_graph_multithread
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
import itertools


def convert_list_to_np(lst, type_int=True):
    n = np.array(lst)
    if type_int:
        n = n.astype(int)
    return n

# Notice: Return Order: ext_full_indptr, ext_full_indices, ext_full_ts, ext_full_eid, df


def load_graph_cpp(dataset_name: str, end_id_exclude: float):

    start_time = time.time()

    # df = pd.read_csv('/data/tgl/{}/edges.csv'.format(dataset_name))  # LINUX
    full_df = pd.read_csv(
        '/home/ubuntu/data/{}/edges.csv'.format(dataset_name))  # AWS

    df_len = len(full_df)
    end_id_exclude = int(end_id_exclude * df_len)
    # slice
    df = full_df[:end_id_exclude]

    # df.rename(columns={'Unnamed: 0': 'eid'}, inplace=True)

    df_eid = df["Unnamed: 0"].values.astype(np.int64)
    df_src = df["src"].values.astype(np.int64)
    df_dst = df["dst"].values.astype(np.int64)
    df_ts = df["time"].values.astype(np.float64)
    df_introll = df["int_roll"].values.astype(np.int64)
    df_extroll = df["ext_roll"].values.astype(np.int64)

    py_num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1
    py_num_edges = len(df)

    # print('num_nodes: ', py_num_nodes)

    result = gen_graph_multithread.run(
        df_eid,
        df_src,
        df_dst,
        df_ts,
        df_introll,
        df_extroll,
        py_num_nodes,
        py_num_edges,
        128,
        dataset_name
    )

    # int_train_indptr = convert_list_to_np(result[0], True)
    # int_train_indices = convert_list_to_np(result[1], True)
    # int_train_ts = convert_list_to_np(result[2], False)
    # int_train_eid = convert_list_to_np(result[3], True)

    # int_full_indptr = convert_list_to_np(result[4], True)
    # int_full_indices = convert_list_to_np(result[5], True)
    # int_full_ts = convert_list_to_np(result[6], False)
    # int_full_eid = convert_list_to_np(result[7], True)

    ext_full_indptr = convert_list_to_np(result[8], True)
    ext_full_indices = convert_list_to_np(result[9], True)
    ext_full_ts = convert_list_to_np(result[10], False)
    ext_full_eid = convert_list_to_np(result[11], True)

    end_time = time.time()
    # print('Total Load Graph Time is {}'.format(end_time - start_time))

    return ext_full_indptr, ext_full_indices, ext_full_ts, ext_full_eid, df, full_df


def load_graph_python(dataset_name, end_id_exclude):
    add_reverse = False
    begin_time = time.time()

    df = pd.read_csv('/data/tgl/{}/edges.csv'.format(dataset_name))  # LINUX
    # df = pd.read_csv('/home/ubuntu/data/{}/edges.csv'.format(dataset_name)) # AWS

    df = df[:end_id_exclude]

    num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1
    # print('num_nodes: ', num_nodes)

    int_train_indptr = np.zeros(num_nodes + 1, dtype=np.int)
    int_train_indices = [[] for _ in range(num_nodes)]
    int_train_ts = [[] for _ in range(num_nodes)]
    int_train_eid = [[] for _ in range(num_nodes)]

    int_full_indptr = np.zeros(num_nodes + 1, dtype=np.int)
    int_full_indices = [[] for _ in range(num_nodes)]
    int_full_ts = [[] for _ in range(num_nodes)]
    int_full_eid = [[] for _ in range(num_nodes)]

    ext_full_indptr = np.zeros(num_nodes + 1, dtype=np.int)
    ext_full_indices = [[] for _ in range(num_nodes)]
    ext_full_ts = [[] for _ in range(num_nodes)]
    ext_full_eid = [[] for _ in range(num_nodes)]

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        src = int(row['src'])
        dst = int(row['dst'])
        if row['int_roll'] == 0:
            int_train_indices[src].append(dst)
            int_train_ts[src].append(row['time'])
            int_train_eid[src].append(idx)
            if add_reverse:
                int_train_indices[dst].append(src)
                int_train_ts[dst].append(row['time'])
                int_train_eid[dst].append(idx)
            # int_train_indptr[src + 1:] += 1
        if row['int_roll'] != 3:
            int_full_indices[src].append(dst)
            int_full_ts[src].append(row['time'])
            int_full_eid[src].append(idx)
            if add_reverse:
                int_full_indices[dst].append(src)
                int_full_ts[dst].append(row['time'])
                int_full_eid[dst].append(idx)
            # int_full_indptr[src + 1:] += 1
        ext_full_indices[src].append(dst)
        ext_full_ts[src].append(row['time'])
        ext_full_eid[src].append(idx)
        if add_reverse:
            ext_full_indices[dst].append(src)
            ext_full_ts[dst].append(row['time'])
            ext_full_eid[dst].append(idx)
        # ext_full_indptr[src + 1:] += 1

    for i in tqdm(range(num_nodes)):
        int_train_indptr[i + 1] = int_train_indptr[i] + \
            len(int_train_indices[i])
        int_full_indptr[i + 1] = int_full_indptr[i] + len(int_full_indices[i])
        ext_full_indptr[i + 1] = ext_full_indptr[i] + len(ext_full_indices[i])

    print("Chaining")

    chain_start_time = time.time()

    int_train_indices = np.array(list(itertools.chain(*int_train_indices)))
    int_train_ts = np.array(list(itertools.chain(*int_train_ts)))
    int_train_eid = np.array(list(itertools.chain(*int_train_eid)))

    int_full_indices = np.array(list(itertools.chain(*int_full_indices)))
    int_full_ts = np.array(list(itertools.chain(*int_full_ts)))
    int_full_eid = np.array(list(itertools.chain(*int_full_eid)))

    ext_full_indices = np.array(list(itertools.chain(*ext_full_indices)))
    ext_full_ts = np.array(list(itertools.chain(*ext_full_ts)))
    ext_full_eid = np.array(list(itertools.chain(*ext_full_eid)))

    chain_end_time = time.time()

    print("Chain cost time {} seconds".format(
        chain_end_time - chain_start_time))

    print('Sorting...')

    def tsort(i, indptr, indices, t, eid):
        beg = indptr[i]
        end = indptr[i + 1]
        sidx = np.argsort(t[beg:end])
        indices[beg:end] = indices[beg:end][sidx]
        t[beg:end] = t[beg:end][sidx]
        eid[beg:end] = eid[beg:end][sidx]

    for i in tqdm(range(int_train_indptr.shape[0] - 1)):
        tsort(i, int_train_indptr, int_train_indices,
              int_train_ts, int_train_eid)
        tsort(i, int_full_indptr, int_full_indices, int_full_ts, int_full_eid)
        tsort(i, ext_full_indptr, ext_full_indices, ext_full_ts, ext_full_eid)

    end_time = time.time()

    print("Sort time in python: {} seconds".format(end_time - chain_end_time))
    return ext_full_indptr, ext_full_indices, ext_full_ts, ext_full_eid


if __name__ == "__main__":
    dataset_name = 'REDDIT'
    end_id_exclude = 500000
    a = load_graph_cpp(dataset_name, end_id_exclude)
    o = load_graph_python(dataset_name, end_id_exclude)

    print(len(a), len(o))
    cnt = 0
    for i in range(len(a)):
        if a[i] != o[i]:
            cnt += 1
            if cnt % 1000 == 0:
                print("Times: {}, {} != {}\n".format(cnt, a[i], o[i]))
    print(cnt)
