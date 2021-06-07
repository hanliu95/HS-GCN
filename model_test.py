import math
import numpy as np
import pickle

f_para = open('../para/movie_load.para', 'rb')
para_load = pickle.load(f_para)
user_num = para_load['user_num']  # total number of users
item_num = para_load['item_num']  # total number of items
eva_size = 50

train_matrix = para_load['train_matrix']
train_matrix.data = np.array(train_matrix.data, dtype=np.int8)
train_matrix = train_matrix.toarray()  # the 0-1 matrix of testing set
test_matrix = para_load['test_matrix']
test_matrix.data = np.array(test_matrix.data, dtype=np.int8)
test_matrix = test_matrix.toarray()  # the 0-1 matrix of testing set
item_ids = np.array(list(range(item_num)))

f2 = open('./para/movie_codes.para', 'rb')
para_nodes = pickle.load(f2)
hash_codes = para_nodes['hash_codes']
hash_codes = hash_codes.astype(np.int8)

P = 0
HR = 0
NDCG = 0

def IDCG(num):
    idcg = 0
    for i in list(range(num)):
        idcg += 1/math.log(i+2)
    return idcg

def descend_sort(array):
    return -np.sort(-array)

for user_id, row in enumerate(train_matrix):
    can_item_ids = item_ids[~np.array(row, dtype=bool)]  # the id list of test items
    I = hash_codes[can_item_ids + user_num, :]
    u = hash_codes[user_id, :]
    inner_pro = np.matmul(u, I.T, dtype=np.int8).reshape(-1)
    sort_index = np.argsort(-inner_pro)
    hit_num = 0
    dcg = 0
    for i, item_id in enumerate(can_item_ids[sort_index][0:eva_size]):
        if test_matrix[user_id, item_id] > 0:
            hit_num = hit_num + 1
            dcg = dcg + 1 / math.log(i + 2)
    P += hit_num / eva_size
    HR += hit_num
    NDCG += dcg / IDCG(np.sum(descend_sort(test_matrix[user_id])[0:eva_size]))

P = P/user_num
HR = HR/np.sum(test_matrix)
NDCG = NDCG/user_num
print('HR@%d: %.4f; NDCG@%d: %.4f; P@%d: %.4f' % (eva_size, HR, eva_size, NDCG, eva_size, P))

