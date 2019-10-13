import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse

from main import init

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size != k:
        raise ValueError('Ranking List length < k')
    return np.sum((2**r - 1) / np.log2(np.arange(2, r.size + 2)))

def ndcg_at_k(r, k):
    sort_r = sorted(r,reverse = True)
    idcg = dcg_at_k(sort_r, k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='model name to save/load checkpoints')
    parser.add_argument('-c', '--checkpoint')
    args = parser.parse_args()

    model, misc = init(args.model, args.checkpoint)
    model.eval()

    file_name = 'tripadv_iter40000_margin0.2_covlr0.1_d10_fixallcovest_nodataleak_8aspects'
    val_tst = 'tst'

    print(file_name)
    paras = pickle.load(open('../results/learnt_models/' + file_name + '.paras','rb'))
    U = paras['U']
    V = paras['V']
    W = paras['W']

    user_candidate_ls = pickle.load(open('../data/user_candidates_150_' + val_tst + '.ls','rb'))

    train_sparse_ts = {}
    test_sparse_ts = {}
    zero_ratings = np.zeros(W.shape[0])

    print('loading training data...')
    with open('../data/processed_user5item5.trn') as fin:
        for line in fin.readlines():
            entry = line.strip().split(':')
            user_idx = int(entry[1].split('\t')[0])
            item_idx = int(entry[2].split('\t')[0])
            rating_ls = entry[3].strip().split('\t')
            ratings = np.array([float(rating) for rating in rating_ls])
            ratings[ratings<0] = 0
            if user_idx not in train_sparse_ts:
                train_sparse_ts[user_idx] = {}
            train_sparse_ts[user_idx][item_idx] = ratings

    print('loading testing data...')
    with open('../data/processed_user5item5.' + val_tst) as fin:
        for line in fin.readlines():
            entry = line.strip().split(':')
            user_idx = int(entry[1].split('\t')[0])
            item_idx = int(entry[2].split('\t')[0])
            rating_ls = entry[3].strip().split('\t')
            ratings = np.array([float(rating) for rating in rating_ls])
            ratings[ratings<0] = 0
            if user_idx not in test_sparse_ts:
                test_sparse_ts[user_idx] = {}
            test_sparse_ts[user_idx][item_idx] = ratings
    '''
    dup_cnt = 0
    for user in train_sparse_ts:
        for item in train_sparse_ts[user]:
            if user in test_sparse_ts:
                if item in test_sparse_ts[user]:
                    dup_cnt += 1
                    print(train_sparse_ts[user][item])
                    print(test_sparse_ts[user][item])
                    print('--------------------------------------------------')

    print(dup_cnt)

    '''
    for user in user_candidate_ls.keys():
        if user not in train_sparse_ts:
            train_sparse_ts[user] = {}
        if user not in test_sparse_ts:
            test_sparse_ts[user] = {}
        for item in user_candidate_ls[user]:
            if item not in train_sparse_ts[user]:
                train_sparse_ts[user][item] = zero_ratings
            if item not in test_sparse_ts[user]:
                test_sparse_ts[user][item] = zero_ratings

    fout_eval_result = open('../results/allaspect_ndcg_result/' + file_name + val_tst + 'aspect8.ndcg', 'w')

    kk = [10,20,50]

    #choose an aspect to evaluate
    aspect_ls = ['Overall', 'Sleep Quality', 'Service', 'Value', 'Rooms', 'Cleanliness', 'Location', 'Check in / front desk', 'Business service']

    for aspect in range(8):
        for k in kk:
            accum_ndcg = 0
            cnt = 0
            for user in user_candidate_ls.keys():
                if user in test_sparse_ts:
                    #train_overall_candidates = np.array([train_sparse_ts[user][item][aspect] for item in user_candidate_ls[user]])
                    test_overall_candidates = np.array([test_sparse_ts[user][item][aspect] for item in user_candidate_ls[user]])
                    if sum(test_overall_candidates) > 0:
                        #print(sum(test_overall))
                        overall_preds_candidates = np.einsum('d, Nd -> Nd', U[user], V[user_candidate_ls[user]])
                        overall_preds_candidates = np.einsum('Nd, d -> N', overall_preds_candidates, W[aspect])

                        #overall_preds_candidates[train_overall_candidates>0] = 0
                        rank_pred_keys = np.argsort(overall_preds_candidates)
                        rank_pred_keys = rank_pred_keys[::-1]
                        ranked_test = test_overall_candidates[rank_pred_keys]
                        ndcg_k = ndcg_at_k(ranked_test, k)
                        #if dcg_at_k(ranked_test, k):
                        #print(ndcg_k)
                        accum_ndcg += ndcg_k
                        cnt += 1

            aver_ndcg = accum_ndcg/cnt
            print('aspect: ' + aspect_ls[aspect] + '\nndcg: ' + str(aver_ndcg))
            fout_eval_result.write('aspect: ' + aspect_ls[aspect] + '\nndcg: ' + str(aver_ndcg) + '\n')
        fout_eval_result.write('-------------------------------------------------------------------\n')

    fout_eval_result.close()
