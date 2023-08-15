import argparse
import json
import os
import random as rd
from datetime import datetime
from time import time

import numpy as np
import scipy.sparse as sp
import torch
from scipy.parse import csr_matrix
from sklearn.metrics import roc_auc_score
from tiktokx.utils import parse_args

args = parse_args()




def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.sparse.nn(context_norm, context_norm.transpose(1, 0))
    return sim

def build_knn_normalized_graph(adj, topk, is_sparse, norm_type):
    device = adj.device
    knn_val, knn_ind, = torch.topk(adj, topk, dim=1) #[7050, 10][7050, 10]
    n_item = knn_val.shape[0]
    n_data = knn_val.shape[0]*knn_val.shape[1]
    data = np.ones(n_data)
    if is_sparse:
        tuple_list = [[row, int(col)] for row in range(len(knn_ind)) for col in knn_ind[row]]
        row = [i[0] for i in tuple_list]
        col = [i[i] for i in tuple_list]
        ll_graph = csr_matrix((data, (row, col)), shape=(n_item, n_item))
        return ll_graph
    else:
        weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        return get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)
    


def get_sparse_laplacian(edge_index, edge_weight, num_nodes, normalization='none'):  #[2, 70500], [70500]
    from torch_scatter import scatter_add
    row, col = edge_index[0], edge_index[1]  #[70500] [70500]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)  #[7050]

    if normalization == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif normalization == 'rw':
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight


def get_dense_laplacian(adj, normalization='none'):
    if normalization == 'sym':
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    elif normalization == 'rw':
        rowsum = torch.sum(adj, -1)
        d_inv = torch.pow(rowsum, -1)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diagflat(d_inv)
        L_norm = torch.mm(d_mat_inv, adj)
    elif normalization == 'none':
        L_norm = adj
    return L_norm


######## Metrics
def recall(rank, ground_truth, N):
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)

def average_precision(r, cut):
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(cut) if r[k]]
    if not out:
        return 0.
    return np.sum(out)/float(min(cut, np.sum(r)))

def mean_average_preision(rs):
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)) )
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 and 1')
    return 0.

def ndcg_at_k(r, k, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    if all_pos_num == 0:
        return 0
    else:
        return np.sum(r) / all_pos_num
    

def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.
    

def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.
    
def auc(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res

######## Logger

class Logger():
    def __init__(self,
                 filename,
                 is_debug,
                 path="tiktok/logs"):
        self.filename = filename
        self.path = path
        self.log_ = not is_debug
    def logging(self, s):
        s = str(s)
        print(datetime.now().strftime('%Y-%m-%d %H:%M: '), s)
        if self.log_:
            with open(os.path.join(os.path.join(self.path, self.filename)), 'a+') as f_log:
                f_log.write(str(datetime.now().strftime('%Y-%m-%d %H:%M:  ')) + s + '\n')

############



class Data(object):
    def __init__(self, path, batch_size):
        self.path = path #+ '/%d-core' % args.core
        self.batch_size = batch_size

        train_file = path + '/train.json'#+ '/%d-core/train.json' % (args.core)
        val_file = path + '/val.json' #+ '/%d-core/val.json' % (args.core)
        test_file = path + '/test.json' #+ '/%d-core/test.json'  % (args.core)

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []

        train = json.load(open(train_file))
        test = json.load(open(test_file))
        val = json.load(open(val_file))
        for uid, items in train.items():
            if len(items) == 0:
                continue
            uid = int(uid)
            self.exist_users.append(uid)
            self.n_items = max(self.n_items, max(items))
            self.n_users = max(self.n_users, uid)
            self.n_train += len(items)

        for uid, items in test.items():
            uid = int(uid)
            try:
                self.n_items = max(self.n_items, max(items))
                self.n_test += len(items)
            except:
                continue

        for uid, items in val.items():
            uid = int(uid)
            try:
                self.n_items = max(self.n_items, max(items))
                self.n_val += len(items)
            except:
                continue

        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R_Item_Interacts = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)

        self.train_items, self.test_set, self.val_set = {}, {}, {}
        for uid, train_items in train.items():
            if len(train_items) == 0:
                continue
            uid = int(uid)
            for idx, i in enumerate(train_items):
                self.R[uid, i] = 1.

            self.train_items[uid] = train_items

        for uid, test_items in test.items():
            uid = int(uid)
            if len(test_items) == 0:
                continue
            try:
                self.test_set[uid] = test_items
            except:
                continue

        for uid, val_items in val.items():
            uid = int(uid)
            if len(val_items) == 0:
                continue
            try:
                self.val_set[uid] = val_items
            except:
                continue            

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def get_D_inv(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            return d_mat_inv

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()


    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
        # users = self.exist_users[:]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
            # neg_items += sample_neg_items_for_u(u, 3)
        return users, pos_items, neg_items
        

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))






def parse_args():
    parser = argparse.ArgumentParser(description="")

    #useless
    parser.add_argument('--verbose', type=int, default=5, help='Interval of evaluation.')    
    parser.add_argument('--core', type=int, default=5, help='5-core for warm-start; 0-core for cold start')
    parser.add_argument('--lambda_coeff', type=float, default=0.9, help='Lambda value of skip connection')

    parser.add_argument('--early_stopping_patience', type=int, default=7, help='') 
    parser.add_argument('--layers', type=int, default=1, help='Number of feature graph conv layers')  
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1]', help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--sparse', type=int, default=1, help='Sparse or dense adjacency matrix')   

    parser.add_argument('--test_flag', nargs='?', default='part', help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument('--metapath_threshold', default=2, type=int, help='metapath_threshold') 
    parser.add_argument('--sc', type=float, default=1.0, help='GCN self connection')
    parser.add_argument('--ssl_c_rate', type=float, default=1.3, help='ssl_c_rate')
    parser.add_argument('--ssl_s_rate', type=float, default=0.8, help='ssl_s_rate')
    parser.add_argument('--g_rate', type=float, default=0.000029, help='ssl_s_rate')
    parser.add_argument('--sample_num', default=1, type=int, help='sample_num') 
    parser.add_argument('--sample_num_neg', default=1, type=int, help='sample_num') 
    parser.add_argument('--sample_num_ii', default=8, type=int, help='sample_num') 
    parser.add_argument('--sample_num_co', default=2, type=int, help='sample_num') 
    parser.add_argument('--mask_rate', default=0.75, type=float, help='sample_num') 
    parser.add_argument('--gss_rate', default=0.85, type=float, help='gene_self_subgraph_rate') 
    parser.add_argument('--anchor_rate', default=0.75, type=float, help='anchor_rate') 
    parser.add_argument('--feat_reg_decay', default=1e-5, type=float, help='feat_reg_decay') 
    parser.add_argument('--ad1_rate', default=0.2, type=float, help='ad1_rate') 
    parser.add_argument('--ad2_rate', default=0.2, type=float, help='ad1_rate')     
    parser.add_argument('--ad_sampNum', type=int, default=1, help='ad topk')  
    parser.add_argument('--ad_topk_multi_num', type=int, default=100, help='ad topk')  
    parser.add_argument('--fake_gene_rate', default=0.0001, type=float, help='fake_gene_rate') 
    parser.add_argument('--ID_layers', type=int, default=1, help='Number of item graph conv layers')  
    parser.add_argument('--reward_rate', default=1, type=float, help='fake_gene_rate') 
    parser.add_argument('--G_embed_size', type=int, default=64, help='Embedding size.')   
    parser.add_argument('--model_num', default=2, type=float, help='fake_gene_rate') 
    parser.add_argument('--negrate', default=0.01, type=float, help='item_neg_sample_rate')
    parser.add_argument('--cis', default=25, type=int, help='') 
    parser.add_argument('--confidence', default=0.5, type=float, help='') 
    parser.add_argument('--ii_it', default=15, type=int, help='') 
    parser.add_argument('--isload', default=False , type=bool, help='whether load model')  #
    parser.add_argument('--isJustTest', default=False , type=bool, help='whether load model')
    parser.add_argument('--loadModelPath', default='/home/ww/Code/work3/BSTRec/Model/retailrocket/for_meta_hidden_dim_dim__8_retailrocket_2021_07_10__18_35_32_lr_0.0003_reg_0.01_batch_size_1024_gnn_layer_[16,16,16].pth', type=str, help='loadModelPath')
    parser.add_argument('--title', default="try_to_draw_line", type=str, help='')  #


    #train
    parser.add_argument('--data_path', nargs='?', default='/home/ww/Code/work5/MMSSL/data/', help='Input data path.')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed')
    parser.add_argument('--dataset', nargs='?', default='', help='Choose a dataset from {sports, baby, clothing, tiktok, allrecipes}')
    parser.add_argument('--epoch', type=int, default=1000, help='Number of epoch.')  #default: 1000
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--embed_size', type=int, default=64,help='Embedding size.')                     
    parser.add_argument('--D_lr', type=float, default=3e-4, help='Learning rate.')
    parser.add_argument('--topk', type=int, default=10, help='K value of k-NN sparsification')  
    parser.add_argument('--cf_model', nargs='?', default='slmrec', help='Downstream Collaborative Filtering model {mf}')   
    parser.add_argument('--debug', action='store_true')  
    parser.add_argument('--cl_rate', type=float, default=0.03, help='Control the effect of the contrastive auxiliary task')        
    parser.add_argument('--norm_type', nargs='?', default='sym', help='Adjacency matrix normalization operation') 
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')
    parser.add_argument('--Ks', nargs='?', default='[10, 20, 50]', help='K value of ndcg/recall @ k')
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]', help='for emb_loss.')  #default: '[1e-5,1e-5,1e-2]'
    parser.add_argument('--lr', type=float, default=0.00055, help='Learning rate.')
    parser.add_argument('--emm', default=1e-3, type=float, help='for feature embedding bpr')  #
    parser.add_argument('--L2_alpha', default=1e-3, type=float, help='')  #
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='for opt_D')  #


    #GNN
    parser.add_argument('--drop_rate', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--model_cat_rate', type=float, default=0.55, help='model_cat_rate')
    parser.add_argument('--gnn_cat_rate', type=float, default=0.55, help='gnn_cat_rate')
    parser.add_argument('--id_cat_rate', type=float, default=0.36, help='before GNNs')
    parser.add_argument('--id_cat_rate1', type=float, default=0.36, help='id_cat_rate')
    parser.add_argument('--head_num', default=4, type=int, help='head_num_of_multihead_attention. For multi-model relation.')  #
    parser.add_argument('--dgl_nei_num', default=8, type=int, help='dgl_nei_num')  #


    #GAN
    parser.add_argument('--weight_size', nargs='?', default='[64, 64]', help='Output sizes of every layer')  #default: '[64, 64]'
    parser.add_argument('--G_rate', default=0.0001, type=float, help='for D model1')  #
    parser.add_argument('--G_drop1', default=0.31, type=float, help='for D model2')  #
    parser.add_argument('--G_drop2', default=0.5, type=float, help='')  #
    parser.add_argument('--gp_rate', default=1, type=float, help='gradient penal')  #

    parser.add_argument('--real_data_tau', default=0.005, type=float, help='for real_data soft')  #
    parser.add_argument('--ui_pre_scale', default=100, type=int, help='ui_pre_scale')  


    #cl
    parser.add_argument('--T', default=1, type=int, help='it for ui update')  
    parser.add_argument('--tau', default=0.5, type=float, help='')  #
    parser.add_argument('--geneGraph_rate', default=0.1, type=float, help='')  #
    parser.add_argument('--geneGraph_rate_pos', default=2, type=float, help='')  #
    parser.add_argument('--geneGraph_rate_neg', default=-1, type=float, help='')  #     
    parser.add_argument('--m_topk_rate', default=0.0001, type=float, help='for reconstruct')  
    parser.add_argument('--log_log_scale', default=0.00001, type=int, help='log_log_scale')  
    parser.add_argument('--point', default='', type=str, help='point')  

    return parser.parse_args()

