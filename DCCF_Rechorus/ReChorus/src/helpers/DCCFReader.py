import os
import numpy as np
from time import time
from tqdm import tqdm
import scipy.sparse as sp
import logging
import pandas as pd

from utils import utils

# from helpers.BaseReader import BaseReader

class DCCFReader(object):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='data/',
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='Grocery_and_Gourmet_Food',
                            help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default='\t',
                            help='sep of csv file.')
        return parser
    
    def __init__(self, args):
        #
        self.sep = args.sep
        self.path = args.path
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.n_batch = args.n_batch
        self._read_data()
        #

        self.train_clicked_set = dict()  # store the clicked item set of each user in training set
        self.residual_clicked_set = dict()  # store the residual clicked item set of each user
        for key in ['train', 'dev', 'test']:
            df = self.data_df[key]
            for t in df:
                uid = t[0]
                iid = t[1]
                if uid not in self.train_clicked_set:
                    self.train_clicked_set[uid] = set()
                    self.residual_clicked_set[uid] = set()
                if key == 'train':
                    self.train_clicked_set[uid].add(iid)
                else:
                    self.residual_clicked_set[uid].add(iid)

    def _read_data(self):
        logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.path, self.dataset))
        self.data_df = dict()
        #
        self.data_mat = dict()
        for key in ['train', 'dev', 'test']:
            self.data_df[key] = pd.read_csv(os.path.join(self.path, self.dataset, key + '.csv'), sep=self.sep).reset_index(drop=True).sort_values(by = ['user_id','time'])
            self.data_df[key] = utils.eval_list_columns(self.data_df[key])
            n_user, n_item = self.data_df[key]['user_id'].max() + 1, self.data_df[key]['item_id'].max() + 1
            matrix = np.zeros((n_user, n_item))
            self.data_df[key] = self.data_df[key].apply(lambda x: tuple(x), axis=1).values.tolist()
            for instance in self.data_df[key]:
                matrix[instance[0], instance[1]] = 1.0
            self.data_mat[key] = sp.coo_matrix(matrix)

        self.n_users, self.n_items = self.data_mat['train'].shape[0], self.data_mat['train'].shape[1]
        self.n_train, self.n_test = len(self.data_mat['train'].row), len(self.data_mat['test'].row)

        self.R = self.data_mat['train'].todok()
        self.train_items, self.test_set = {}, {}
        train_uid, train_iid = self.data_mat['train'].row, self.data_mat['train'].col
        for i in range(len(train_uid)):
            uid = train_uid[i]
            iid = train_iid[i]
            if uid not in self.train_items:
                self.train_items[uid] = [iid]
            else:
                self.train_items[uid].append(iid)
        test_uid, test_iid = self.data_mat['test'].row, self.data_mat['test'].col
        for i in range(len(test_uid)):
            uid = test_uid[i]
            iid = test_iid[i]
            if uid not in self.test_set:
                self.test_set[uid] = [iid]
            else:
                self.test_set[uid].append(iid)
        
        self.plain_adj = self.get_adj_mat()
        self.all_h_list, self.all_t_list, self.all_v_list = self.load_adjacency_list_data(self.plain_adj)
        #
        self.data_df = dict()
        for key in ['train', 'dev', 'test']:
            self.data_df[key] = pd.read_csv(os.path.join(self.path, self.dataset, key + '.csv'), sep=self.sep).reset_index(drop=True).sort_values(by = ['user_id','time'])
            self.data_df[key] = utils.eval_list_columns(self.data_df[key])

        logging.info('Counting dataset statistics...')
        key_columns = ['user_id','item_id','time']
        if 'label' in self.data_df['train'].columns: # Add label for CTR prediction
            key_columns.append('label')
        self.all_df = pd.concat([self.data_df[key][key_columns] for key in ['train']])
        self.n_users, self.n_items = self.all_df['user_id'].max() + 1, self.all_df['item_id'].max() + 1
        for key in ['dev', 'test']:
            if 'neg_items' in self.data_df[key]:
                neg_items = np.array(self.data_df[key]['neg_items'].tolist())
                assert (neg_items >= self.n_items).sum() == 0  # assert negative items don't include unseen ones
        logging.info('"# user": {}, "# item": {}'.format(
            self.n_users - 1, self.n_items - 1))
        if 'label' in key_columns:
            positive_num = (self.all_df.label==1).sum()
            logging.info('"# positive interaction": {} ({:.1f}%)'.format(
				positive_num))

    def get_adj_mat(self):
        adj_mat = self.create_adj_mat()
        return adj_mat

    def create_adj_mat(self):
        t1 = time()
        rows = self.R.tocoo().row
        cols = self.R.tocoo().col
        new_rows = np.concatenate([rows, cols + self.n_users], axis=0)
        new_cols = np.concatenate([cols + self.n_users, rows], axis=0)
        adj_mat = sp.coo_matrix((np.ones(len(new_rows)), (new_rows, new_cols)), shape=[self.n_users + self.n_items, self.n_users + self.n_items]).tocsr().tocoo()
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)
        return adj_mat.tocsr()

    def load_adjacency_list_data(self, adj_mat):
        tmp = adj_mat.tocoo()
        all_h_list = list(tmp.row)
        all_t_list = list(tmp.col)
        all_v_list = list(tmp.data)

        return all_h_list, all_t_list, all_v_list
    
    def uniform_sample(self):
        users = np.random.randint(1, self.n_users, int(self.n_batch * self.batch_size))
        train_data = []
        for i, user in tqdm(enumerate(users), desc='Sampling Data', total=len(users)):
            pos_for_user = self.train_items[user]
            pos_index = np.random.randint(1, len(pos_for_user))
            pos_item = pos_for_user[pos_index]
            while True:
                neg_item = np.random.randint(0, self.n_items)
                if self.R[user, neg_item] == 1:
                    continue
                else:
                    break
            train_data.append([user, pos_item, neg_item])
            self.train_data = np.array(train_data)
        return len(self.train_data)
    
    

