import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from models.BaseModel import GeneralModel

def load_adjacency_list_data(adj_mat):
	tmp = adj_mat.tocoo()
	all_h_list = list(tmp.row)
	all_t_list = list(tmp.col)
	all_v_list = list(tmp.data)
	return all_h_list, all_t_list, all_v_list

class DCCFEncoder(nn.Module):
	@staticmethod
	def _convert_sp_mat_to_sp_tensor(X):
		coo = X.tocoo()
		i = torch.LongTensor([coo.row, coo.col])
		v = torch.from_numpy(coo.data).float()
		return torch.sparse.FloatTensor(i, v, coo.shape)
	
	def __init__(self, user_count, item_count, emb_size, n_intents, norm_adj, n_layers=2, temp=1):
		super(DCCFEncoder, self).__init__()
		self.user_count = user_count
		self.item_count = item_count
		self.emb_size = emb_size
		self.n_intents = n_intents
		self.layers = [emb_size] * n_layers
		self.norm_adj = norm_adj
		self.all_h_list, self.all_t_list, self.all_v_list = load_adjacency_list_data(self.norm_adj)
		self.all_h_list = torch.LongTensor(self.all_h_list).cuda()
		self.all_t_list = torch.LongTensor(self.all_t_list).cuda()
		self.embedding_dict = self._init_model()
		self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).cuda()
		self.A_in_shape = self.norm_adj.tocoo().shape
		self.temp = temp

	def _init_model(self):
		embedding_dict = self._define_params()
		nn.init.xavier_normal_(embedding_dict['user_emb'].weight)
		nn.init.xavier_normal_(embedding_dict['item_emb'].weight)
		return embedding_dict
	
	def _define_params(self):
		initializer = nn.init.xavier_uniform_
		user_emb = nn.Embedding(self.user_count, self.emb_size)
		item_emb = nn.Embedding(self.item_count, self.emb_size)
		user_intent = nn.Parameter(initializer(torch.empty(self.emb_size, self.n_intents)), requires_grad=True)
		item_intent = nn.Parameter(initializer(torch.empty(self.emb_size, self.n_intents)), requires_grad=True)
		embedding_dict = nn.ParameterDict({'user_emb': user_emb, 'item_emb': item_emb, 
									'user_intent': user_intent, 'item_intent': item_intent,})
		return embedding_dict
	
	def _adaptive_mask(self, head_embeddings, tail_embeddings):
		# 计算解耦掩码M和应用后求得的隐性关系矩阵G
		head_embeddings = nn.functional.normalize(head_embeddings)
		tail_embeddings = nn.functional.normalize(tail_embeddings)

		edge_alpha = (torch.sum(head_embeddings * tail_embeddings, dim=1).view(-1) + 1) / 2
		indices = torch.stack([self.all_h_list, self.all_t_list]) 
		values = edge_alpha 
		A_tensor = torch.sparse_coo_tensor(indices, values, size=self.A_in_shape).cuda()

		A_tensor = A_tensor.coalesce()
		indices = A_tensor.indices() 
		values = A_tensor.values() 
		rows = indices[0] 

		row_sums = torch.zeros(A_tensor.size(0), device=A_tensor.device).scatter_add(0, rows, values)

		D_scores_inv = row_sums.pow(-1)
		D_scores_inv[torch.isnan(D_scores_inv)] = 0 
		G_indices = torch.stack([self.all_h_list, self.all_t_list], dim=0) 
		G_values = D_scores_inv[self.all_h_list] * edge_alpha 
		G = torch.sparse_coo_tensor(G_indices, G_values, size=(self.A_in_shape[0], self.A_in_shape[1]))
		return G

	def forward(self):
		all_embeddings = [torch.concat([self.embedding_dict['user_emb'].weight, self.embedding_dict['item_emb'].weight], 0)]
		user_intent, item_intent = self.embedding_dict['user_intent'], self.embedding_dict['item_intent']
		gnn_embeddings = []
		int_embeddings = []
		gaa_embeddings = []
		iaa_embeddings = []

		for k in range(len(self.layers)):

			gnn_layer_embeddings = torch.sparse.mm(self.sparse_norm_adj, all_embeddings[k])
			u_embeddings, i_embeddings = torch.split(all_embeddings[k], [self.user_count, self.item_count], 0)

			# 获取意图编码
			u_int_embeddings = torch.softmax(u_embeddings @ user_intent, dim=1) @ user_intent.T
			i_int_embeddings = torch.softmax(i_embeddings @ item_intent, dim=1) @ item_intent.T
			int_layer_embeddings = torch.concat([u_int_embeddings, i_int_embeddings], 0)

			gnn_head_embeddings = torch.index_select(gnn_layer_embeddings, 0, self.all_h_list)

			gnn_tail_embeddings = torch.index_select(gnn_layer_embeddings, 0, self.all_t_list)
			int_head_embeddings = torch.index_select(int_layer_embeddings, 0, self.all_h_list)
			int_tail_embeddings = torch.index_select(int_layer_embeddings, 0, self.all_t_list)

			G_graph = self._adaptive_mask(gnn_head_embeddings, gnn_tail_embeddings)
			G_inten = self._adaptive_mask(int_head_embeddings, int_tail_embeddings)
			gaa_layer_embeddings = torch.sparse.mm(G_graph, all_embeddings[k])
			iaa_layer_embeddings = torch.sparse.mm(G_inten, all_embeddings[k])

			gnn_embeddings.append(gnn_layer_embeddings)
			int_embeddings.append(int_layer_embeddings)
			gaa_embeddings.append(gaa_layer_embeddings)
			iaa_embeddings.append(iaa_layer_embeddings)

			all_embeddings.append(gnn_layer_embeddings + int_layer_embeddings + gaa_layer_embeddings + iaa_layer_embeddings + all_embeddings[k])

		all_embeddings = torch.stack(all_embeddings, dim=1)
		all_embeddings = torch.mean(all_embeddings, dim=1, keepdim=False)
		self.ua_embeddings, self.ia_embeddings = torch.split(all_embeddings, [self.user_count, self.item_count], 0)

		user_embeddings = self.ua_embeddings
		item_embeddings = self.ia_embeddings

		self.gnn_embeddings = gnn_embeddings
		self.int_embeddings = int_embeddings
		self.gaa_embeddings = gaa_embeddings
		self.iaa_embeddings = iaa_embeddings

		return user_embeddings, item_embeddings, gnn_embeddings, int_embeddings, gaa_embeddings, iaa_embeddings

class DCCFBase(object):
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=32,
					  		help='Size of embedding vectors.')
		parser.add_argument('--n_layers', type=int, default=2,
					  		help='Number of DCCF layers.')
		parser.add_argument('--n_intents', type=int, default=128,
					  		help='Size of the intent of users/items.')
		parser.add_argument('--temp', type=float, default=1, 
					  		help='temperature in ssl loss')
		return parser

	@staticmethod
	def build_adjmat(user_count, item_count, train_mat, selfloop_flag=False):
		R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
		for user in train_mat:
			for item in train_mat[user]:
				R[user, item] = 1
		R = R.tolil()

		adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
		adj_mat = adj_mat.tolil()

		adj_mat[:user_count, user_count:] = R
		adj_mat[user_count:, :user_count] = R.T
		adj_mat = adj_mat.todok()

		def normalized_adj_single(adj):
			rowsum = np.array(adj.sum(1)) + 1e-10
			d_inv_sqrt = np.power(rowsum, -0.5).flatten()
			d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
			d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
			bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
			return bi_lap.tocoo()

		if selfloop_flag:
			norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
		else:
			norm_adj_mat = normalized_adj_single(adj_mat)

		return norm_adj_mat.tocsr()
	def _base_init(self, args, corpus):
		self.emb_size = args.emb_size
		self.n_layers = args.n_layers
		self.n_intents = args.n_intents
		self.temp = args.temp
		self.norm_adj = self.build_adjmat(corpus.n_users, corpus.n_items, corpus.train_clicked_set)
		self._base_define_params()
		self.apply(self.init_weights)

	def _base_define_params(self):
		self.encoder = DCCFEncoder(self.user_num, self.item_num, self.emb_size, self.n_intents, self.norm_adj, self.n_layers, self.temp)

	def forward(self, feed_dict):
		self.check_list = []
		user, items = feed_dict['user_id'], feed_dict['item_id']
		ua_embed, ia_embed, gnn_embeddings, int_embeddings, gaa_embeddings, iaa_embeddings = self.encoder.forward()
		u_embed = ua_embed[user.long()]
		i_embed = ia_embed[items.long()]

		prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1)

		u_v = u_embed.repeat(1,items.shape[1]).view(items.shape[0],items.shape[1],-1)
		i_v = i_embed

		out_dict = {'prediction': prediction.view(feed_dict['batch_size'], -1),
				'user_id': user, 'item_id':items,
				'u_embed':ua_embed,'i_embed':ia_embed,
				'u_v': u_v, 'i_v':i_v,
				'gnn_embed': gnn_embeddings, 'int_embed': int_embeddings,
				'gaa_embed': gaa_embeddings, 'iaa_embed': iaa_embeddings,
				'batch_size': feed_dict['batch_size']}
		return out_dict

class DCCF(GeneralModel, DCCFBase):
	reader = 'BaseReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size', 'n_layers', 'n_intents', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = DCCFBase.parse_model_args(parser)
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		GeneralModel.__init__(self, args, corpus)
		self._base_init(args, corpus)
		self.user_count = self.encoder.user_count
		self.item_count = self.encoder.item_count

	def forward(self, feed_dict):
		out_dict = DCCFBase.forward(self, feed_dict)
		self.user_id, self.item_id = feed_dict['user_id'], feed_dict['item_id']
		gnn_embeddings = out_dict['gnn_embed']
		int_embeddings = out_dict['int_embed']
		gaa_embeddings = out_dict['gaa_embed']
		iaa_embeddings = out_dict['iaa_embed']
		batch_size = feed_dict['batch_size']
		u_v, i_v = out_dict['u_v'], out_dict['i_v']
		ua_embed, ia_embed = out_dict['u_embed'], out_dict['i_embed']
		return {'user_id': self.user_id, 'item_id': self.item_id,
				'prediction': out_dict['prediction'],
				'u_v':u_v, 'i_v':i_v,
				'gnn_embed': gnn_embeddings, 'int_embed': int_embeddings,
				'gaa_embed': gaa_embeddings, 'iaa_embed': iaa_embeddings,
				'u_embed': ua_embed, 'i_embed':ia_embed,
				'batch_size':batch_size}

	def cal_ssl_loss(self, users, items, gnn_emb, int_emb, gaa_emb, iaa_emb):
		users = torch.unique(users).long()
		items = torch.unique(items).long()

		cl_loss = 0.0

		def cal_loss(emb1, emb2):
			pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.temp)
			neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.temp), axis=1)

			loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))

			loss /= pos_score.shape[0]
			return loss

		for i in range(len(gnn_emb)):
			u_gnn_embs, i_gnn_embs = torch.split(gnn_emb[i], [self.user_count, self.item_count], 0)
			u_int_embs, i_int_embs = torch.split(int_emb[i], [self.user_count, self.item_count], 0)
			u_gaa_embs, i_gaa_embs = torch.split(gaa_emb[i], [self.user_count, self.item_count], 0)
			u_iaa_embs, i_iaa_embs = torch.split(iaa_emb[i], [self.user_count, self.item_count], 0)

			u_gnn_embs = F.normalize(u_gnn_embs[users], dim=1)
			u_int_embs = F.normalize(u_int_embs[users], dim=1)
			u_gaa_embs = F.normalize(u_gaa_embs[users], dim=1)
			u_iaa_embs = F.normalize(u_iaa_embs[users], dim=1)

			i_gnn_embs = F.normalize(i_gnn_embs[items], dim=1)
			i_int_embs = F.normalize(i_int_embs[items], dim=1)
			i_gaa_embs = F.normalize(i_gaa_embs[items], dim=1)
			i_iaa_embs = F.normalize(i_iaa_embs[items], dim=1)

			cl_loss += cal_loss(u_gnn_embs, u_int_embs)
			cl_loss += cal_loss(u_gnn_embs, u_gaa_embs)	
			cl_loss += cal_loss(u_gnn_embs, u_iaa_embs)	

			cl_loss += cal_loss(i_gnn_embs, i_int_embs)
			cl_loss += cal_loss(i_gnn_embs, i_gaa_embs)
			cl_loss += cal_loss(i_gnn_embs, i_iaa_embs)

		return cl_loss

	def loss(self, out_dict: dict) -> torch.Tensor:
		user_id = out_dict['user_id'].long()
		item_id = out_dict['item_id'].long()
		pos_id = item_id[:, 0]
		neg_id = item_id[:, 1]
		u_embed = out_dict['u_embed'][user_id]
		i_embed = out_dict['i_embed'][item_id]
		pos_embed = i_embed[:, 0]
		neg_embed = i_embed[:, 1]
		pos_scores = torch.sum(u_embed * pos_embed, 1)
		neg_scores = torch.sum(u_embed * neg_embed, 1)
		gnn_embed = out_dict['gnn_embed']
		int_embed = out_dict['int_embed']
		gaa_embed = out_dict['gaa_embed']
		iaa_embed = out_dict['iaa_embed']

		bpr_loss = torch.mean(F.softplus(-(pos_scores - neg_scores)))

		u_embeddings_pre = self.encoder.embedding_dict['user_emb'](user_id)
		pos_embeddings_pre = self.encoder.embedding_dict['item_emb'](pos_id)
		neg_embeddings_pre = self.encoder.embedding_dict['item_emb'](neg_id)
		emb_loss = (u_embeddings_pre.norm(2).pow(2) + pos_embeddings_pre.norm(2).pow(2) + neg_embeddings_pre.norm(
			2).pow(2))

		cen_loss = (self.encoder.embedding_dict['user_intent'].norm(2).pow(2) + self.encoder.embedding_dict['item_intent'].norm(2).pow(2))

		cl_loss = self.cal_ssl_loss(user_id, item_id[:, 0], gnn_embed, int_embed, gaa_embed, iaa_embed)
		return bpr_loss +2.5e-5 * emb_loss + 5e-3 * cen_loss + 1e-1 * cl_loss
