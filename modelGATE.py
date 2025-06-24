import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch import einsum
from torch.autograd import Function

import args

def glorot_init(input_dim, output_dim):
	initial = torch.empty(input_dim, output_dim)
	nn.init.xavier_uniform_(initial)
	return nn.Parameter(initial)

class GraphAttentionLayer(nn.Module):
	def __init__(self, output_dim, device_info):
		super(GraphAttentionLayer, self).__init__()
		self.ln1 = nn.Linear(output_dim, args.hidden2_dim)
		self.ln2 = nn.Linear(output_dim, args.hidden2_dim)
		for m in self.modules():
			self.weights_init(m)

	def weights_init(self, m):
		if isinstance(m, (nn.Linear)):
			torch.nn.init.xavier_uniform_(m.weight.data)
			if m.bias is not None:
				m.bias.data.fill_(0.0)

	def forward(self, feat, adj):
		x = feat
		f1 = self.ln1(x)
		f1_a = einsum("b w h, b w j -> b h j", adj, f1)
		f2 = self.ln2(x)
		f2_a = einsum("b w h, b w j -> b w j", adj, f2)

		logits12 = torch.add(f1_a, f2_a)

		attentions = torch.softmax(logits12, dim=2)

		return attentions

class GraphAttentionLayer_adj1(nn.Module):
	def __init__(self):
		super(GraphAttentionLayer_adj1, self).__init__()

	def forward(self, adj):

		f1 = torch.matmul(adj, adj.permute(0, 2, 1))
		attentions = torch.softmax(f1, dim=2)

		return attentions

class EncoderLayer_adj(nn.Module):
	def __init__(self, input_dim, output_dim, device_info):
		super(EncoderLayer_adj,self).__init__()
		self.weight_a = glorot_init(args.default_node_dim, output_dim)
		self.weight_f = glorot_init(input_dim, output_dim)
		self.attlayer_f = GraphAttentionLayer(output_dim, device_info)
		self.attlayer_a = GraphAttentionLayer_adj1()

	def forward(self, feat, adj):
		H_adj = torch.matmul(adj, self.weight_a)
		self.C_a = self.attlayer_a(H_adj)
		H_a = einsum("b n i, b n h -> b n i", self.C_a , H_adj)

		H = torch.matmul(feat, self.weight_f)
		self.C = self.attlayer_f(H, H_a)
		H = einsum("b n i, b n h -> b n h", self.C, H)

		return H, self.C, self.C_a


class Encoder(nn.Module):
	def __init__(self, device_info):
		super(Encoder, self).__init__()
		self.layer1 = EncoderLayer_adj(args.input_dim, args.hidden1_dim, device_info)
		self.layer2 = EncoderLayer_adj(args.hidden1_dim, args.hidden2_dim, device_info)

	def forward(self, feat, adj):
		H, C1, Ca1 = self.layer1(feat, adj)
		H, C2, Ca2 = self.layer2(H, adj)
		return H, C1, C2, Ca1, Ca2


class DecoderLayer(nn.Module):
	def __init__(self):
		super(DecoderLayer, self).__init__()

	def forward(self, enc_output, weight):
		H = torch.matmul(enc_output, weight.t())
		return H

class DecoderX1(nn.Module):
	def __init__(self):
		super(DecoderX1, self).__init__()
		self.layer1 = DecoderLayer()
		self.layer2 = DecoderLayer()

	def forward(self, H, enc_weight1, enc_weight2):
		tmp = self.layer1(H, enc_weight2)
		tmp = torch.relu(tmp)
		X_ = self.layer2(tmp, enc_weight1)
		return X_, tmp

class DecoderA1(nn.Module):
	def __init__(self):
		super(DecoderA1, self).__init__()
		self.layer1 = DecoderLayer()
		self.layer2 = DecoderLayer()
		self.norm2 = nn.LayerNorm(args.default_node_dim)
		self.norm1 = nn.LayerNorm(args.default_node_dim)

	def forward(self, H, Dec1_out, enc_a_weight1, enc_a_weight2):
		tmp2 = self.layer1(H, enc_a_weight2)
		tmp1 = self.layer2(Dec1_out, enc_a_weight1)
		tmp2 = self.norm2(tmp2)
		tmp1 = self.norm1(tmp1)
		A_pred = torch.matmul(tmp2, tmp1.permute(0,2,1))
		A_pred = torch.sigmoid(A_pred)
		return A_pred

class GATE(nn.Module):
	def __init__(self, device_info):
		super(GATE,self).__init__()
		self.Encoder = Encoder(device_info)
		self.DecoderX1 = DecoderX1()
		self.DecoderA = DecoderA1()

	def forward(self, X, adj):
		H = X

		H, C1, C2, C1a, C2a = self.Encoder(H, adj)
		self.H = H

		X_, Dec1_out = self.DecoderX1(H, self.Encoder.layer1.weight_f, self.Encoder.layer2.weight_f)

		A_pred = self.DecoderA(H, Dec1_out, self.Encoder.layer1.weight_a, self.Encoder.layer2.weight_a)


		return X_, A_pred, H, self.Encoder.layer1.C, self.Encoder.layer2.C
