import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys, logging

def parse_index_file(filename):
	"""Parse index file."""
	index = []
	for line in open(filename):
			index.append(int(line.strip()))
	return index

def sample_mask(idx, l):
	"""Create mask."""
	mask = np.zeros(l)
	mask[idx] = 1
	return np.array(mask, dtype=np.bool)

def sparse_to_tuple(sparse_mx):
	"""Convert sparse matrix to tuple representation."""
	def to_tuple(mx):
		if not sp.isspmatrix_coo(mx):
			mx = sp.csr_matrix(mx)
			mx = mx.tocoo()
		coords = np.vstack((mx.row, mx.col)).transpose()
		values = mx.data
		shape = mx.shape
		return coords, values, shape

	if isinstance(sparse_mx, list):
		for i in range(len(sparse_mx)):
			sparse_mx[i] = to_tuple(sparse_mx[i])
	else:
		sparse_mx = to_tuple(sparse_mx)

	return sparse_mx

def preprocess_features(features):
	"""Row-normalize feature matrix and convert to tuple representation"""

	# Do not row-normalize...
	# TODO: col-normalize?
	#rowsum = np.array(features.sum(1))
	#r_inv = np.power(rowsum, -1).flatten()
	#r_inv[np.isinf(r_inv)] = 0.
	#r_mat_inv = sp.diags(r_inv)
	#features = r_mat_inv.dot(features)
	return sparse_to_tuple(features)

def normalize_adj(adj):
	"""Symmetrically normalize adjacency matrix."""
	if not sp.isspmatrix_coo(adj):
		adj = sp.csr_matrix(adj)
		adj = adj.tocoo()
	rowsum = np.array(adj.sum(1))
	d_inv_sqrt = np.power(rowsum, -0.5).flatten()
	d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
	d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
	return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
	"""Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
	adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
	return adj_normalized

def chebyshev_polynomials(adj, k):
	"""Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
	print("Calculating Chebyshev polynomials up to order {}...".format(k))

	adj_normalized = normalize_adj(adj)
	laplacian = sp.eye(adj.shape[0]) - adj_normalized
	largest_eigval, _ = eigsh(laplacian, 1, which='LM')
	scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

	t_k = list()
	t_k.append(sp.eye(adj.shape[0]))
	t_k.append(scaled_laplacian)

	def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
		s_lap = sp.csr_matrix(scaled_lap, copy=True)
		return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

	for i in range(2, k+1):
		t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

	return sparse_to_tuple(t_k)

def delete_from_csr(mat, row_indices=[], col_indices=[]):
		"""
		Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``) from the CSR sparse matrix ``mat``.
		WARNING: Indices of altered axes are reset in the returned matrix
		"""
		#if not isinstance(mat, sp.csr_matrix):
		#		raise ValueError("works only for CSR format -- use .tocsr() first")

		row_mask = np.ones(mat.shape[0], dtype=np.bool)
		col_mask = np.ones(mat.shape[1], dtype=np.bool)
		row_mask[row_indices] = False
		col_mask[col_indices] = False
		return mat[row_mask][:,col_mask]

def zero_rows(M, rows):
	diag = sp.eye(M.shape[0]).tolil()
	for r in rows:
		diag[r, r] = 0
	return diag.dot(M)

def zero_columns(M, columns):
	diag = sp.eye(M.shape[1]).tolil()
	for c in columns:
		diag[c, c] = 0
	return diag.dot(M)
