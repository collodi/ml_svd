import torch

def batch_svd(X):
	matrices = [torch.svd(x) for x in X]
	U = torch.stack([u for u, _, _ in matrices], dim=0)
	E = torch.stack([e.diag() for _, e, _ in matrices], dim=0)
	V = torch.stack([v for _, _, v in matrices], dim=0)
	return U, E, V

def svdcols(U, V):
	return torch.stack([
		torch.cat((U[:, i], V[:, j]), dim=0)
		for i in range(U.size(1))
		for j in range(V.size(1))
	], dim=0)

def batch_svdcols(U, V):
	 return torch.cat([
		svdcols(u, v) for u, v in zip(U, V)
	], dim=0)

def batch_unsvd(U, E, V):
	return U.bmm(E).bmm(V.transpose(-1, -2))
