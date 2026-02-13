import os
import numpy as np
import h5py
import scipy.io as sio
import matplotlib.pyplot as plt


def load_bitwidth(base='results/bitwidth'):
	h5 = base + '.h5'
	mat = base + '.mat'
	npz = base + '.mat.npz'
	if os.path.exists(h5):
		with h5py.File(h5, 'r') as f:
			epochs = np.array(f['epochs']).astype(int)
			bw_pct = np.array(f['bw_pct']).astype(float)
			ebops = np.array(f['ebops']).astype(float) if 'ebops' in f else None
		return epochs, bw_pct, ebops
	if os.path.exists(mat):
		data = sio.loadmat(mat)
		epochs = np.array(data.get('epoch', [])).astype(int).flatten()
		bw_pct = np.array(data.get('bw_pct', [])).astype(float)
		ebops = np.array(data.get('ebops', [])).astype(float) if 'ebops' in data else None
		return epochs, bw_pct, ebops
	if os.path.exists(npz):
		data = np.load(npz)
		epochs = np.array(data['epoch']).astype(int).flatten() if 'epoch' in data else np.arange(data['bw_pct'].shape[0])
		bw_pct = np.array(data['bw_pct']).astype(float)
		ebops = np.array(data['ebops']).astype(float) if 'ebops' in data else None
		return epochs, bw_pct, ebops
	return np.array([]), np.empty((0,)), np.array([])


def load_acc_ebops(base='results/acc_ebops'):
	h5 = base + '.h5'
	mat = base + '.mat'
	npz = base + '.mat.npz'
	if os.path.exists(h5):
		with h5py.File(h5, 'r') as f:
			acc = np.array(f['acc']).astype(float)
			ebops = np.array(f['ebops']).astype(float)
		return acc, ebops
	if os.path.exists(mat):
		data = sio.loadmat(mat)
		acc = np.array(data.get('acc_values', data.get('acc_values', []))).astype(float).flatten()
		ebops = np.array(data.get('ebops_values', [])).astype(float).flatten()
		return acc, ebops
	if os.path.exists(npz):
		data = np.load(npz)
		acc = np.array(data.get('acc_values', []))
		ebops = np.array(data.get('ebops_values', []))
		return acc, ebops
	return np.array([]), np.array([])


def load_lr_beta(base='results/lr_beta'):
	h5 = base + '.h5'
	mat = base + '.mat'
	npz = base + '.mat.npz'
	if os.path.exists(h5):
		with h5py.File(h5, 'r') as f:
			epochs = np.array(f['epochs']).astype(int)
			lr = np.array(f['lr']).astype(float)
			beta = np.array(f['beta']).astype(float)
		return epochs, lr, beta
	if os.path.exists(mat):
		data = sio.loadmat(mat)
		epochs = np.array(data.get('epoch', [])).astype(int).flatten()
		lr = np.array(data.get('lr', [])).astype(float).flatten()
		beta = np.array(data.get('beta', [])).astype(float).flatten()
		return epochs, lr, beta
	if os.path.exists(npz):
		data = np.load(npz)
		epochs = np.array(data.get('epoch', np.arange(data['lr'].shape[0]))).astype(int).flatten()
		lr = np.array(data.get('lr', [])).astype(float)
		beta = np.array(data.get('beta', [])).astype(float)
		return epochs, lr, beta
	return np.array([]), np.array([]), np.array([])


def pareto_mask(acc, ebops):
	# Efficient Pareto: sort by increasing ebops, break ties by decreasing acc,
	# then scan keeping the best accuracy seen so far. Points with acc > best
	# are Pareto-optimal (maximize acc, minimize ebops).
	acc = np.asarray(acc)
	ebops = np.asarray(ebops)
	n = len(acc)
	mask = np.zeros(n, dtype=bool)
	if n == 0:
		return mask

	# Only consider finite pairs; non-finite entries are not Pareto
	finite = np.isfinite(acc) & np.isfinite(ebops)
	if not np.any(finite):
		return mask

	idx = np.nonzero(finite)[0]
	# sort indices by ebops asc, acc desc
	order = np.lexsort((-acc[idx], ebops[idx]))
	sorted_idx = idx[order]

	best_acc = -np.inf
	for i in sorted_idx:
		a = acc[i]
		if a > best_acc:
			mask[i] = True
			best_acc = a
	return mask


def plot_all():
	os.makedirs('results', exist_ok=True)

	# Bitwidth
	epochs_bw, bw_pct, ebops_bw = load_bitwidth()
	if bw_pct.size > 0:
		bits = bw_pct.shape[1]
		plt.figure(figsize=(12, 6))
		for b in range(bits):
			plt.plot(epochs_bw, bw_pct[:, b], label=f'{b}-bit')
		plt.xlabel('Epoch')
		plt.ylabel('Percentage (%)')
		plt.title('Bitwidth percentages (all epochs)')
		plt.legend(loc='upper right', ncol=2, fontsize=8)
		plt.grid(True)
		plt.tight_layout()
		plt.savefig('results/bitwidth_whole.png', dpi=150)
		plt.close()

		if ebops_bw is not None and ebops_bw.size > 0:
			plt.figure(figsize=(12, 3))
			plt.plot(epochs_bw, ebops_bw, color='tab:red')
			plt.xlabel('Epoch')
			plt.ylabel('EBOPs')
			plt.title('EBOPs (all epochs)')
			plt.grid(True)
			plt.tight_layout()
			plt.savefig('results/ebops_whole.png', dpi=150)
			plt.close()

	# Acc vs EBOPs
	acc, ebops = load_acc_ebops()
	if acc.size > 0 and ebops.size > 0:
		plt.figure(figsize=(8, 8))
		plt.scatter(ebops, acc, s=18, alpha=0.6)
		try:
			mask = pareto_mask(acc.tolist(), ebops.tolist())
			plt.scatter(ebops[mask], acc[mask], color='red', s=22, label='Pareto')
			plt.legend()
		except Exception:
			pass
		plt.xlabel('EBOPs')
		plt.ylabel('Accuracy')
		plt.title('Accuracy vs EBOPs (all epochs)')
		plt.grid(True)
		plt.tight_layout()
		plt.savefig('results/acc_ebops_whole.png', dpi=150)
		plt.close()

	# LR / Beta
	epochs_lr, lr, beta = load_lr_beta()
	if lr.size > 0 or beta.size > 0:
		plt.figure(figsize=(10, 6))
		ax1 = plt.gca()
		if lr.size > 0:
			ax1.plot(epochs_lr, lr, color='tab:blue', label='lr')
		ax1.set_yscale('log')
		ax1.set_xlabel('Epoch')
		ax1.set_ylabel('LR (log)')
		ax2 = ax1.twinx()
		if beta.size > 0:
			ax2.plot(epochs_lr, beta, color='tab:orange', label='beta')
		ax2.set_yscale('log')
		ax2.set_ylabel('Beta (log)')
		lines, labels = ax1.get_legend_handles_labels()
		lines2, labels2 = ax2.get_legend_handles_labels()
		plt.title('LR and Beta (all epochs)')
		ax1.legend(lines + lines2, labels + labels2, loc='best')
		plt.grid(True)
		plt.tight_layout()
		plt.savefig('results/lr_beta_whole.png', dpi=150)
		plt.close()


if __name__ == '__main__':
	plot_all()
