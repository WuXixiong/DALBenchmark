import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .almethod import ALMethod
import nets
from .combinedataset import PairedDataset

import torch.optim as optim
from tqdm import tqdm
'''
This implementation is with reference of https://github.com/cjshui/WAAL.
Please cite the original paper if you plan to use this method.
@inproceedings{shui2020deep,
  title={Deep active learning: Unified and principled method for query and training},
  author={Shui, Changjian and Zhou, Fan and Gagn{\'e}, Christian and Wang, Boyu},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={1308--1318},
  year={2020},
  organization={PMLR}
}
'''
class WAAL(ALMethod):
	def __init__(self, args, models, unlabeled_dst, U_index, **kwargs):
		super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
		self.args.waal_selection = 10

	def select(self):
		unlabeled_data = self.unlabeled_dst
		unlabeled_data = self.U_index

		# unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
		probs = self.predict_prob(unlabeled_data)
		uncertainly_score = 0.5* self.L2_upper(probs) + 0.5* self.L1_upper(probs)

		# prediction output discriminative score
		dis_score = self.pred_dis_score_waal(unlabeled_data)

		# computing the decision score
		total_score = uncertainly_score - self.args.waal_selection * dis_score
		# b = total_score.sort()[1][:n]
		sorted_indices = np.argsort(total_score)
		Q_index = [self.U_index[idx] for idx in sorted_indices]

		return Q_index, total_score

	def L2_upper(self, probas):
		value = torch.norm(torch.log(probas))
		return value


	def L1_upper(self, probas):
		value = torch.sum(-1*torch.log(probas))
		return value

	def pred_dis_score_waal(self, data):
		selection_loader = torch.utils.data.DataLoader(self.unlabeled_set, batch_size=self.args.test_batch_size, num_workers=self.args.workers)
		# self.net_fea = nets.vae.fea()
		self.net_fea = nets.vae.FEA()
		self.net_dis = nets.vae.Discriminator(z_dim=512)

		self.net_fea.cuda()
		self.net_dis.cuda()

		self.net_fea.eval()
		self.net_dis.eval()

		scores = torch.zeros(len(data))

		with torch.no_grad():
			# print(next(iter(selection_loader)))
			for i, data in enumerate(selection_loader):
				# i, data = i.cuda(), data.cuda()
				inputs = data[0].to(self.args.device)
				latent = self.net_fea(inputs)
				out = self.net_dis(latent[0]).cpu()
				scores[i] = out.view(-1)

		return scores
	
	def predict_prob(self, data):
		self.models['backbone'].eval()
		selection_loader = torch.utils.data.DataLoader(self.unlabeled_set, batch_size=self.args.test_batch_size, num_workers=self.args.workers)

		preds = np.array([])
		batch_num = len(selection_loader)
		print("| Calculating uncertainty of Unlabeled set")
		for i, data in enumerate(selection_loader):
			inputs = data[0].to(self.args.device)
			if i % self.args.print_freq == 0:
				print("| Selecting for batch [%3d/%3d]" % (i + 1, batch_num))

			with torch.no_grad():
				pred, _ = self.models['backbone'](inputs)
				pred = pred.cpu().numpy()

				preds = np.append(preds, pred)

		return torch.from_numpy(preds)

	def train_waal(self, label_x, _, unlabel_x, alpha = 1e-3):
		n_epoch = self.args.epochs


		# setting three optimizers
		opt_fea = optim.Adam(self.net_fea.parameters())
		opt_dis = optim.Adam(self.net_dis.parameters())
		

		# computing the unbalancing ratio, a value betwwen [0,1]
		#gamma_ratio = X_labeled.shape[0]/X_unlabeled.shape[0]
		gamma_ratio = 1
		combined_dataset = PairedDataset(self.train_dst, self.unlabeled_dst)
		combined_dataloader = DataLoader(combined_dataset, batch_size=self.args.test_batch_size, num_workers=self.args.workers)
		for epoch in tqdm(range(1, n_epoch+1), ncols=100):
			# setting the training mode in the beginning of EACH epoch
			# (since we need to compute the training accuracy during the epoch, optional)
			self.net_fea.train()
			self.net_dis.train()

			for _, label_x, _, unlabel_x, _ in combined_dataloader:

				label_x, label_y = label_x.to(self.device), label_y.to(self.device)
				unlabel_x = unlabel_x.to(self.device)

				# training feature extractor and predictor
				self.set_requires_grad(self.fea,requires_grad=True)
				self.set_requires_grad(self.clf,requires_grad=True)
				self.set_requires_grad(self.dis,requires_grad=False)

				#print(label_x.shape)
				lb_z   = self.fea(label_x)
				unlb_z = self.fea(unlabel_x)

				opt_fea.zero_grad()

				lb_out, _ = self.clf(lb_z)

				# prediction loss (deafult we use F.cross_entropy)
				pred_loss = torch.mean(F.cross_entropy(lb_out,label_y))

				# Wasserstein loss (unbalanced loss, used the redundant trick)
				wassertein_distance = self.dis(unlb_z).mean() - gamma_ratio * self.dis(lb_z).mean()

				with torch.no_grad():

					lb_z = self.fea(label_x)
					unlb_z = self.fea(unlabel_x)

				gp = self.gradient_penalty(self.dis, unlb_z, lb_z)

				loss = pred_loss + alpha * wassertein_distance + alpha * gp * 5
				# for CIFAR10 the gradient penality is 5
				# for SVHN the gradient penality is 2

				loss.backward()
				opt_fea.step()


				# Then the second step, training discriminator

				self.set_requires_grad(self.fea, requires_grad=False)
				self.set_requires_grad(self.clf, requires_grad=False)
				self.set_requires_grad(self.dis, requires_grad=True)


				with torch.no_grad():

					lb_z = self.fea(label_x)
					unlb_z = self.fea(unlabel_x)


				for _ in range(1):

					# gradient ascent for multiple times like GANS training

					gp = self.gradient_penalty(self.dis, unlb_z, lb_z)

					wassertein_distance = self.dis(unlb_z).mean() - gamma_ratio * self.dis(lb_z).mean()

					dis_loss = -1 * alpha * wassertein_distance - alpha * gp * 2

					opt_dis.zero_grad()
					dis_loss.backward()
					opt_dis.step()