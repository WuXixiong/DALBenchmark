import copy
import math
import numpy as np
from torch.autograd import Variable
from .almethod import ALMethod
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
import torch.nn as nn

class AlphaMixSampling(ALMethod):
	def __init__(self, args, models, unlabeled_dst, U_index, I_index, **kwargs):
		super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
		self.I_index = I_index
		self.labeled_set = torch.utils.data.Subset(unlabeled_dst, I_index)
		self.fc2 = None
		# subset settings
		subset_idx = np.random.choice(len(self.U_index), size=(min(self.args.subset, len(self.U_index)),), replace=False)
		self.U_index_sub = np.array(self.U_index)[subset_idx]

	def select(self):
		n = self.args.n_query
		self.models['backbone'].eval()
		unlabeled_subset = torch.utils.data.Subset(self.unlabeled_dst, self.U_index_sub)
		selection_loader = torch.utils.data.DataLoader(unlabeled_subset, batch_size=self.args.n_query, num_workers=self.args.workers)
		scores = np.array([])
		batch_num = len(selection_loader)
		ulb_probs = None
		org_ulb_embedding = None

		# get the last layer of the backbone
		if self.args.textset:
			old_fc = self.models['backbone'].classifier
		else:
			old_fc = self.models['backbone'].get_last_layer()
		out_features = old_fc.out_features
		in_features  = old_fc.in_features
		# create a new linear layer with the same weights and biases
		self.fc2 = nn.Linear(in_features, out_features, bias=True).to(self.args.device)
		with torch.no_grad():
			self.fc2.weight.copy_(old_fc.weight)
			self.fc2.bias.copy_(old_fc.bias)
		# not to update the weights of the new linear layer
		self.fc2.weight.requires_grad = False
		self.fc2.bias.requires_grad = False

		print("| Calculating uncertainty of Unlabeled set")
		for i, data in enumerate(selection_loader):
			if i % self.args.print_freq == 0:
				print("| Selecting for batch [%3d/%3d]" % (i + 1, batch_num))
				with torch.no_grad():
					if self.args.textset:
					# Extract input_ids, attention_mask, and labels from the dictionary
						input_ids = data['input_ids'].to(self.args.device)
						attention_mask = data['attention_mask'].to(self.args.device)
						outputs = self.models['backbone'](input_ids=input_ids, attention_mask=attention_mask)
						preds = outputs.logits
						hidden_states = outputs.hidden_states
						last_hidden_state = hidden_states[-1]
						embedding = last_hidden_state[:, 0, :]
					else:
						inputs = data[0].to(self.args.device)
						preds, embedding = self.models['backbone'](inputs)
					probs = torch.nn.functional.softmax(preds, dim=1)

					# If this is the first batch, initialize all_probs and all_embeddings
					if ulb_probs is None:
						ulb_probs = torch.empty((0, probs.size(1))).to(self.args.device)
						org_ulb_embedding = torch.empty((0, embedding.size(1))).to(self.args.device)
        
        			# Concatenate probs and embeddings from this batch with the previously accumulated ones
					ulb_probs = torch.cat((ulb_probs, probs), dim=0)
					org_ulb_embedding = torch.cat((org_ulb_embedding, embedding), dim=0)

		probs_sorted, probs_sort_idxs = ulb_probs.sort(descending=True)
		pred_1 = probs_sort_idxs[:, 0]

		labeled_loader = torch.utils.data.DataLoader(self.labeled_set, batch_size=self.args.test_batch_size, num_workers=self.args.workers)
		batch_num = len(labeled_loader)
		lb_probs = None
		org_lb_embedding = None
		print("| Calculating uncertainty of labeled set")
		for i, data in enumerate(labeled_loader):
			if i % self.args.print_freq == 0:
				print("| Selecting for batch [%3d/%3d]" % (i + 1, batch_num))
				with torch.no_grad():
					if self.args.textset:
					# Extract input_ids, attention_mask, and labels from the dictionary
						input_ids = data['input_ids'].to(self.args.device)
						attention_mask = data['attention_mask'].to(self.args.device)
						outputs = self.models['backbone'](input_ids=input_ids, attention_mask=attention_mask)
						hidden_states = outputs.hidden_states
						last_hidden_state = hidden_states[-1]
						embedding = last_hidden_state[:, 0, :]
					else:
						inputs = data[0].to(self.args.device)
						preds, embedding = self.models['backbone'](inputs)
					probs = torch.nn.functional.softmax(preds, dim=1)

					# If this is the first batch, initialize all_probs and all_embeddings
					if lb_probs is None:
						lb_probs = torch.empty((0, probs.size(1))).to(self.args.device)
						org_lb_embedding = torch.empty((0, embedding.size(1))).to(self.args.device)
        
        			# Concatenate probs and embeddings from this batch with the previously accumulated ones
					lb_probs = torch.cat((lb_probs, probs), dim=0)
					org_lb_embedding = torch.cat((org_lb_embedding, embedding), dim=0)

		ulb_embedding = org_ulb_embedding
		lb_embedding = org_lb_embedding

		unlabeled_size = ulb_embedding.size(0)
		embedding_size = ulb_embedding.size(1)

		min_alphas = torch.ones((unlabeled_size, embedding_size), dtype=torch.float)
		candidate = torch.zeros(unlabeled_size, dtype=torch.bool)

		if self.args.alpha_closed_form_approx:
			var_emb = Variable(ulb_embedding, requires_grad=True).to(self.args.device)
			out = self.fc2(var_emb)
			loss = F.cross_entropy(out, pred_1.to(self.args.device))
			grads = torch.autograd.grad(loss, var_emb)[0].data.cpu()
			del loss, var_emb, out
		else:
			grads = None

		alpha_cap = 0.
		while alpha_cap < 1.0:
			alpha_cap += self.args.alpha_cap

			selected_targets = [self.labeled_set.dataset.targets[i] for i in self.I_index]
			tmp_pred_change, tmp_min_alphas = \
				self.find_candidate_set(
					lb_embedding, ulb_embedding, pred_1, ulb_probs, alpha_cap=alpha_cap,
					Y=selected_targets, # targets in labelled data
					grads=grads)
			
			min_alphas = min_alphas.cpu()
			tmp_min_alphas = tmp_min_alphas.cpu()
			is_changed = min_alphas.norm(dim=1) >= tmp_min_alphas.norm(dim=1)

			min_alphas[is_changed] = tmp_min_alphas[is_changed]
			candidate += tmp_pred_change

			print('With alpha_cap set to %f, number of inconsistencies: %d' % (alpha_cap, int(tmp_pred_change.sum().item())))

			if candidate.sum() > n:
				break

		if candidate.sum() > 0:
			print('Number of inconsistencies: %d' % (int(candidate.sum().item())))

			print('alpha_mean_mean: %f' % min_alphas[candidate].mean(dim=1).mean().item())
			print('alpha_std_mean: %f' % min_alphas[candidate].mean(dim=1).std().item())
			print('alpha_mean_std %f' % min_alphas[candidate].std(dim=1).mean().item())

			c_alpha = F.normalize(org_ulb_embedding[candidate].view(candidate.sum(), -1), p=2, dim=1).detach()
			c_alpha = c_alpha.cpu()

			selected_idxs = self.sample(min(n, candidate.sum().item()), feats=c_alpha)
			selected_idxs = self.U_index_sub[selected_idxs]

		else:
			selected_idxs = np.array([], dtype=np.int64)

		if len(selected_idxs) < n:
			remained = n - len(selected_idxs)
			remained_index = list(set(self.U_index) - set(selected_idxs))
			selected_idxs = np.concatenate([selected_idxs, np.random.choice(remained_index, remained, replace=False)])
			print('picked %d samples from RandomSampling.' % (remained))
		
		scores = list(np.ones(len(selected_idxs)))  # equally assign 1 (meaningless)

		return selected_idxs, scores

	def find_alpha(self):
		assert self.args.alpha_num_mix <= self.args.n_label - (
			0 if self.args.alpha_use_highest_class_mix else 1), 'c_num_mix should not be greater than number of classes'

		idxs_unlabeled = np.arange(self.n_pool)[self.idxs_lb]

		ulb_probs, ulb_embedding = self.predict_prob_embed(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])

		probs_sorted, probs_sort_idxs = ulb_probs.sort(descending=True)
		pred_1 = probs_sort_idxs[:, 0]
		gt_lables = self.Y[idxs_unlabeled]
		preds = pred_1 == gt_lables

		ulb_embedding = ulb_embedding[preds]
		ulb_probs = ulb_probs[preds]
		pred_1 = pred_1[preds]

		lb_embedding = self.get_embedding(self.X[self.idxs_lb], self.Y[self.idxs_lb])

		alpha_cap = 0.
		for i in range(self.args.alpha_alpha_scales if self.args.alpha_alpha_growth_method == 'exponential' else int(pow(2, self.args.alpha_alpha_scales) - 1)):
			if self.args.alpha_alpha_growth_method == 'exponential':
				alpha_cap = self.args.alpha_cap / (pow(2, self.args.alpha_alpha_scales - i - 1))
			else:
				alpha_cap += self.args.alpha_cap / (pow(2, self.args.alpha_alpha_scales - 1))

			tmp_pred_change, tmp_pred_change_sum, tmp_min_alphas, tmp_min_added_feats, tmp_cf_probs, _, tmp_min_mixing_labels, tmp_min_cf_feats = \
				self.find_candidate_set(
					lb_embedding, ulb_embedding, pred_1, ulb_probs, probs_sort_idxs, alpha_cap=alpha_cap,
					Y=self.Y[self.idxs_lb])

			if tmp_pred_change.sum() > 0:
				print('selected alpha_max %f' % alpha_cap)
				self.writer.add_scalar('stats/alpha_cap', alpha_cap, self.query_count)

				return alpha_cap

		print('no labelled sample change!!!')
		return 0.5

	def find_candidate_set(self, lb_embedding, ulb_embedding, pred_1, ulb_probs, alpha_cap, Y, grads):

		unlabeled_size = ulb_embedding.size(0)
		embedding_size = ulb_embedding.size(1)

		min_alphas = torch.ones((unlabeled_size, embedding_size), dtype=torch.float)
		pred_change = torch.zeros(unlabeled_size, dtype=torch.bool)

		if self.args.alpha_closed_form_approx:
			alpha_cap /= math.sqrt(embedding_size)
			grads = grads.to(self.args.device)
			
		for i in range(int(self.args.n_class)):
			emb = lb_embedding[Y == i]
			if emb.size(0) == 0:
				emb = lb_embedding
			anchor_i = emb.mean(dim=0).view(1, -1).repeat(unlabeled_size, 1)

			if self.args.alpha_closed_form_approx:
				embed_i, ulb_embed = anchor_i.to(self.args.device), ulb_embedding.to(self.args.device)
				alpha = self.calculate_optimum_alpha(alpha_cap, embed_i, ulb_embed, grads)

				embedding_mix = (1 - alpha) * ulb_embed + alpha * embed_i
				out = self.fc2(embedding_mix)
				out = out.detach().cpu()
				alpha = alpha.cpu()

				pc = out.argmax(dim=1) != pred_1
			else:
				alpha = self.generate_alpha(unlabeled_size, embedding_size, alpha_cap)
				if self.args.alpha_opt:
					alpha, pc = self.learn_alpha(ulb_embedding, pred_1, anchor_i, alpha, alpha_cap,
												 log_prefix=str(i))
				else:
					ulb_embedding = ulb_embedding.to(self.args.device)
					anchor_i = anchor_i.to(self.args.device)
					alpha = alpha.to(self.args.device)
					embedding_mix = (1 - alpha) * ulb_embedding + alpha * anchor_i
					out = self.fc2(embedding_mix.to(self.args.device))
					out = out.detach().cpu()
					pred_1 = pred_1.detach().cpu()

					pc = out.argmax(dim=1) != pred_1

			torch.cuda.empty_cache()

			alpha[~pc] = 1.
			pred_change[pc] = True
			min_alphas = min_alphas.to(self.args.device)
			alpha = alpha.to(self.args.device)
			is_min = min_alphas.norm(dim=1) > alpha.norm(dim=1)
			min_alphas[is_min] = alpha[is_min]
			
		return pred_change, min_alphas

	def calculate_optimum_alpha(self, eps, lb_embedding, ulb_embedding, ulb_grads):
		z = (lb_embedding - ulb_embedding) #* ulb_grads
		alpha = (eps * z.norm(dim=1) / ulb_grads.norm(dim=1)).unsqueeze(dim=1).repeat(1, z.size(1)) * ulb_grads / (z + 1e-8)

		return alpha

	def sample(self, n, feats):
		feats = feats.numpy()
		cluster_learner = KMeans(n_clusters=n)
		cluster_learner.fit(feats)

		cluster_idxs = cluster_learner.predict(feats)
		centers = cluster_learner.cluster_centers_[cluster_idxs]
		dis = (feats - centers) ** 2
		dis = dis.sum(axis=1)
		return np.array(
			[np.arange(feats.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(n) if
			 (cluster_idxs == i).sum() > 0])

	def retrieve_anchor(self, embeddings, count):
		return embeddings.mean(dim=0).view(1, -1).repeat(count, 1)

	def generate_alpha(self, size, embedding_size, alpha_cap):
		alpha = torch.normal(
			mean=alpha_cap / 2.0,
			std=alpha_cap / 2.0,
			size=(size, embedding_size))

		alpha[torch.isnan(alpha)] = 1

		return self.clamp_alpha(alpha, alpha_cap)

	def clamp_alpha(self, alpha, alpha_cap):
		return torch.clamp(alpha, min=1e-8, max=alpha_cap)

	def learn_alpha(self, org_embed, labels, anchor_embed, alpha, alpha_cap, log_prefix=''):
		labels = labels.to(self.args.device)
		min_alpha = torch.ones(alpha.size(), dtype=torch.float)
		pred_changed = torch.zeros(labels.size(0), dtype=torch.bool)

		loss_func = torch.nn.CrossEntropyLoss(reduction='none')

		# self.model.clf.eval()
		self.models['backbone'].eval()

		for i in range(self.args.alpha_learning_iters):
			tot_nrm, tot_loss, tot_clf_loss = 0., 0., 0.
			for b in range(math.ceil(float(alpha.size(0)) / self.args.alpha_learn_batch_size)):
				self.models['backbone'].zero_grad()
				start_idx = b * self.args.alpha_learn_batch_size
				end_idx = min((b + 1) * self.args.alpha_learn_batch_size, alpha.size(0))

				l = alpha[start_idx:end_idx]
				l = torch.autograd.Variable(l.to(self.args.device), requires_grad=True)
				opt = torch.optim.Adam([l], lr=self.args.alpha_learning_rate / (1. if i < self.args.alpha_learning_iters * 2 / 3 else 10.))
				e = org_embed[start_idx:end_idx].to(self.args.device)
				c_e = anchor_embed[start_idx:end_idx].to(self.args.device)
				embedding_mix = (1 - l) * e + l * c_e

				out = self.fc2(embedding_mix)

				label_change = out.argmax(dim=1) != labels[start_idx:end_idx]

				tmp_pc = torch.zeros(labels.size(0), dtype=torch.bool).to(self.args.device)
				tmp_pc[start_idx:end_idx] = label_change
				pred_changed[start_idx:end_idx] += tmp_pc[start_idx:end_idx].detach().cpu()

				tmp_pc[start_idx:end_idx] = tmp_pc[start_idx:end_idx] * (l.norm(dim=1) < min_alpha[start_idx:end_idx].norm(dim=1).to(self.args.device))
				min_alpha[tmp_pc] = l[tmp_pc[start_idx:end_idx]].detach().cpu()

				clf_loss = loss_func(out, labels[start_idx:end_idx].to(self.args.device))

				l2_nrm = torch.norm(l, dim=1)

				clf_loss *= -1

				loss = self.args.alpha_clf_coef * clf_loss + self.args.alpha_l2_coef * l2_nrm
				loss.sum().backward(retain_graph=True)
				opt.step()

				l = self.clamp_alpha(l, alpha_cap)

				alpha[start_idx:end_idx] = l.detach().cpu()

				tot_clf_loss += clf_loss.mean().item() * l.size(0)
				tot_loss += loss.mean().item() * l.size(0)
				tot_nrm += l2_nrm.mean().item() * l.size(0)

				del l, e, c_e, embedding_mix
				torch.cuda.empty_cache()

		return min_alpha.cpu(), pred_changed.cpu()

