import numpy as np
import torch
from .almethod import ALMethod
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad

from tqdm import tqdm

# from utils import *
import nets
# from utils import *
# from load_split_data import *
# from arguments import parser

'''
This implementation is with reference of https://github.com/sinhasam/vaal.
You need to write task-specific VAE in nets.py if you plan to apply this method in new task.
Please cite the original paper if you use this method.
@inproceedings{sinha2019variational,
  title={Variational adversarial active learning},
  author={Sinha, Samarth and Ebrahimi, Sayna and Darrell, Trevor},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5972--5981},
  year={2019}
}
'''

class VAAL(ALMethod):
    def __init__(self, args, models, train_dst, unlabeled_dst, U_index, **kwargs):
        # 在这里处理额外的参数，比如 train_dst
        self.train_dst = train_dst
        # 调用父类构造函数时不包括 train_dst 和其他不需要的参数
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
            # backbone = nets.__dict__[model](args.channel, args.num_IN_class, args.im_size).to(args.device)
        self.net_vae = nets.vae.VAE()
        self.net_dis = nets.vae.Discriminator(z_dim=32)

    def run(self):
        self.train_vaal()
        scores = self.pred_dis_score_vaal(self.unlabeled_dst)
        # print("Length of unlabeled_dst:", len(self.unlabeled_dst)) # HERE IS 50000
        # selection_result = np.argsort(scores)[:self.args.n_query]
        selection_result = np.argsort(-scores)[:self.args.n_query] # descending
        return selection_result, scores

    def select(self, **kwargs):
        selected_indices, scores = self.run()
        # print("Length of U_index:", len(self.U_index)) # HERE IS 49600
        # print("Selected indices:", selected_indices) # HERE EXISTS SOME INDICES LARGER THAN 49600
        selected_indices = selected_indices[selected_indices < len(self.U_index)] # TEMP SOLUTION
        Q_index = [self.U_index[idx] for idx in selected_indices]

        return Q_index, scores

    def train_vaal(self, total_epoch=1,num_vae_steps=1, beta=1, adv_param=1): # NEED MORE TRAINING EPOCHES
        n_epoch = total_epoch
        num_vae_steps=num_vae_steps
        beta=beta
        adv_param=adv_param
        # args = parser.parse_args()
        opt_vae = torch.optim.Adam(self.net_vae.parameters())
        opt_dis = torch.optim.Adam(self.net_dis.parameters())
        
        # dim = self.train_dst.shape[1:]
        self.net_vae = self.net_vae.cuda()
        self.net_dis = self.net_dis.cuda()

		#labeled and unlabeled data
        combined_dataset = PairedDataset(self.train_dst, self.unlabeled_dst)
        combined_dataloader = DataLoader(combined_dataset, batch_size=self.args.test_batch_size, num_workers=self.args.workers)

        print("Start training the VAE_DIS net...")
        for epoch in tqdm(range(n_epoch)):

            self.net_vae.train()
            self.net_dis.train()

            for _, label_x, _, unlabel_x, _ in combined_dataloader:
                label_x = label_x.to("cuda")
                unlabel_x = unlabel_x.to("cuda")

				# vae
                for count in range(num_vae_steps):
                    recon, z, mu, logvar = self.net_vae(label_x)
                    unsup_loss = self.vae_loss(label_x, recon, mu, logvar, beta)
                    unlabel_recon, unlabel_z, unlabel_mu, unlabel_logvar = self.net_vae(unlabel_x)
                    transductive_loss = self.vae_loss(unlabel_x, unlabel_recon, unlabel_mu, unlabel_logvar, beta)
                    label_preds = self.net_dis(mu)
                    unlabel_preds = self.net_dis(unlabel_mu)
            
                    label_preds_real = torch.ones(label_x.size(0)).cuda()
                    unlabel_preds_real = torch.ones(unlabel_x.size(0)).cuda()
                    bce_loss = nn.BCELoss()

                    label_preds_real = label_preds_real.view(-1, 1)  # or label_preds_real.unsqueeze(1)
                    unlabel_preds_real = unlabel_preds_real.view(-1, 1)  # or unlabel_preds_real.unsqueeze(1)

                    dsc_loss = bce_loss(label_preds, label_preds_real) + bce_loss(unlabel_preds, unlabel_preds_real)

                    total_vae_loss = unsup_loss + transductive_loss + adv_param * dsc_loss

                    opt_vae.zero_grad()
                    total_vae_loss.backward()
                    opt_vae.step()
                    
				# disc
                # for count in range(num_vae_steps):
                    with torch.no_grad():
                        _, _, mu, _ = self.net_vae(label_x)
                        _, _, unlabel_mu, _ = self.net_vae(unlabel_x)
				
                    label_preds = self.net_dis(mu)
                    unlabel_preds = self.net_dis(unlabel_mu)
					
                    label_preds_real = torch.ones(label_x.size(0)).cuda()
                    unlabel_preds_real = torch.ones(unlabel_x.size(0)).cuda()
					
                    bce_loss = nn.BCELoss()

                    label_preds_real = label_preds_real.view(-1, 1)  # or label_preds_real.unsqueeze(1)
                    unlabel_preds_real = unlabel_preds_real.view(-1, 1)  # or unlabel_preds_real.unsqueeze(1)

                    dsc_loss = bce_loss(label_preds, label_preds_real) + bce_loss(unlabel_preds, unlabel_preds_real)

					
                    opt_dis.zero_grad()
                    dsc_loss.backward()
                    opt_dis.step()

                label_x = label_x.to("cpu")
                unlabel_x = unlabel_x.to("cpu")

    def pred_dis_score_vaal(self, data):
        loader_te = DataLoader(data, shuffle=False)
        self.net_vae.eval()
        self.net_dis.eval()

        scores = torch.zeros(len(data))

        with torch.no_grad():
            print("Start select data points...")
            # for x, y, idxs in tqdm(loader_te): # structure in DATALOADER has changed
            for batch_idx, (x, y, idxs) in tqdm(loader_te):
                x, y = x.cuda(), y.cuda()
                _,_,mu,_ = self.net_vae(x)
                out = self.net_dis(mu).cpu()
                scores[idxs] = out.view(-1)

        return scores

    def vae_loss(self, x, recon, mu, logvar, beta):
        mse_loss = nn.MSELoss()

        MSE = mse_loss(recon, x)
	
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD