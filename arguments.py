import argparse
from utils import str_to_bool

parser = argparse.ArgumentParser(description='Parameter Processing')

# new arguments
parser.add_argument("--num-IN-class", type=int, default=100, help="how many classes are in domain")
parser.add_argument("--init_dist", type=str, default='random',
                    help="uniform / random.")
parser.add_argument("--chunk_size", type=int, default=10,
                    help="divide dataset into chunk_size parts")
# PAL
parser.add_argument('--wdecay', default=5e-4, type=float,help='weight decay')
parser.add_argument('--use-ema',  action='store_true', default=True, help='use EMA model')
parser.add_argument('--ema-decay', default=0.999, type=float,help='EMA decay rate')
parser.add_argument('--is-filter', type=bool, default=True)
parser.add_argument('--is-mini', type=bool, default=True)
parser.add_argument('--lr_wnet', type=float, default=6e-5)
parser.add_argument('--meta-step', default=100, type=int, help='number of eval steps to run')
parser.add_argument('--miu', default=1.0, type=float)
parser.add_argument('--need-ID', type=int, default=1450)
# VESSAL
parser.add_argument('--zeta', help='z_t in equation 3', type=float, default=1)
parser.add_argument('--single_pass', default=False, action='store_true')
parser.add_argument('--embs', default="grad_embs", help="whether to use gradient embeddings (grad_embs) or penultimate layer embeddings (penultimate).", type=str)
parser.add_argument('--rank', help='rank of the sample-wise fisher information matrix', type=int, default=1)
parser.add_argument('--cov_inv_scaling', help='covariance inverse scaling', type=float, default=100)
parser.add_argument('--early_stop', default=False, action='store_true')
parser.add_argument('--fill_random', default=True, action='store_true')
# SAAL
parser.add_argument('--acqMode', default='Max_Diversity', type=str, help='acquisition mode / Max (max_perturbed_loss), Diff (max_perturbed_loss-original_loss), Max_Diversity, Diff_Diversity')
parser.add_argument('--labelMode', default='Pseudo', type=str, help='label mode / True (sharpness with true label), Pseudo (sharpness with pseudo label), InversePseudo (sharpness with inverse pseudo label')
parser.add_argument('--rho', default=0.05, type=float, help='sharpness computation parameter')
parser.add_argument('--pool_subset', default=2000, type=int, help='number of data points in the subset of the whole unlabelled pool')
parser.add_argument('--pool_batch_size', default=1, type=int, help='batch_size for calculating score in pool data')
# noise stabitity
parser.add_argument('--ns_subset', type=int, default=25000, help="subset in noise stabitity")
parser.add_argument('--noise_scale', type=float, default=0.001, help="noise_scale in noise stabitity")
parser.add_argument('--addendum', type=int, default=1, help="ADDENDUM in noise stabitity")
# EOAL
# parser.add_argument('--eoal_diversity', action='store_true', default=False, help='Whether to use diversity in EOAL')
# parser.add_argument('--lr-model', type=float, default=0.01, help="learning rate for model in EOAL")
# parser.add_argument('--w-unk-cls', type=int, default=1, help="w_unk_cls in EOAL")
# parser.add_argument('--pareta-alpha', type=float, default=0.8)
# parser.add_argument('--is-filter', type=bool, default=True)
# parser.add_argument('--is-mini', type=bool, default=True)
# parser.add_argument('--known-class', type=int, default=20)
# parser.add_argument('--modelB-T', type=float, default=1)
# parser.add_argument('--init-percent', type=int, default=16)
# parser.add_argument('--diversity', type=int, default=1)
# parser.add_argument('--reg-w', type=float, default=0.1)
# parser.add_argument('--w-ent', type=float, default=1)
# LFOSA
parser.add_argument('--known-T', type=float, default=0.5)
parser.add_argument('--unknown-T', type=float, default=0.5)
parser.add_argument('--lr-cent', type=float, default=0.5, help="learning rate for center loss")
parser.add_argument('--weight-cent', type=float, default=0, help="weight for center loss")
# others
parser.add_argument('--target_per_class', type=int, default=1000, help='# of target per classes in EPIG')
parser.add_argument('--max_iter', type=int, default=100, help='# of max iterations in Adversarialdeepfool')
parser.add_argument('--n-drop', type=int, default=10, help='# of drop out')
parser.add_argument('--eps', type=int, default=0.05, help='The size of the perturbation at each iteration, representing the strength of the attack in AdversarialBIM')
parser.add_argument('--openset', action='store_true', default=False, help='Whether to use openset')
parser.add_argument('--imbalanceset', action='store_true', default=False, help='Whether to use imbalanceset')
parser.add_argument('--imb-factor', type=float, default=None, help='The ratio of max class and min class in imbalanceset')
parser.add_argument('--imb_type', type=str, default='uniform', help='The relation between trainset and testset in imbalanceset')
parser.add_argument('--waal_selection', type=int, default=10, help='# selections in WAAL')
parser.add_argument('--tidal_query', type=str, default='Entropy', help='query method of TiDal')
# AlphaMix hyper-parameters
parser.add_argument('--alpha_cap', type=float, default=0.03125)
parser.add_argument('--alpha_opt', action="store_const", default=False, const=True)
parser.add_argument('--alpha_closed_form_approx', action="store_const", default=False, const=True)
parser.add_argument('--alpha_learning_rate', type=float, default=0.1,
                        help='The learning rate of finding the optimised alpha')
parser.add_argument('--alpha_clf_coef', type=float, default=1.0)
parser.add_argument('--alpha_l2_coef', type=float, default=0.01)
parser.add_argument('--alpha_learning_iters', type=int, default=5,
                        help='The number of iterations for learning alpha')
parser.add_argument('--alpha_learn_batch_size', type=int, default=1000000)

# Basic arguments
parser.add_argument('--n-initial', type=int, default=100, help='# of initial labelled samples')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset') #CIFAR10, CIFAR100, ImageNet50
parser.add_argument('--textset', action='store_true', help='text dataset')
parser.add_argument('--data_path', type=str, default='data', help='dataset path')
parser.add_argument('--gpu', default=[0], nargs="+", type=list, help='GPU id to use')
parser.add_argument("--data-parallel", default=False, type=str_to_bool, help="whether parallel or not")
parser.add_argument('--ood-rate', type=float, default=0.6, metavar='N', help='OOD rate in unlabeled set')
parser.add_argument('--n-class', type=int, default=10, help='# of classes') # originally str
parser.add_argument('--trial', type=int, default=5, help='# of runs')
parser.add_argument('--cycle', type=int, default=10, help='# of AL cycles')
parser.add_argument('--n-query', type=int, default=1000, help='# of query samples')
parser.add_argument('--subset', type=int, default=50000, help='subset')
parser.add_argument('--resolution', type=int, default=32, help='resolution') # 32
parser.add_argument('--model', type=str, default='ResNet18', help='model')
parser.add_argument('--print_freq', '-p', default=300, type=int, help='print frequency (default: 20)')
parser.add_argument('--seed', default=0, type=int, help="random seed")
parser.add_argument('-j', '--workers', default=5, type=int, help='number of data loading workers (default: 4)')
parser.add_argument("--ssl-save", default=True, type=str_to_bool, help="whether save ssl model or not")

# Optimizer and scheduler
parser.add_argument('--optimizer', default="SGD", help='optimizer to use, e.g. SGD, Adam')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate for updating network parameters')
parser.add_argument('--lr-mqnet', type=float, default=0.001, help='learning rate for updating mqnet')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum (default: 0.9)')
parser.add_argument('-wd', '--weight_decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)', dest='weight_decay')
parser.add_argument("--scheduler", default="MultiStepLR", type=str, help="Learning rate scheduler") #CosineAnnealingLR, StepLR, MultiStepLR
parser.add_argument('--min_lr', type=float, default=1e-4, help='minimum learning rate for CosineAnnealingLR')
parser.add_argument("--gamma", type=float, default=.5, help="Gamma value for StepLR")
parser.add_argument("--step_size", type=float, default=50, help="Step size for StepLR")
parser.add_argument('--milestone', type=int, nargs='+', default=[100, 150], metavar='M', help='Milestone for MultiStepLR')
parser.add_argument('--warmup', type=int, default=10, metavar='warmup', help='warmup epochs')

# ing
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--epoch-loss', default=120, type=int, help='number of epochs for training loss module in LL')
parser.add_argument('--epochs-ccal', default=700, type=int, help='number of epochs for training contrastive coders in CCAL')
parser.add_argument('--epochs-csi', default=1000, type=int, help='number of epochs for training CSI')
parser.add_argument('--epochs-mqnet', default=100, type=int, help='number of epochs for training mqnet')
parser.add_argument('--steps-per-epoch', type=int, default=100, metavar='N', help='number of steps per epoch')
parser.add_argument('--batch-size', "-b", default=64, type=int, metavar='N')
parser.add_argument("--test-batch-size", "-tb", default=500, type=int)
parser.add_argument('--ccal-batch-size', default=32, type=int, metavar='N')
parser.add_argument('--csi-batch-size', default=32, type=int, metavar='N')

# Testing
parser.add_argument("--test_interval", '-ti', default=1, type=int, help=
"the number of training epochs to be preformed between two test epochs; a value of 0 means no test will be run (default: 1)")
parser.add_argument("--test_fraction", '-tf', type=float, default=1.,
                    help="proportion of test dataset used for evaluating the model (default: 1.)")

# AL Algorithm
parser.add_argument('--method', default="Uncertainty", help="specifiy AL method to use") #Uncertainty, Coreset, LL, BADGE, CCAL, SIMILAR, MQNet
parser.add_argument('--submodular', default="logdetcmi", help="specifiy submodular function to use") #flcmi, logdetcmi
parser.add_argument('--submodular_greedy', default="LazyGreedy", help="specifiy greedy algorithm for submodular optimization")
parser.add_argument('--uncertainty', default="CONF", help="specifiy uncertanty score to use") #CONF, Margin, Entropy
# for CCAL
parser.add_argument("--resize_factor", help='resize scale is sampled from [resize_factor, 1.0]',default=0.08, type=float)
parser.add_argument("--resize_fix", help='resize scale is fixed to resize_factor (not (resize_factor, 1.0])',action='store_true')
parser.add_argument('--sim_lambda', help='Weight for SimCLR loss',default=1.0, type=float)
parser.add_argument('--shift_trans_type', help='shifting transformation type', default='rotation',choices=['rotation', 'cutperm', 'none'], type=str)
parser.add_argument("--ood_samples", help='number of samples to compute OOD score',default=1, type=int)
parser.add_argument('--k', help='Initial learning rate', default=100.0, type=float)
parser.add_argument('--t', help='Initial learning rate', default=0.9, type=float)
# for MQNet
parser.add_argument('--mqnet-mode', default="CONF", help="specifiy the mode of MQNet to use") #CONF, LL

# Checkpoint and resumption
parser.add_argument('--save_path', "-sp", type=str, default='', help='path to save results (default: do not save)')
parser.add_argument('--resume', '-r', type=str, default='', help="path to latest checkpoint (default: do not load)")

args = parser.parse_args()