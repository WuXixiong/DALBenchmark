# CIFAR10-NORMAL-100 ood-rate 0.2
python main.py --method AlphaMixSampling --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.2
python main.py --method BADGE --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.2
python main.py --method corelog --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.2
python main.py --method coremse --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.2
python main.py --method Coreset --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.2
python main.py --method CoresetCB --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.2
python main.py --method EntropyCB --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.2
python main.py --method LL --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.2
python main.py --method LFOSA --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.2
python main.py --method noise_stability --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.2
python main.py --method Random --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.2
python main.py --method SAAL --uncertainty Entropy --dataset CIFAR10 --trial 3 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.2
python main.py --method TIDAL --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5
python main.py --method Uncertainty --uncertainty CONF --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.2
python main.py --method Uncertainty --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.2
python main.py --method Uncertainty --uncertainty Margin --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.2
python main.py --method Uncertainty --uncertainty MeanSTD --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.2
python main.py --method Uncertainty --uncertainty BALD --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.2
python main.py --method Uncertainty --uncertainty VarRatio --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.2

# CIFAR10-NORMAL-100 ood-rate 0.4
python main.py --method AlphaMixSampling --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.4
python main.py --method BADGE --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.4
python main.py --method corelog --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.4
python main.py --method coremse --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.4
python main.py --method Coreset --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.4
python main.py --method CoresetCB --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.4
python main.py --method EntropyCB --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.4
python main.py --method LL --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.4
python main.py --method LFOSA --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.4
python main.py --method noise_stability --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.4
python main.py --method Random --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.4
python main.py --method SAAL --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.4
python main.py --method TIDAL --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5
python main.py --method Uncertainty --uncertainty CONF --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.4
python main.py --method Uncertainty --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.4
python main.py --method Uncertainty --uncertainty Margin --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.4
python main.py --method Uncertainty --uncertainty MeanSTD --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.4
python main.py --method Uncertainty --uncertainty BALD --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.4
python main.py --method Uncertainty --uncertainty VarRatio --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.4

# CIFAR10 ood-rate 0.6
python main.py --method AlphaMixSampling --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.6
python main.py --method BADGE --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.6
python main.py --method corelog --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.6
python main.py --method coremse --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.6
python main.py --method Coreset --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.6
python main.py --method CoresetCB --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.6
python main.py --method EntropyCB --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.6
python main.py --method LL --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.6
python main.py --method LFOSA --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.6
python main.py --method noise_stability --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.6
python main.py --method Random --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.6
python main.py --method SAAL --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.6
python main.py --method TIDAL --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5
python main.py --method Uncertainty --uncertainty CONF --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.6
python main.py --method Uncertainty --uncertainty Entropy --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.6
python main.py --method Uncertainty --uncertainty Margin --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.6
python main.py --method Uncertainty --uncertainty MeanSTD --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.6
python main.py --method Uncertainty --uncertainty BALD --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.6
python main.py --method Uncertainty --uncertainty VarRatio --dataset CIFAR10 --trial 5 --cycle 10 --n-initial 100 --n-query 100 --n-class 10 --epochs 200 --n-drop 5 --imb-factor 0.6