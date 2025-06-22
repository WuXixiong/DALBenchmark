# CIFAR100-NORMAL-500
python main.py --method AlphaMixSampling --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
python main.py --method BADGE --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
python main.py --method corelog --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
python main.py --method coremse --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
python main.py --method Coreset --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
python main.py --method CoresetCB --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
python main.py --method EntropyCB --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
python main.py --method LL --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
python main.py --method LFOSA --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
python main.py --method noise_stability --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
python main.py --method Random --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
python main.py --method SAAL --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
python main.py --method TIDAL --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5
python main.py --method Uncertainty --uncertainty CONF --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
python main.py --method Uncertainty --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
python main.py --method Uncertainty --uncertainty Margin --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
python main.py --method Uncertainty --uncertainty MeanSTD --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
python main.py --method Uncertainty --uncertainty BALD --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
python main.py --method Uncertainty --uncertainty VarRatio --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
python main.py --method Uncertainty --uncertainty MarginDropout --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
python main.py --method Uncertainty --uncertainty CONFDropout --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
python main.py --method Uncertainty --uncertainty EntropyDropout --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
python main.py --method MQNet --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 100 --n-drop 5
python main.py --method SIMILAR --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
python main.py --method CCAL --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 

# CIFAR100-NORMAL-1000
# python main.py --method AlphaMixSampling --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 1000 --n-query 1000 --n-class 100 --epochs 200 --n-drop 5  
# python main.py --method BADGE --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 1000 --n-query 1000 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method corelog --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 1000 --n-query 1000 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method coremse --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 1000 --n-query 1000 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method Coreset --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 1000 --n-query 1000 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method CoresetCB --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 1000 --n-query 1000 --n-class 100 --epochs 200 --n-drop 5  
# python main.py --method EntropyCB --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 1000 --n-query 1000 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method LL --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 1000 --n-query 1000 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method LFOSA --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 1000 --n-query 1000 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method noise_stability --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 1000 --n-query 1000 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method Random --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 1000 --n-query 1000 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method SAAL --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 1000 --n-query 1000 --n-class 100 --epochs 200 --n-drop 5  
# python main.py --method TIDAL --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 1000 --n-query 1000 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method Uncertainty --uncertainty CONF --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 1000 --n-query 1000 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method Uncertainty --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 1000 --n-query 1000 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method Uncertainty --uncertainty Margin --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 1000 --n-query 1000 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method Uncertainty --uncertainty MeanSTD --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 1000 --n-query 1000 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method Uncertainty --uncertainty BALD --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 1000 --n-query 1000 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method Uncertainty --uncertainty VarRatio --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 1000 --n-query 1000 --n-class 100 --epochs 200 --n-drop 5  
# python main.py --method Uncertainty --uncertainty MarginDropout --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 1000 --n-query 1000 --n-class 100 --epochs 200 --n-drop 5  
# python main.py --method Uncertainty --uncertainty CONFDropout --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 1000 --n-query 1000 --n-class 100 --epochs 200 --n-drop 5  
# python main.py --method Uncertainty --uncertainty EntropyDropout --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 1000 --n-query 1000 --n-class 100 --epochs 200 --n-drop 5  
# python main.py --method MQNet --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 1000 --n-query 1000 --n-class 100 --epochs 100 --n-drop 5  
# python main.py --method SIMILAR --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 1000 --n-query 1000 --n-class 100 --epochs 200 --n-drop 5  
# python main.py --method CCAL --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 1000 --n-query 1000 --n-class 100 --epochs 200 --n-drop 5  

# CIFAR100-NORMAL-2000
python main.py --method AlphaMixSampling --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 200 --n-query 200 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method BADGE --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 200 --n-query 200 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method corelog --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 200 --n-query 200 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method coremse --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 200 --n-query 200 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method Coreset --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 200 --n-query 200 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method CoresetCB --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 200 --n-query 200 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method EntropyCB --uncertainty Entropy --dataset CIFAR100 --trial 1 --cycle 10 --n-initial 200 --n-query 200 --n-class 100 --epochs 200 --n-drop 5  --seed 4
python main.py --method LL --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 200 --n-query 200 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method LFOSA --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 200 --n-query 200 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method noise_stability --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 200 --n-query 200 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method Random --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 200 --n-query 200 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method SAAL --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 200 --n-query 200 --n-class 100 --epochs 200 --n-drop 5  
# python main.py --method TIDAL --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 200 --n-query 200 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method Uncertainty --uncertainty CONF --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 200 --n-query 200 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method Uncertainty --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 200 --n-query 200 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method Uncertainty --uncertainty Margin --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 200 --n-query 200 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method Uncertainty --uncertainty MeanSTD --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 200 --n-query 200 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method Uncertainty --uncertainty BALD --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 200 --n-query 200 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method Uncertainty --uncertainty VarRatio --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 200 --n-query 200 --n-class 100 --epochs 200 --n-drop 5  
# python main.py --method Uncertainty --uncertainty MarginDropout --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 200 --n-query 200 --n-class 100 --epochs 200 --n-drop 5  
# python main.py --method Uncertainty --uncertainty CONFDropout --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 200 --n-query 200 --n-class 100 --epochs 200 --n-drop 5  
# python main.py --method Uncertainty --uncertainty EntropyDropout --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 200 --n-query 200 --n-class 100 --epochs 200 --n-drop 5  
# python main.py --method MQNet --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 200 --n-query 200 --n-class 100 --epochs 100 --n-drop 5  
# python main.py --method SIMILAR --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 200 --n-query 200 --n-class 100 --epochs 200 --n-drop 5  
# python main.py --method CCAL --uncertainty Entropy --dataset CIFAR100 --trial 5 --cycle 10 --n-initial 200 --n-query 200 --n-class 100 --epochs 200 --n-drop 5  
