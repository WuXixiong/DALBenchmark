# AGNEWS-NORMAL-50
python main.py --method AlphaMixSampling --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method BADGE --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method corelog --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method coremse --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Coreset --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method CoresetCB --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method EntropyCB --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method LFOSA --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method noise_stability --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Random --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method SAAL --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty CONF --dataset AGNEWS --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty Margin --dataset AGNEWS --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty MeanSTD --dataset AGNEWS --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty BALD --dataset AGNEWS --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty VarRatio --dataset AGNEWS --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty MarginDropout --dataset AGNEWS --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty CONFDropout --dataset AGNEWS --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty EntropyDropout --dataset AGNEWS --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5

# AGNEWS-NORMAL-20 
python main.py --method AlphaMixSampling --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method BADGE --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method corelog --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method coremse --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Coreset --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method CoresetCB --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method EntropyCB --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method LFOSA --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method noise_stability --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Random --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method SAAL --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty CONF --dataset AGNEWS --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty Margin --dataset AGNEWS --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty MeanSTD --dataset AGNEWS --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty BALD --dataset AGNEWS --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty VarRatio --dataset AGNEWS --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty MarginDropout --dataset AGNEWS --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty CONFDropout --dataset AGNEWS --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty EntropyDropout --dataset AGNEWS --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5

# AGNEWS-NORMAL-10
python main.py --method AlphaMixSampling --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method BADGE --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method corelog --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method coremse --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Coreset --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method CoresetCB --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method EntropyCB --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method LFOSA --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method noise_stability --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Random --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method SAAL --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty CONF --dataset AGNEWS --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty Margin --dataset AGNEWS --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty MeanSTD --dataset AGNEWS --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty BALD --dataset AGNEWS --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty VarRatio --dataset AGNEWS --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty MarginDropout --dataset AGNEWS --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty CONFDropout --dataset AGNEWS --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty EntropyDropout --dataset AGNEWS --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5