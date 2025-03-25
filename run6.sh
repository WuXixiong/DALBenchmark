# IMDB-NORMAL-10
python main.py --method AlphaMixSampling --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method BADGE --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method corelog --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method coremse --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Coreset --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method CoresetCB --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method EntropyCB --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method LFOSA --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method noise_stability --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Random --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method SAAL --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty CONF --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty Margin --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty MeanSTD --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty BALD --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty VarRatio --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty MarginDropout --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty CONFDropout --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty EntropyDropout --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5

# IMDB-NORMAL-5
python main.py --method AlphaMixSampling --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 5 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method BADGE --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 5 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method corelog --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 5 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method coremse --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 5 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Coreset --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 5 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method CoresetCB --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 5 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method EntropyCB --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 5 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method LFOSA --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 5 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method noise_stability --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 5 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Random --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 5 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method SAAL --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 5 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty CONF --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 5 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 5 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty Margin --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 5 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty MeanSTD --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 5 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty BALD --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 5 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty VarRatio --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 5 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty MarginDropout --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 5 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty CONFDropout --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 5 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty EntropyDropout --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 5 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5

# IMDB-NORMAL-1
python main.py --method AlphaMixSampling --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 1 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method BADGE --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 1 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method corelog --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 1 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method coremse --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 1 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Coreset --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 1 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method CoresetCB --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 1 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method EntropyCB --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 1 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method LFOSA --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 1 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method noise_stability --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 1 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Random --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 1 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method SAAL --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 1 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty CONF --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 1 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty Entropy --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 1 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty Margin --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 1 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty MeanSTD --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 1 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty BALD --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 1 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty VarRatio --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 1 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty MarginDropout --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 1 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty CONFDropout --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 1 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty EntropyDropout --dataset IMDB --trial 5 --cycle 10 --n-initial 10 --n-query 1 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5