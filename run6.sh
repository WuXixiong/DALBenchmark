# AGNEWS-NORMAL-50
python main.py --method AlphaMixSampling --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method BADGE --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method corelog --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method coremse --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method Coreset --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method CoresetCB --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method EntropyCB --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method LFOSA --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method noise_stability --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method Random --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method SAAL --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method Uncertainty --uncertainty CONF --dataset YELP --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method Uncertainty --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method Uncertainty --uncertainty Margin --dataset YELP --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method Uncertainty --uncertainty MeanSTD --dataset YELP --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method Uncertainty --uncertainty BALD --dataset YELP --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method Uncertainty --uncertainty VarRatio --dataset YELP --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
# python main.py --method Uncertainty --uncertainty MarginDropout --dataset YELP --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
# python main.py --method Uncertainty --uncertainty CONFDropout --dataset YELP --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
# python main.py --method Uncertainty --uncertainty EntropyDropout --dataset YELP --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset

# AGNEWS-NORMAL-20 
python main.py --method AlphaMixSampling --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method BADGE --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method corelog --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method coremse --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method Coreset --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method CoresetCB --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method EntropyCB --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method LFOSA --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method noise_stability --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method Random --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method SAAL --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method Uncertainty --uncertainty CONF --dataset YELP --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method Uncertainty --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method Uncertainty --uncertainty Margin --dataset YELP --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method Uncertainty --uncertainty MeanSTD --dataset YELP --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method Uncertainty --uncertainty BALD --dataset YELP --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method Uncertainty --uncertainty VarRatio --dataset YELP --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
# python main.py --method Uncertainty --uncertainty MarginDropout --dataset YELP --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
# python main.py --method Uncertainty --uncertainty CONFDropout --dataset YELP --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
# python main.py --method Uncertainty --uncertainty EntropyDropout --dataset YELP --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset

# AGNEWS-NORMAL-100
python main.py --method AlphaMixSampling --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 100 --n-query 100 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method BADGE --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 100 --n-query 100 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method corelog --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 100 --n-query 100 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method coremse --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 100 --n-query 100 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method Coreset --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 100 --n-query 100 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method CoresetCB --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 100 --n-query 100 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method EntropyCB --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 100 --n-query 100 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method LFOSA --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 100 --n-query 100 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method noise_stability --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 100 --n-query 100 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method Random --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 100 --n-query 100 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method SAAL --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 100 --n-query 100 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method Uncertainty --uncertainty CONF --dataset YELP --trial 5 --cycle 10 --n-initial 100 --n-query 100 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method Uncertainty --uncertainty Entropy --dataset YELP --trial 5 --cycle 10 --n-initial 100 --n-query 100 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method Uncertainty --uncertainty Margin --dataset YELP --trial 5 --cycle 10 --n-initial 100 --n-query 100 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method Uncertainty --uncertainty MeanSTD --dataset YELP --trial 5 --cycle 10 --n-initial 100 --n-query 100 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method Uncertainty --uncertainty BALD --dataset YELP --trial 5 --cycle 10 --n-initial 100 --n-query 100 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method Uncertainty --uncertainty VarRatio --dataset YELP --trial 5 --cycle 10 --n-initial 100 --n-query 100 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
# python main.py --method Uncertainty --uncertainty MarginDropout --dataset YELP --trial 5 --cycle 10 --n-initial 100 --n-query 100 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
# python main.py --method Uncertainty --uncertainty CONFDropout --dataset YELP --trial 5 --cycle 10 --n-initial 100 --n-query 100 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
# python main.py --method Uncertainty --uncertainty EntropyDropout --dataset YELP --trial 5 --cycle 10 --n-initial 100 --n-query 100 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset