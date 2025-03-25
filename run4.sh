# SST5-NORMAL-50
python main.py --method AlphaMixSampling --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method BADGE --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method corelog --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method coremse --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Coreset --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method CoresetCB --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method EntropyCB --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method LFOSA --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method noise_stability --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Random --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method SAAL --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty CONF --dataset SST5 --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty Margin --dataset SST5 --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty MeanSTD --dataset SST5 --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty BALD --dataset SST5 --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty VarRatio --dataset SST5 --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty MarginDropout --dataset SST5 --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty CONFDropout --dataset SST5 --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty EntropyDropout --dataset SST5 --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5

# SST5-NORMAL-20 
python main.py --method AlphaMixSampling --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method BADGE --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method corelog --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method coremse --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Coreset --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method CoresetCB --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method EntropyCB --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method LFOSA --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method noise_stability --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Random --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method SAAL --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty CONF --dataset SST5 --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty Margin --dataset SST5 --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty MeanSTD --dataset SST5 --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty BALD --dataset SST5 --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty VarRatio --dataset SST5 --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty MarginDropout --dataset SST5 --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty CONFDropout --dataset SST5 --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty EntropyDropout --dataset SST5 --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5

# SST5-NORMAL-10
python main.py --method AlphaMixSampling --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method BADGE --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method corelog --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method coremse --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Coreset --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method CoresetCB --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method EntropyCB --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method LFOSA --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method noise_stability --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Random --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method SAAL --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty CONF --dataset SST5 --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty Margin --dataset SST5 --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty MeanSTD --dataset SST5 --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty BALD --dataset SST5 --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
python main.py --method Uncertainty --uncertainty VarRatio --dataset SST5 --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty MarginDropout --dataset SST5 --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty CONFDropout --dataset SST5 --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5
# python main.py --method Uncertainty --uncertainty EntropyDropout --dataset SST5 --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert --n-class 2 --epochs 30 --n-drop 5