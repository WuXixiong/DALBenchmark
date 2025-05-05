# # NOISE
# python main.py --method noise_stability --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
# python main.py --method noise_stability --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
# python main.py --method noise_stability --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
# python main.py --method noise_stability --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
# python main.py --method noise_stability --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
# python main.py --method noise_stability --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset

# # CB
# python main.py --method EntropyCB --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
# python main.py --method EntropyCB --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
# python main.py --method EntropyCB --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
# python main.py --method CoresetCB --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
# python main.py --method CoresetCB --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
# python main.py --method CoresetCB --uncertainty Entropy --dataset SST5 --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset

python main.py --method TIDAL --uncertainty Entropy --dataset TINYIMAGENET --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5
python main.py --method TIDAL --uncertainty Entropy --dataset TINYIMAGENET --trial 5 --cycle 10 --n-initial 1000 --n-query 1000 --n-class 100 --epochs 200 --n-drop 5  
python main.py --method TIDAL --uncertainty Entropy --dataset TINYIMAGENET --trial 5 --cycle 10 --n-initial 2000 --n-query 2000 --n-class 100 --epochs 200 --n-drop 5  