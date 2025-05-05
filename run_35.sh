#python main.py --method AlphaMixSampling --uncertainty Entropy --dataset TINYIMAGENET --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
#python main.py --method corelog --uncertainty Entropy --dataset TINYIMAGENET --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
#python main.py --method coremse --uncertainty Entropy --dataset TINYIMAGENET --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
#python main.py --method LL --uncertainty Entropy --dataset TINYIMAGENET --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
#python main.py --method SAAL --uncertainty Entropy --dataset TINYIMAGENET --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5 
#python main.py --method TIDAL --uncertainty Entropy --dataset TINYIMAGENET --trial 5 --cycle 10 --n-initial 500 --n-query 500 --n-class 100 --epochs 200 --n-drop 5

python main.py --method AlphaMixSampling --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method corelog --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method coremse --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method SAAL --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 50 --n-query 50 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset

python main.py --method AlphaMixSampling --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method corelog --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method coremse --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method SAAL --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 20 --n-query 20 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset

python main.py --method AlphaMixSampling --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method corelog --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method coremse --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset
python main.py --method SAAL --uncertainty Entropy --dataset AGNEWS --trial 5 --cycle 10 --n-initial 10 --n-query 10 --optimizer AdamW --lr 0.0001 --model DistilBert  --epochs 30 --n-drop 5 --textset





