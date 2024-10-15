export CUDA_VISIBLE_DEVICES=0 # 0,1,2,3
PYTHONPATH="./:${PYTHONPATH}" python kg_net/train.py -opt "options/train/train_restoration.yml"