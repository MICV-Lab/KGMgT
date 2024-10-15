export CUDA_VISIBLE_DEVICES=0 # 0,1,2,3
PYTHONPATH="./:${PYTHONPATH}" python kg_net/test.py -opt "options/test/test_restoration.yml"