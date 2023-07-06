conda activate qqEnv
nohup python -u train.py --model senet154 --multi_gpu True --gpus 0 --epoch 10 --batch_size 32 --step_lr 3 --train_dataset_path data/metadata_ISIAFood_500/train_full.txt --test_dataset_path data/metadata_ISIAFood_500/test_private.txt >senet154_32.log 2>&1 &
ps aux |grep train_senet154
tensorboard --logdir ./log --port 8080
ssh -L port_name: 127.0.0.1:8080 account_name@server.address