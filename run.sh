conda activate qqEnv
nohup python -u train.py --model senet154 --multi_gpu True --gpus 7 --epoch 100 --batch_size 16 --step_lr 5 >senet154_16.log 2>&1 &
ps aux |grep train_senet154
tensorboard --logdir ./log --port 8080
ssh -L port_name: 127.0.0.1:8080 account_name@server.address