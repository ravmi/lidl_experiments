#/bin/bash

source ~/glow-pytorch/venv/bin/activate
sstatus=$?
if [ "$sstatus" -ne 0 ]; then
    exit 2
fi

srun --time 3-0 --qos=8gpu7d  --gres=gpu:1 python example.py &
deactivate
