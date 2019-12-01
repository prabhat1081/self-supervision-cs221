# /bin/bash
export CUDA_VISIBLE_DEVICES=0
# Reduce num timesteps when pretraining for sequence loss
export BASE_DIR=/lfs/local/local/prabhat8/rl
export EXP_NAME=pong-sequence-turn
export EXP_DIR=$BASE_DIR/$EXP_NAME
python -m baselines.run --alg=deepq_self_supervised \
    --env=PongNoFrameskip-v4 \
    --save_path=$EXP_DIR/models \
    --log_path=$EXP_DIR/logs \
    --checkpoint_path=$EXP_DIR/checkpoints \
    --num_timesteps=1e6 \
    --num_env=$1 \
    --learning_starts=1e4 \
    --train_deepq=True \
    --train_supervised_task=True 