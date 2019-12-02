# /bin/bash
export CUDA_VISIBLE_DEVICES=0
# Reduce num timesteps when pretraining for sequence loss
export BASE_DIR=/lfs/local/local/prabhat8/rl
export EXP_NAME=$1-sequence-turn-latent-hidden
export EXP_DIR=$BASE_DIR/$EXP_NAME
mkdir -p $EXP_DIR
python -m baselines.run --alg=deepq_self_supervised \
    --env=$1NoFrameskip-v4 \
    --save_path=$EXP_DIR/models \
    --checkpoint_path=$EXP_DIR/checkpoints \
    --num_timesteps=0 \
    --num_env=1 \
    --print_freq=1 \
    --learning_starts=1e4 \
    --play
    > $EXP_DIR/run.log
