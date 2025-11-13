export PYTHONPATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/LVLA
torchrun --standalone --nnodes 1 --nproc-per-node 2 latent_action_model/UniVLA/main.py fit \
    --config latent_action_model/UniVLA/config/test.yaml \
    2>&1 | tee lam-stage-1.log