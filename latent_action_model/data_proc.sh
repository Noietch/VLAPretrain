# /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/envs/lerobot_rdt/bin/python extract_data/extract_videos_by_task.py \
#   --input /home/hadoop-aipnlp/dolphinfs_ssd_hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/datasets/bridge_dataset/1.0.0 \
#   --output /home/hadoop-aipnlp/dolphinfs_ssd_hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/datasets/bridge_v2 \
#   --fps 5
which python
which pip
python3 -c "import os; print('AFO_ENV_CLUSTER_SPEC: ', os.environ['AFO_ENV_CLUSTER_SPEC'])"
python3 -c "import json; import os; cluster_spec = json.loads(os.environ['AFO_ENV_CLUSTER_SPEC']); role = cluster_spec['role']; print('master: ', cluster_spec[role][0])"
python3 -c "import torch; print('torch: ', torch.__version__)"
python3 -c "import torch; print('torch.cuda.is_available(): ', torch.cuda.is_available())"
python3 -c "import torch; print('torch.version.cuda: ', torch.version.cuda)"
python3 -c "import torch; print('torch.backends.cudnn.version: ', torch.backends.cudnn.version())"

# # 或者使用自定义参数运行
# CUDA_VISIBLE_DEVICES=0,1,2,3 /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/envs/lerobot_rdt/bin/torchrun --nproc_per_node=4 latent_action_model/gen_latent_lapa_video_ddp.py \
#     --video_dir /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/datasets/lerobot_local \
#     --base_output_dir /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/datasets/lerobot_local/latent_action/ \

/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/envs/lerobot_rdt/bin/torchrun --nproc_per_node=8 latent_action_model/gen_latent_lapa_video_ddp.py \
    --video_dir /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/datasets/lerobot_local \
    --base_output_dir /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/datasets/lerobot_local/latent_action/


# CUDA_VISIBLE_DEVICES=0,1 /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/envs/lerobot_rdt/bin/torchrun --nproc_per_node=2 --master_port=12345 latent_action_model/gen_latent_lapa_video_ddp.py \
#     --video_dir /home/hadoop-aipnlp/dolphinfs_ssd_hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/datasets/lerobot_local/aloha_mobile_cabinet \
#     --base_output_dir /home/hadoop-aipnlp/dolphinfs_ssd_hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/datasets/lerobot_local/aloha_mobile_cabinet/latent_action/ 