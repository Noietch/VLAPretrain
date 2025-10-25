/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/envs/lerobot_rdt/bin/python extract_data/extract_videos_by_task.py \
  --input /home/hadoop-aipnlp/dolphinfs_ssd_hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/datasets/bridge_dataset/1.0.0 \
  --output /home/hadoop-aipnlp/dolphinfs_ssd_hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/datasets/bridge_v2 \
  --fps 5

# 或者使用自定义参数运行
CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/envs/lerobot_rdt/bin/torchrun --nproc_per_node=8 --master_port=12345 latent_action_model/gen_latent_lapa_video_ddp.py \
    --video_dir /home/hadoop-aipnlp/dolphinfs_ssd_hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/datasets/bridge_v2/video \
    --base_output_dir /home/hadoop-aipnlp/dolphinfs_ssd_hadoop-aipnlp/EVA/yangheqing/workspace/LVLA/datasets/bridge_v2/latent_action/ \

