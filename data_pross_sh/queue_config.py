import os

# ==========================================
# 默认队列配置（当环境变量未设置时使用）
# ==========================================
DEFAULT_QUEUE = "sh"

# 从环境变量读取队列名称，如果没有设置则使用默认值
CURRENT_QUEUE = os.environ.get('DEXBOTIC_QUEUE', DEFAULT_QUEUE)

# sh:/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/3A/multimodal/yiyang11

# ==========================================
# 队列路径配置字典
# ==========================================
QUEUE_CONFIGS = {
    "zw": {
        "code_root": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing/workspace/dexbotic",
        "data_root": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/yangheqing",
    },
    "sh": {
        "code_root": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/3A/multimodal/yiyang11/dexbotic",
        "data_root": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/3A/multimodal/yiyang11",
    },
    
    "queue3": {
        "code_root": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/queue3/workspace/dexbotic",
        "data_root": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/queue3",
    },
}

MODEL_CONFIGS = {
    "lapa": {
        "model_path": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/yiyang11/huggingface/latent-action-pretraining/LAPA-7B-openx/laq_openx.pt",
        "image_size": 256,
    },
    "univla": {
        "model_path": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/yiyang11/huggingface/qwbu/univla-latent-action-model/main/lam-stage-2.ckpt",
        "image_size": 224,
    },
}

# ==========================================
# 自动加载当前队列配置（无需修改）
# ==========================================
if CURRENT_QUEUE not in QUEUE_CONFIGS:
    raise ValueError(
        f"队列 '{CURRENT_QUEUE}' 未在配置中找到。\n"
        f"可用的队列: {', '.join(QUEUE_CONFIGS.keys())}\n"
        f"请在 dexbotic/queue_config.py 中检查 CURRENT_QUEUE 的值"
    )

_config = QUEUE_CONFIGS[CURRENT_QUEUE]

# 导出为模块级变量，可以直接 import 使用
CODE_ROOT = _config["code_root"]
DATA_ROOT = _config["data_root"]


# ==========================================
# 辅助函数（可选使用）
# ==========================================
def get_code_root():
    """获取代码根路径"""
    return CODE_ROOT


def get_data_root():
    """获取数据根路径"""
    return DATA_ROOT


def get_config():
    """获取完整配置字典"""
    return _config.copy()