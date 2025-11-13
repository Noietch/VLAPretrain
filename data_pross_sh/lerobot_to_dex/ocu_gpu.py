import torch
import time


def occupy_gpu_memory_forever(device_id=0, target_gb=20):
    """
    在指定显卡上创建大张量并通过死循环保持占用，长期占用约target_gb显存
    
    Args:
        device_id: 显卡ID（如0, 1, 2...）
        target_gb: 目标显存占用（GB）
    """
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please use a GPU with CUDA support."
        )

    # 检查设备ID有效性
    if device_id >= torch.cuda.device_count():
        raise ValueError(
            f"Device ID {device_id} is invalid. Available devices: {torch.cuda.device_count()}"
        )

    device = torch.device(f"cuda:{device_id}")
    print(
        f"Using device: {torch.cuda.get_device_name(device)} (ID: {device_id})"
    )

    # 计算目标显存对应的参数
    target_bytes = target_gb * 1024**3
    dtype = torch.float32
    element_size = dtype.itemsize  # 4字节/元素
    total_elements = (target_bytes + element_size - 1) // element_size
    dim = int(total_elements**0.5) + 1
    actual_gb = (dim * dim * element_size) / 1024**3
    print(
        f"Tensor shape: [{dim}, {dim}], Theoretical memory: {actual_gb:.2f}GB (target: {target_gb}GB)"
    )

    # 创建大张量（保持引用不被释放）
    try:
        big_tensor = torch.randn((dim, dim), dtype=dtype, device=device)
        allocated_gb = torch.cuda.memory_allocated(device) / 1024**3
        print(f"Tensor created. Allocated memory: {allocated_gb:.2f}GB")
    except RuntimeError as e:
        print(f"Failed to create tensor: {e}. Insufficient memory?")
        return

    # 执行一次矩阵乘法确保显存完全占用
    try:
        # result = big_tensor @ big_tensor
        # del result
        torch.cuda.synchronize(device)
        print("Initial computation done. Starting to hold memory...")
    except RuntimeError as e:
        print(f"Initial computation failed: {e}")
        return

    # 死循环保持占用（定期打印状态）
    try:
        while True:
            # 定期刷新显存状态（避免被优化释放）
            result = big_tensor @ big_tensor
            del result
            current_used = torch.cuda.memory_allocated(device) / 1024**3
            print(
                f"[{time.strftime('%H:%M:%S')}] Holding memory on device {device_id}: {current_used:.2f}GB",
                end='\r'
            )
            time.sleep(5)  # 每5秒检查一次
    except KeyboardInterrupt:
        print("\nReceived stop signal. Cleaning up...")
    finally:
        # 清理资源（仅在手动中断时执行）
        del big_tensor
        torch.cuda.empty_cache()
        print(f"Memory released on device {device_id}")


import multiprocessing


def arger_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Occupy GPU memory indefinitely."
    )
    parser.add_argument(
        "--device_id",
        type=str,
        default="0,1",
        help="GPU device ID to use (default: 0)",
    )
    return parser


# 使用示例：占用0号显卡，保持10GB显存占用
if __name__ == "__main__":
    device_ids = [
        int(i) for i in arger_parser().parse_args().device_id.split(',')
    ]

    def worker(device_id):
        occupy_gpu_memory_forever(
            device_id=device_id,  # 指定显卡ID
            target_gb=40  # 目标显存占用
        )

    processes = []
    for device_id in device_ids:
        p = multiprocessing.Process(target=worker, args=(device_id, ))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
