
import json, os, subprocess, sys, glob
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


DEFAULT_CONFIG = {
    "base_path": "/home/hadoop-aipnlp/dolphinfs_ssd_hadoop-aipnlp/EVA/yiyang11/workspace/VLAPretrain/datasets/lerobot_local_debug",
    "episodes_jsonl_paths": [
        "main/jsonl_meta/episodes",  # 新格式：meta_jsonl/episodes/*.jsonl
        "main/meta"                   # 旧格式：meta/episodes.jsonl
    ],
    "videos_path": "main/videos",
    "output_base_path": "main/video",
    "cameras": [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist"
    ],
    "video_settings": {
        "fps": 50.0,
        "codec_video": "libx264",
        "codec_audio": "aac",
        "timeout_seconds": 300
    },
    "output_naming": {
        "format": "observation.images.{camera_short_name}_episode_{episode_index:06d}.mp4"
    },
    "processing": {
        "parallel_jobs": 50
    }
}

class VideoSliceProcessor:

    def __init__(self, dataset_path: Optional[str] = None):
        self.config = DEFAULT_CONFIG.copy()

        if dataset_path:
            self.base_path = dataset_path
        else:
            self.base_path = self.config['base_path']

        # 支持多个可能的episodes路径
        self.episodes_jsonl_paths = [
            os.path.join(self.base_path, path)
            for path in self.config['episodes_jsonl_paths']
        ]

        self.videos_path = os.path.join(
            self.base_path,
            self.config['videos_path']
        )
        self.output_base_path = os.path.join(
            self.base_path,
            self.config['output_base_path']
        )
        self.cameras = self.config['cameras']
        self.video_settings = self.config['video_settings']
        self.output_naming_format = self.config['output_naming']['format']
        self.parallel_jobs = self.config['processing']['parallel_jobs']

        self.total_videos = 0
        self.successful_videos = 0
        self.failed_videos = 0

    @staticmethod
    def find_datasets(base_dir: str) -> List[str]:
        datasets = []
        base_path = Path(base_dir)

        # 查找新格式：meta_jsonl/episodes
        for episodes_dir in base_path.rglob("jsonl_meta/episodes"):
            dataset_root = episodes_dir.parent.parent.parent
            if (dataset_root / "main" / "videos").exists() and (dataset_root / "main" / "jsonl_meta" / "episodes").exists():
                datasets.append(str(dataset_root))

        # 查找旧格式：meta/episodes.jsonl
        for meta_dir in base_path.rglob("meta"):
            if (meta_dir / "episodes.jsonl").exists():
                dataset_root = meta_dir.parent.parent
                if (dataset_root / "main" / "videos").exists():
                    datasets.append(str(dataset_root))

        return sorted(list(set(datasets)))

    @staticmethod
    def build_output_path(dataset_path: str, camera_short_name: str, episode_index: int) -> str:
        output_dir = os.path.join(dataset_path, "main/video", camera_short_name)
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, f"observation.images.{camera_short_name}_episode_{episode_index:06d}.mp4")

    def _extract_config_from_jsonl(self, episodes: List[Dict]) -> None:
        if not episodes:
            print("⚠️  未找到episode数据，使用默认配置")
            return
        print("=" * 80 + "\n从jsonl文件中提取配置信息...\n" + "=" * 80)
        first_episode = episodes[0]

        # 从jsonl中自动检测所有摄像头
        detected_cameras = []
        for key in first_episode.keys():
            if key.startswith("videos/") and key.endswith("/chunk_index"):
                camera_name = key.replace("videos/", "").replace("/chunk_index", "")
                detected_cameras.append(camera_name)

        if detected_cameras:
            self.cameras = detected_cameras
            for c in detected_cameras:
                print(f"✓ 检测到摄像头: {c}")
            print(f"✓ 使用检测到的摄像头: {len(self.cameras)}个")
        else:
            print(f"⚠️  未检测到摄像头，使用默认摄像头配置")
        print(f"✓ Episode数据:\n  - 总数: {len(episodes)}\n  - 第一个episode索引: {first_episode.get('episode_index')}\n  - 帧范围: {first_episode.get('dataset_from_index')} - {first_episode.get('dataset_to_index')}\n✓ 使用的帧率: {self.video_settings['fps']} fps\n" + "=" * 80 + "\n")

    def load_episodes(self) -> List[Dict]:
        episodes = []

        # 尝试多个可能的路径
        for episodes_path in self.episodes_jsonl_paths:
            if not os.path.exists(episodes_path):
                continue

            # 查找jsonl文件
            jsonl_files = glob.glob(os.path.join(episodes_path, "**/*.jsonl"), recursive=True)

            if jsonl_files:
                for jsonl_file in sorted(jsonl_files):
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                episodes.append(json.loads(line))
                break  # 找到文件后就停止搜索

        if not episodes:
            print(f"警告：未找到jsonl文件在以下路径:")
            for path in self.episodes_jsonl_paths:
                print(f"  - {path}")

        return episodes

    def get_video_file_path(self, camera: str, chunk_index: int, file_index: int) -> str:
        return os.path.join(self.videos_path, camera, f"chunk-{chunk_index:03d}", f"file-{file_index:03d}.mp4")

    def slice_video(self, input_video: str, output_video: str,
                   start_time: float, end_time: float) -> bool:
        if not os.path.exists(input_video):
            print(f"  ✗ 错误：输入视频不存在 {input_video}")
            return False
        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        duration = end_time - start_time
        cmd = ["ffmpeg", "-i", input_video, "-ss", str(start_time), "-t", str(duration),
               "-c:v", self.video_settings['codec_video'], "-c:a", self.video_settings['codec_audio'], "-y", output_video]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.video_settings['timeout_seconds'])
            if result.returncode == 0:
                file_size = os.path.getsize(output_video) / (1024 * 1024)
                print(f"  ✓ 成功: {os.path.basename(output_video)} ({file_size:.2f}MB)")
                return True
            print(f"  ✗ 失败: {os.path.basename(output_video)}")
            if result.stderr:
                print(f"    错误: {result.stderr[:200]}")
            return False
        except subprocess.TimeoutExpired:
            print(f"  ✗ 超时: {os.path.basename(output_video)}")
        except Exception as e:
            print(f"  ✗ 异常: {os.path.basename(output_video)}\n    错误: {str(e)}")
        return False

    def process_single_video(self, episode: Dict, camera: str) -> bool:
        camera_key_prefix = f"videos/{camera}"
        chunk_index = episode.get(f"{camera_key_prefix}/chunk_index")
        file_index = episode.get(f"{camera_key_prefix}/file_index")
        from_timestamp = episode.get(f"{camera_key_prefix}/from_timestamp")
        to_timestamp = episode.get(f"{camera_key_prefix}/to_timestamp")

        if chunk_index is None or file_index is None or from_timestamp is None or to_timestamp is None:
            return False

        input_video = self.get_video_file_path(camera, chunk_index, file_index)
        camera_short_name = camera.split('.')[-1]
        output_video = self.build_output_path(self.base_path, camera_short_name, episode.get('episode_index'))
        return self.slice_video(input_video, output_video, from_timestamp, to_timestamp)

    def process_episodes(self):
        print("=" * 80 + "\n视频切片处理器 v2\n" + "=" * 80 + "\n开始加载episode元数据...\n" + "=" * 80)
        episodes = self.load_episodes()
        print(f"✓ 加载了 {len(episodes)} 个episode\n")
        if not episodes:
            print("错误：没有找到任何episode数据")
            return

        self._extract_config_from_jsonl(episodes)

        print("=" * 80 + "\n开始切片视频...\n" + "=" * 80)

        tasks = [(episode, camera) for episode in episodes for camera in self.cameras]

        self.total_videos = len(tasks)

        if self.parallel_jobs > 1:
            self._process_parallel(tasks)
        else:
            self._process_sequential(tasks)

        self._print_summary()

    def _process_sequential(self, tasks: List[tuple]):
        for episode, camera in tasks:
            if self.process_single_video(episode, camera):
                self.successful_videos += 1
            else:
                self.failed_videos += 1

    def _process_parallel(self, tasks: List[tuple]):
        with ThreadPoolExecutor(max_workers=self.parallel_jobs) as executor:
            futures = {executor.submit(self.process_single_video, e, c): (e, c) for e, c in tasks}
            for future in as_completed(futures):
                if future.result():
                    self.successful_videos += 1
                else:
                    self.failed_videos += 1

    def _print_summary(self):
        print("\n" + "=" * 80 + "\n切片完成！\n" + "=" * 80)
        print(f"总视频数:  {self.total_videos}")
        print(f"成功:      {self.successful_videos}")
        print(f"失败:      {self.failed_videos}")
        if self.total_videos > 0:
            print(f"成功率:    {self.successful_videos/self.total_videos*100:.1f}%")
        print(f"输出目录:  {self.output_base_path}")
        print("=" * 80)


def main():
    try:
        base_dir = DEFAULT_CONFIG['base_path']

        print("=" * 80 + "\n视频切片处理器 - 批量处理模式\n" + "=" * 80 + f"\n基础目录: {base_dir}\n\n搜索数据集...")
        datasets = VideoSliceProcessor.find_datasets(base_dir)

        if not datasets:
            print("未找到任何数据集")
            return

        print(f"找到 {len(datasets)} 个数据集\n")

        total_stats = {
            'total_videos': 0,
            'successful_videos': 0,
            'failed_videos': 0,
            'datasets_processed': 0
        }

        for i, dataset_path in enumerate(datasets, 1):
            print("=" * 80 + f"\n处理数据集 [{i}/{len(datasets)}]\n" + "=" * 80 + f"\n路径: {dataset_path}\n")
            try:
                processor = VideoSliceProcessor(dataset_path)
                processor.process_episodes()

                total_stats['total_videos'] += processor.total_videos
                total_stats['successful_videos'] += processor.successful_videos
                total_stats['failed_videos'] += processor.failed_videos
                total_stats['datasets_processed'] += 1

            except Exception as e:
                print(f"处理数据集失败: {str(e)}\n")
                continue

        print("\n" + "=" * 80 + "\n批量处理完成！\n" + "=" * 80)
        print(f"处理的数据集数: {total_stats['datasets_processed']}/{len(datasets)}")
        print(f"总视频数:       {total_stats['total_videos']}")
        print(f"成功:           {total_stats['successful_videos']}")
        print(f"失败:           {total_stats['failed_videos']}")
        if total_stats['total_videos'] > 0:
            print(f"成功率:         {total_stats['successful_videos']/total_stats['total_videos']*100:.1f}%")
        print("=" * 80)

    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
