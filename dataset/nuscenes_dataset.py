import torch
import copy
import pickle
import tqdm

from torch.utils.data import Dataset
from models.structures import Instances


NuScenesClasses = {
    'car' : 0,
    'pedestrian' : 1,
    'bicycle' : 2,
    'bus' : 3,
    'motorcycle' : 4,
    'trailer' : 5,
    'truck' : 6,
}


NuScenesAttributed = {
    'cycle.with_rider': 0,
    'cycle.without_rider': 1,
    'pedestrian.moving': 2,
    'pedestrian.standing': 3,
    'pedestrian.sitting_lying_down': 4,
    'vehicle.moving': 5,
    'vehicle.parked': 6,
    'vehicle.stopped': 7,
}


class NusceneseDataset(Dataset):
    def __init__(self, ann_file):
        super().__init__()
        self.stride = torch.tensor(0.8, dtype=torch.float32)
        self.point_range = torch.tensor(51.2, dtype=torch.float32)
        
        print("NuScene train open.")
        with open(ann_file, "rb") as handle:
            video_clips = pickle.load(handle)
        print("Convert instances")
        video_instances = []
        for seq in tqdm.tqdm(video_clips):
            seq_length = len(seq)
            seq_instances = []
            for frame_id in range(seq_length):
                frame_detected = seq[frame_id]
                translation = torch.from_numpy(frame_detected['translation'])
                size = torch.from_numpy(frame_detected['size'])
                yaw = torch.from_numpy(frame_detected['yaw']).unsqueeze(-1)
                rotation = torch.from_numpy(frame_detected['rotation'])
                velocity = torch.from_numpy(frame_detected['velocity'])
                score = torch.from_numpy(frame_detected['score']).unsqueeze(-1)
                classes = torch.from_numpy(frame_detected['classes']).unsqueeze(-1)
                tracking_id = torch.from_numpy(frame_detected['tracking_id']).unsqueeze(-1)
                scene_token = frame_detected['scene_token']
                sample_token = frame_detected['sample_token']
                timestamp = frame_detected['timestamp']

                times = torch.zeros([len(size), 1], dtype=torch.float64) + timestamp

                img_meta = {
                    "scene_token": scene_token,
                    "sample_token": sample_token,
                    "timestamp": timestamp,
                }

                instance_ = Instances([1, 1], img_meta)
                instance_.set("translation", translation)
                instance_.set("size", size)
                instance_.set("yaw", yaw)
                instance_.set("times", times)
                instance_.set("rotation", rotation)
                instance_.set("velocity", velocity)
                instance_.set("score", score)
                instance_.set("classes", classes)
                instance_.set("tracking_id", tracking_id.clone())
                instance_inds = torch.zeros_like(tracking_id) - 1
                instance_.set("instance_inds", instance_inds)
                instance_ = instance_[instance_.score[..., 0] >= 0.1]
                seq_instances.append(instance_)
            video_instances.append(seq_instances)
        print("Convert mini batch")
        clip_video = []
        for seq in tqdm.tqdm(video_instances):
            start_frame = 0
            end_frame = len(seq)
            clip_len_frames = 10
            for idx in range(start_frame, end_frame - 1, 3):
                clip_frame_ids = []
                for frame_idx in range(idx, idx + clip_len_frames):
                    if frame_idx < end_frame:
                        input_dict = seq[frame_idx]
                        if len(input_dict) > 0:
                            clip_frame_ids.append(input_dict)
                if len(clip_frame_ids) >= 10:
                    seq_times = [s._img_meta['timestamp'] for s in clip_frame_ids]
                    inverse_sequence_instances = copy.deepcopy(clip_frame_ids[::-1])
                    for iidx, inv_seq in enumerate(inverse_sequence_instances):
                        inv_seq.times = inv_seq.times.to(torch.float64)
                        inv_seq.times[inv_seq.times > 0] = torch.tensor(seq_times[iidx], dtype=torch.float64)
                        inv_seq._img_meta['timestamp'] = seq_times[iidx]
                    clip_video.append(copy.deepcopy(inverse_sequence_instances))
                    clip_video.append(copy.deepcopy(clip_frame_ids))
        self.data_infos = clip_video

    def __len__(self):
        return len(self.data_infos)

    def get_instance(self, sequence):
        return {
            "instances": sequence
        }

    def __getitem__(self, idx):
        idx = min(idx, len(self) - 1)
        sequence = self.data_infos[idx]
        return self.get_instance(sequence)


class NusceneseValDataset(Dataset):
    def __init__(self, ann_file):
        super().__init__()
        self.stride = 0.8
        self.point_range = 51.2

        with open(ann_file, "rb") as handle:
            video_clips = pickle.load(handle)

        video_instances = []
        for seq in video_clips:
            seq_length = len(seq)
            seq_instances = []
            for frame_id in range(seq_length):
                frame_detected = seq[frame_id]
                translation = torch.from_numpy(frame_detected['translation'])
                size = torch.from_numpy(frame_detected['size'])
                yaw = torch.from_numpy(frame_detected['yaw']).unsqueeze(-1)
                rotation = torch.from_numpy(frame_detected['rotation'])
                velocity = torch.from_numpy(frame_detected['velocity'])
                score = torch.from_numpy(frame_detected['score']).unsqueeze(-1)
                classes = torch.from_numpy(frame_detected['classes']).unsqueeze(-1)
                tracking_id = torch.from_numpy(frame_detected['tracking_id']).unsqueeze(-1)
                scene_token = frame_detected['scene_token']
                sample_token = frame_detected['sample_token']
                timestamp = frame_detected['timestamp']

                times = torch.zeros([len(size), 1], dtype=torch.float64) + timestamp

                img_meta = {
                    "scene_token": scene_token,
                    "sample_token": sample_token,
                    "timestamp": timestamp,
                }

                instance_ = Instances([1, 1], img_meta)
                instance_.set("translation", translation)
                instance_.set("size", size)
                instance_.set("yaw", yaw)
                instance_.set("times", times)
                instance_.set("rotation", rotation)
                instance_.set("velocity", velocity)
                instance_.set("score", score)
                instance_.set("classes", classes)
                instance_.set("tracking_id", tracking_id.clone())
                instance_inds = torch.zeros_like(tracking_id) - 1
                instance_.set("instance_inds", instance_inds)
                instance_ = instance_[instance_.score[..., 0] >= 0.1]
                seq_instances.append(instance_)
            video_instances.append(seq_instances)

        self.data_infos = video_instances

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        idx = min(idx, len(self) - 1)
        sequence = self.data_infos[idx]
        return [None, sequence]