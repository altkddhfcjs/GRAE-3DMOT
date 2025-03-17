import copy
import torch
import lap
import tqdm
import numpy as np

from fvcore.common.timer import Timer
from models.structures import Instances
from nuscenes.nuscenes import NuScenes


def get_color(N=40):
    from matplotlib import cm
    colours = cm.rainbow(np.linspace(0, 1, N))
    return colours
colors = get_color(40)

class BasePredictor:
    def __init__(self,
                 model,
                 device,
                 data_loader,
                 outputs="test.json"):
        self.device = device
        self.data_loader = data_loader
        self.model = model

        # tracking hyper-parameter
        self.iter_time = 0
        self.alpha = 0.24 # detection accuracy
        self.conf_th = 0.12 # mathcing threshold
        self.age = 12  # trajectory age
        
        self.outputs = outputs
        self.save_notmatched_track = False
        self.class_names = [
            'car', 'pedestrian', 'bicycle', 'bus', 'motorcycle', 'trailer', 'truck'
        ]
        self.total_time = 0
        self.nusc = NuScenes(version="v1.0-trainval", verbose=True, dataroot="../data/nuscenes")

    def setup_instance(self, inst):
        N = len(inst)

        temp_ = torch.zeros((N, self.model.d_model)).to(self.device)
        timestamp = torch.zeros((N,), dtype=torch.float64).to(self.device)
        age = torch.zeros((N,)).to(self.device)

        inst.set("size", inst.size.clone())
        inst.set("tgt", temp_.clone())
        inst.set("instance_feature", temp_.clone())
        inst.set("motion_feature", temp_.clone())
        inst.set("timestamp", timestamp)
        inst.set("time", torch.zeros((N,), dtype=torch.int64).to(self.device))
        inst.set("age", age)

        inst.set("ct", inst.translation[..., :2].clone())
        inst.set("pred_ct", inst.translation[..., :2].clone())
        inst.set("offset", inst.velocity[..., :2].clone())
        inst.set("ar", age.clone())

    def _progress(self, iteration):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = iteration * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = iteration
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def test(self):
        import json
        from eval.nusc_eval import eval_nusc_tracking
        print("Start inference...")
        result = self._test()

        json_output = self.outputs + ".json"
        with open(json_output, "w") as f:
            json.dump(result, f)

        metrics_summary = eval_nusc_tracking(
            json_output, 'val', "./", "../data/nuscenes",
            verbose=True,
            num_vis=0
        )

        mot_metrics = {
            "mot_metrics/amota": metrics_summary["amota"],
            "mot_metrics/amotp": metrics_summary["amotp"],
            "mot_metrics/motar": metrics_summary["motar"],
            "mot_metrics/mota": metrics_summary["mota"],
            "mot_metrics/motp": metrics_summary["motp"],
            "mot_metrics/recall": metrics_summary["recall"],

            "count_metrics/gt": metrics_summary["gt"],
            "count_metrics/mt": metrics_summary["mt"],
            "count_metrics/ml": metrics_summary["ml"],
            "count_metrics/tp": metrics_summary["tp"],
            "count_metrics/fp": metrics_summary["fp"],
            "count_metrics/fn": metrics_summary["fn"],
            "count_metrics/ids": metrics_summary["ids"],
            "count_metrics/frag": metrics_summary["frag"],
            "count_metrics/faf": metrics_summary["faf"],
            "time_metrics/tid": metrics_summary["tid"],
            "time_metrics/lgd": metrics_summary["lgd"],
        }

        txt_output = self.outputs + ".txt"
        with open(txt_output, "w") as file:
            file.write(str(mot_metrics))
        return mot_metrics

    def _test(self):
        nusc_annos = {
            "results": {},
            "meta": {
                "use_camera": False,
                "use_lidar": True,
                "use_radar": False,
                "use_map": False,
                "use_external": False,
            }
        }
        self.model.eval()
        for iteration, data_seq in tqdm.tqdm(enumerate(self.data_loader)):
            instances = [d.to(self.device) for d in data_seq[1][0]]
            for idx, instance in enumerate(instances):
                if len(instance) > 0:
                    self.setup_instance(instance)
                    times = torch.zeros([len(instance), 1], dtype=torch.float64) + instance._img_meta['timestamp']
                    times = times.to(self.device)
                    instance.set("time", times)

            with torch.no_grad():
                results = self._test_mini_seq(instances)
            nusc_annos['results'].update(results)
        return nusc_annos

    def spatial_feature(self, current_instance):
        score = current_instance.score.clone()
        instance_center = current_instance.translation.clone()
        instance_yaw = current_instance.rotation.clone()
        instance_size = current_instance.size.clone()

        label_feat = torch.nn.functional.one_hot(current_instance.classes[..., 0], 7)

        coordinate_info = torch.cat([instance_center, instance_yaw, instance_size, score, label_feat], dim=-1).to(
            torch.float32)

        # Construct spatial relation graph
        asso_coord = instance_center[:, None, :] - instance_center[None, ...]
        asso_size = instance_size[:, None, :] - instance_size[None, ...]
        asso_yaw = instance_yaw[:, None, :] - instance_yaw[None, ...]
        asso_labels = label_feat[:, None, :] - label_feat[None, ...]
        asso_score = score.expand(len(asso_yaw), -1, -1)

        spatial_dist = torch.norm((instance_center[:, None, :] - instance_center[None, ...]), dim=-1)
        spatial_dist = torch.sqrt(spatial_dist)
        asso_dist = spatial_dist.unsqueeze(-1)

        spatial_info = torch.cat([asso_coord, asso_yaw, asso_size, asso_score, asso_labels, asso_dist], dim=-1).to(
            torch.float32)
        return coordinate_info, spatial_info, spatial_dist

    def temporal_feature(self, current_instance, current_time):
        # Construct temporal relation graph
        det_center = current_instance.translation.clone()
        det_yaw = current_instance.rotation.clone()
        det_size = current_instance.size.clone()
        det_score = current_instance.score.clone()
        det_time = current_time - current_instance.time.clone()
        det_info = torch.cat([det_center, det_yaw, det_size, det_score, det_time], dim=-1).to(torch.float32)
        return det_info

    def _test_mini_seq(self, data_instances):
        track = []
        annos = {}
        tracked_instance_inds = 0

        for frame_id, dets in enumerate(data_instances):
            frame_anns = []
            sample_token = dets._img_meta['sample_token']
            if len(track) == 0:
                last_time_stamp = dets._img_meta['timestamp']
            time_lag = dets._img_meta['timestamp'] - last_time_stamp
            last_time_stamp = dets._img_meta['timestamp']
            current_time = torch.tensor(dets._img_meta['timestamp'], dtype=torch.float64).to(self.device)

            if len(track) == 0 and len(dets) > 0:
                dets = dets[dets.score[..., 0] > self.conf_th]
                dets.offset = dets.velocity.clone() * 0.5
                new_inds = torch.arange(0, len(dets), dtype=torch.int64).to(self.device)
                dets.instance_inds = new_inds.unsqueeze(-1)
                tracked_instance_inds += len(dets)

                coordinate_info, spatial_info, spatial_dist = self.spatial_feature(dets)

                current_time = torch.tensor(dets._img_meta['timestamp'], dtype=torch.float64).to(self.device)
                det_info = self.temporal_feature(dets, current_time)
                temporal_info = torch.zeros_like(det_info, dtype=torch.float32)
                coordinate_feature, instance_feature, motion_feature = self.model(
                    coordinate_info, spatial_info, spatial_dist, temporal_info, first_frame=True
                )
                dets.set("coord_features", coordinate_feature)
                dets.set("instance_feature", instance_feature)
                dets.set("motion_feature", motion_feature)

                track = copy.deepcopy(dets)
                track.age += 1
                for idx in range(len(dets)):
                    tracking_name = self.class_names[int(dets.classes[idx])]
                    current_outputs = {
                        "sample_token": sample_token,
                        "translation": dets.translation[idx].cpu().numpy().tolist(),
                        "size": dets.size[idx].cpu().numpy().tolist(),
                        "rotation": dets.rotation[idx].cpu().numpy().tolist(),
                        "velocity": dets.velocity[idx].cpu().numpy().tolist(),
                        "tracking_id": str(dets.instance_inds[idx].cpu().numpy()),
                        "tracking_name": tracking_name,
                        "tracking_score": float(dets.score[idx].cpu().numpy())
                    }
                    frame_anns.append(current_outputs)
            elif len(track) == 0 and len(dets) == 0:
                annos.update({sample_token: frame_anns})
                continue
            elif len(track) > 0 and len(dets) == 0:
                track = track[track.age < self.age]
                track.pred_ct = track.ct.clone() + track.offset.clone()
                track.ct = track.pred_ct.clone()
                track.age += 1
            elif len(track) > 0 and len(dets) > 0:
                track.pred_ct = track.ct.clone() + track.offset.clone()
                track.ct = track.pred_ct.clone()
                track.translation[..., :2] = track.ct.clone()
                dets.offset = dets.velocity.clone() * time_lag

                coordinate_info, spatial_info, spatial_dist = self.spatial_feature(dets)
                current_time = torch.tensor(dets._img_meta['timestamp'], dtype=torch.float64).to(self.device)
                det_info = self.temporal_feature(dets, current_time)
                track_info = self.temporal_feature(track, current_time)
                # construct temporal relation graph
                temporal_info = det_info[:, None, :] - track_info[None, ...]

                track_ct = track.ct.clone()
                det_ct = dets.ct.clone()
                temporal_dist = torch.norm(
                    (det_ct.reshape(1, -1, 2) - track_ct.reshape(-1, 1, 2)), dim=-1
                )
                temporal_dist = torch.sqrt(temporal_dist)
                tracked_feature = track.motion_feature.clone()

                coordinate_feature, instance_feature, motion_feature, \
                    motion_features, affinity_scores, attention_scores = self.model(
                    coordinate_info, spatial_info, spatial_dist,
                    temporal_info, temporal_dist, tracked_feature,
                    first_frame=False
                )

                dets.set("coord_features", coordinate_feature)
                dets.set("instance_feature", instance_feature)
                dets.set("motion_feature", motion_feature)

                # aux_attn_score = attention_scores[-1].mean(1).sum(-1).T
                # matching process
                affinity_score = affinity_scores[-1]
                affinity_score = torch.sigmoid(affinity_score[..., 0]).T

                affinity = affinity_score

                tracked_ct = track.ct
                detected_ct = dets.ct
                dist = torch.sum(
                    (tracked_ct.reshape(1, -1, 2) -
                     detected_ct.reshape(-1, 1, 2)) ** 2, dim=2
                )
                dist = torch.sqrt(dist)
                dist_affinity = torch.exp(-dist)
                invalid = (dets.classes.view(-1, 1) != track.classes.view(1, -1)) > 0
                affinity = affinity * 0.5 + dist_affinity * 0.5
                cost = affinity + -1e6 * invalid

                high_dets = dets[dets.score[..., 0] > self.alpha]
                low_dets = dets[dets.score[..., 0] <= self.alpha]

                high_affinity_score = cost[dets.score[..., 0] > self.alpha]
                low_affinity_score = cost[dets.score[..., 0] <= self.alpha]

                if len(high_dets) > 0:
                    high_cost = high_affinity_score

                    _, row_ind, col_ind = lap.lapjv(
                        1 - high_cost.detach().cpu().numpy(), extend_cost=True, cost_limit=0.9
                    )
                    track_instance_inds = track.instance_inds.clone()
                    det_instance_inds = torch.full([len(high_dets), 1], -2, dtype=torch.int64).to(self.device)
                    matches = []
                    for iy, my in enumerate(col_ind):
                        if my >= 0:
                            det_instance_inds[my] = track_instance_inds[iy]
                            track_instance_inds[iy] = -2
                            conf = high_cost[my, iy]
                            high_dets.motion_feature[my] = (track.motion_feature[iy] * (1 - conf) +
                                                            high_dets.motion_feature[my] * conf)
                            matches.append([my, iy])
                    high_dets.instance_inds = det_instance_inds
                    low_affinity_score = low_affinity_score[:, torch.where(track_instance_inds != -2)[0]]
                    track = track[track_instance_inds[..., 0] != -2]

                if len(track) > 0 and len(low_dets) > 0:
                    low_cost = low_affinity_score

                    _, row_ind, col_ind = lap.lapjv(
                        1 - low_cost.detach().cpu().numpy(), extend_cost=True, cost_limit=0.8
                    )
                    track_instance_inds = track.instance_inds.clone()
                    det_instance_inds = torch.full([len(low_dets), 1], -2, dtype=torch.int64).to(self.device)
                    matches = []
                    for iy, my in enumerate(col_ind):
                        if my >= 0:
                            det_instance_inds[my] = track_instance_inds[iy]
                            track_instance_inds[iy] = -2
                            conf = low_cost[my, iy]
                            low_dets.motion_feature[my] = (track.motion_feature[iy] * (1 - conf) +
                                                           low_dets.motion_feature[my] * conf)
                            matches.append([my, iy])
                    low_dets.instance_inds = det_instance_inds
                    track = track[track_instance_inds[..., 0] != -2]

                if len(high_dets) == 0 and len(low_dets) != 0:
                    dets = low_dets
                elif len(low_dets) == 0 and len(high_dets) != 0:
                    dets = high_dets
                else:
                    dets = Instances.cat([high_dets, low_dets], dets._img_meta)

                matched_dets = dets[dets.instance_inds[..., 0] > -1]
                unmatched_dets = dets[dets.instance_inds[..., 0] < 0]
                unmatched_dets = unmatched_dets[unmatched_dets.score[..., 0] > self.conf_th]

                unmatched_len = len(unmatched_dets)
                current_instance_ind = tracked_instance_inds
                new_inds = torch.arange(
                    current_instance_ind,
                    current_instance_ind + unmatched_len,
                    dtype=torch.int64
                ).to(self.device)
                unmatched_dets.instance_inds = new_inds.unsqueeze(-1)
                tracked_instance_inds += unmatched_len + 1

                dets = Instances.cat([matched_dets, unmatched_dets], matched_dets._img_meta)

                fresh = copy.deepcopy(track[track.age < 2])
                draw_track = fresh
                draw_track.ct = draw_track.pred_ct.clone()
                draw_track.translation[..., :2] = draw_track.ct.clone()
                s_idx = torch.sort(draw_track.score[..., 0], descending=True)[1]
                draw_track = draw_track[s_idx]
                d_length = 500 - len(dets)
                draw_track = draw_track[:d_length]

                for idx in range(len(draw_track)):
                    tracking_name = self.class_names[int(draw_track.classes[idx])]
                    current_outputs = {
                        "sample_token": sample_token,
                        "translation": draw_track.translation[idx].cpu().numpy().tolist(),
                        "size": draw_track.size[idx].cpu().numpy().tolist(),
                        "rotation": draw_track.rotation[idx].cpu().numpy().tolist(),
                        "velocity": draw_track.velocity[idx].cpu().numpy().tolist(),
                        "tracking_id": str(draw_track.instance_inds[idx].cpu().numpy()),
                        "tracking_name": tracking_name,
                        "tracking_score": float(draw_track.score[idx].cpu().numpy()) * 0.1,
                    }
                    frame_anns.append(current_outputs)

                track.ct = track.pred_ct.clone()
                track = Instances.cat([dets, track])
                track = track[track.age < self.age]
                track.age += 1

                for idx in range(len(dets)):
                    tracking_name = self.class_names[int(dets.classes[idx])]
                    current_outputs = {
                        "sample_token": sample_token,
                        "translation": dets.translation[idx].cpu().numpy().tolist(),
                        "size": dets.size[idx].cpu().numpy().tolist(),
                        "rotation": dets.rotation[idx].cpu().numpy().tolist(),
                        "velocity": dets.velocity[idx].cpu().numpy().tolist(),
                        "tracking_id": str(dets.instance_inds[idx].cpu().numpy()),
                        "tracking_name": tracking_name,
                        "tracking_score": float(dets.score[idx].cpu().numpy()),
                    }
                    frame_anns.append(current_outputs)
            annos.update({sample_token: frame_anns})

        return annos

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        checkpoint = torch.load(resume_path, map_location='cpu')

        load_model = checkpoint['state_dict']
        model_state = self.model.state_dict()
        for k, v in load_model.items():
            k_ = ".".join(k.split(".")[1:])
            model_state[k_] = v

        self.model.load_state_dict(model_state)
        self.model = self.model.to(self.device)
