from models.structures import Instances
from models.losses import L2LossTrack

import random
import torch
import copy
import lap

import torch.distributed as dist
from torchvision.ops import sigmoid_focal_loss


class BaseTrainer:
    def __init__(self,
                 model,
                 optimizer,
                 dataloader,
                 sampler,
                 metrics,
                 timer,
                 checkpoint_dir,
                 opts,
                 log_step,
                 writer,
                 logger,
                 device):
        self.device = 'cuda:{}'.format(device)
        self.model = model

        self.optimizer = optimizer
        self.train_dataloader = dataloader
        self.train_metrics = metrics
        self.train_sampler = sampler
        self.start_epoch = 1
        self.end_epoch = 25
        self.global_step_time = timer

        self.checkpoint_dir = checkpoint_dir
        self.opts = opts
        self.log_step = log_step
        self.writer = writer
        self.logger=logger
        self.loss_track = L2LossTrack(
            neg_pos_ub=3,
            pos_margin=0,
            neg_margin=0.1,
            hard_mining=True,
            reduction='mean',
            loss_weight=1.0
        )
        self.global_rank = dist.get_rank()
        # self.opts.rank = 0

    def run(self):
        iter_time = 0
        len_epoch = len(self.train_dataloader)

        for epoch in range(self.start_epoch, self.end_epoch):
            self.model.train()
            self.train_metrics.reset()
            self.train_sampler.set_epoch(epoch)

            for iteration, data in enumerate(self.train_dataloader):
                # self.global_step_time.reset()
                # data = data.cuda(async_=False)
                loss_dict = self.train(data)
                total_loss = torch.stack(list(loss_dict.values()), dim=0).sum()

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                if self.global_rank == 0:
                    self.writer.set_step((epoch - 1) * len_epoch + iteration)
                    log_str = f'Total loss: {total_loss}, '
                    for k, v in loss_dict.items():
                        self.train_metrics.update(k, v)
                        log_str += '{}: {:.4f}, '.format(k, v)
                    self.train_metrics.update('loss/total', total_loss)

                    if (iteration + 1) % self.log_step == 0 or (iteration + 1) == len_epoch:
                        self.logger.debug('Train Epoch: {} {} '.format(
                            epoch, _progress(iteration + 1, len_epoch)) + log_str)
                    if iteration == len_epoch:
                        break

            if self.global_rank == 0:
                iter_time += 1
                result = self.train_metrics.result()
                log = {'epoch': epoch}
                log.update(result)
                for key, value in log.items():
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

            if self.opts.rank == 0:
                arch = type(self.model).__name__
                state = {
                    'arch': arch,
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }
                filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
                torch.save(state, filename)
                self.logger.info("Saving checkpoint: {} ...".format(filename))

    def spatial_feature(self, current_instance):
        score = current_instance.score.clone()
        instance_center = current_instance.translation.clone()
        instance_yaw = current_instance.rotation.clone()
        instance_size = current_instance.size.clone()

        label_feat = torch.nn.functional.one_hot(current_instance.classes[..., 0], 7)

        coordinate_info = torch.cat([instance_center, instance_yaw, instance_size, score, label_feat], dim=-1).to(torch.float32)

        # association feature
        asso_coord = instance_center[:, None, :] - instance_center[None, ...]
        asso_size = instance_size[:, None, :] - instance_size[None, ...]
        asso_yaw = instance_yaw[:, None, :] - instance_yaw[None, ...]
        asso_labels = label_feat[:, None, :] - label_feat[None, ...]
        asso_score = score.expand(len(asso_yaw), -1, -1)

        spatial_dist = torch.norm((instance_center[:, None, :] - instance_center[None, ...]), dim=-1)
        spatial_dist = torch.sqrt(spatial_dist)
        asso_dist = spatial_dist.unsqueeze(-1)

        spatial_info = torch.cat([asso_coord, asso_yaw, asso_size, asso_score, asso_labels, asso_dist], dim=-1).to(torch.float32)
        return coordinate_info, spatial_info, spatial_dist

    def temporal_feature(self, current_instance, current_time):
        det_center = current_instance.translation.clone()
        det_yaw = current_instance.rotation.clone()
        det_size = current_instance.size.clone()
        det_score = current_instance.score.clone()
        det_time = current_time - current_instance.time.clone()
        det_info = torch.cat([det_center, det_yaw, det_size, det_score, det_time], dim=-1).to(torch.float32)
        return det_info

    def train(self, data_instances):
        device = self.device
        data_instances = data_instances['instances']

        ori_len = len(data_instances)
        start_frame = random.randint(0, ori_len - 8)
        sequence_instances = data_instances[start_frame:start_frame + 8]
        sequence_instances = [d.to(device) for d in sequence_instances]
        seq_length = len(sequence_instances)

        self.model.eval()
        with (torch.no_grad()):
            frame_id = 0
            det_instance = sequence_instances[frame_id]
            times = torch.zeros([len(det_instance), 1], dtype=torch.float64) + det_instance._img_meta['timestamp']
            times = times.to(device)
            det_instance.set("time", times)
            det_instance.set("ct", det_instance.translation[..., :2].clone())
            instance_ids = torch.zeros_like(det_instance.tracking_id) - 1
            det_instance.set("instance_ids", instance_ids)

            current_instance = det_instance

            ##############################
            score = current_instance.score
            age = torch.zeros_like(score.clone())
            current_instance.set("age", age)

            coordinate_info, spatial_info, spatial_dist = self.spatial_feature(current_instance)

            current_time = torch.tensor(current_instance._img_meta['timestamp'], dtype=torch.float64).to(device)
            det_info = self.temporal_feature(current_instance, current_time)
            temporal_info = torch.zeros_like(det_info, dtype=torch.float32)
            current_instance.set("motion_info", temporal_info)

            coordinate_feature, instance_feature, motion_feature = self.model(
                coordinate_info, spatial_info, spatial_dist, temporal_info, first_frame=True
            )

            current_instance.set("coord_features", coordinate_feature)
            current_instance.set("instance_feature", instance_feature)
            current_instance.set("motion_feature", motion_feature)
            tracked = copy.deepcopy(current_instance)
            tracked.age += 1
        self.model.train()

        losses = dict()

        for frame_id in range(1, seq_length):
            det_instance = sequence_instances[frame_id]
            instance_ids = torch.zeros_like(det_instance.tracking_id) - 1
            det_instance.set("instance_ids", instance_ids)
            times = torch.zeros([len(det_instance), 1], dtype=torch.float64) + det_instance._img_meta['timestamp']
            times = times.to(device)
            det_instance.set("time", times)
            det_instance.set("ct", det_instance.translation[..., :2].clone())

            current_instance = det_instance
            # det
            score = current_instance.score
            age = torch.zeros_like(score.clone())
            current_instance.set("age", age)

            coordinate_info, spatial_info, spatial_dist = self.spatial_feature(current_instance)
            current_time = torch.tensor(current_instance._img_meta['timestamp'], dtype=torch.float64).to(device)
            det_info = self.temporal_feature(current_instance, current_time)
            track_info = self.temporal_feature(tracked, current_time)

            temporal_info = det_info[:, None, :] - track_info[None, ...]

            track_ct = tracked.ct.clone()
            det_ct = current_instance.ct.clone()
            temporal_dist = torch.norm(
                (det_ct.reshape(1, -1, 2) - track_ct.reshape(-1, 1, 2)), dim=-1
            )
            temporal_dist = torch.sqrt(temporal_dist)
            tracked_feature = tracked.motion_feature.clone()

            (coordinate_feature, instance_feature, motion_feature,
             motion_features, affinity_scores, attention_scores) = \
                self.model(coordinate_info, spatial_info, spatial_dist,
                           temporal_info, temporal_dist, tracked_feature,
                           first_frame=False)

            current_instance.set("coord_features", coordinate_feature)
            current_instance.set("instance_feature", instance_feature)
            current_instance.set("motion_feature", motion_feature)

            # loss
            key_num = len(current_instance)
            ref_num = len(tracked)
            embed_targets = torch.zeros((key_num, ref_num))
            pos2pos = (current_instance.tracking_id[..., 0].view(-1, 1) ==
                       tracked.tracking_id[..., 0].view(1, -1))
            embed_targets[:, :pos2pos.size(1)] = pos2pos
            track_cond = tracked.tracking_id[..., 0] < 0
            track_cond = track_cond.expand(key_num, -1)
            embed_targets[track_cond] = 0.
            embed_targets = embed_targets.to(device)
            embed_targets = embed_targets.T.unsqueeze(-1)

            affinity_loss_list = []
            attention_loss_list = []
            for lvl in range(len(attention_scores)):
                ori_affinity_score = affinity_scores[lvl].clone()
                pred = ori_affinity_score[..., 0]
                target = embed_targets[..., 0].clone()
                affinity_loss = sigmoid_focal_loss(pred, target, alpha=-1, gamma=1.0, reduction='mean')
                affinity_loss_list.append(affinity_loss)

                attn_score = attention_scores[lvl].clone()
                pred = attn_score.mean(1)[..., 0]
                target = embed_targets[..., 0].clone()
                attention_loss = self.loss_track(pred, target)
                attention_loss_list.append(attention_loss * 0.1)

            frame_loss_dict = dict()
            frame_loss_dict['loss_affinity'] = affinity_loss_list[-1]
            frame_loss_dict['loss_attention'] = attention_loss_list[-1]
            num_dec_layer = 0
            for loss_i in range(len(affinity_loss_list[:-1])):
                frame_loss_dict[f'd{num_dec_layer}.loss_affinity'] = affinity_loss_list[loss_i]
                frame_loss_dict[f'd{num_dec_layer}.loss_attention'] = attention_loss_list[loss_i]
                num_dec_layer += 1

            with torch.no_grad():
                affinity_sigmoid = torch.sigmoid(affinity_scores[-1][..., 0]).T
                tracked_ct = tracked.ct + tracked.velocity * 0.5
                tracked.ct = tracked_ct
                tracked.translation[..., :2] = tracked.ct.clone()
                invalid = (current_instance.classes.view(-1, 1) != tracked.classes.view(1, -1)) > 0
                cost = affinity_sigmoid + -1e6 * invalid
                _, row_ind, col_ind = lap.lapjv(
                    1 - cost.detach().cpu().numpy(), extend_cost=True, cost_limit=0.95
                )
                tracked = Instances.cat([current_instance, tracked[col_ind == -1]], current_instance._img_meta)
            for key, value in frame_loss_dict.items():
                losses['frame_' + str(frame_id) + "_" + key] = value
        return losses


def _progress(iteration, len_epoch):
    base = '[{}/{} ({:.0f}%)]'
    # if hasattr(data_loader, 'n_samples'):
    #     current = iteration * data_loader.batch_size
    #     total = data_loader.n_samples
    # else:
    current = iteration
    total = len_epoch
    return base.format(current, total, 100.0 * current / total)
