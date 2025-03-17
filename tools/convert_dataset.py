import torch
import numpy as np
import pickle
import json
import tqdm
import iou3d_nms_cuda
from scipy.optimize import linear_sum_assignment
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.core.bbox import get_box_type
from nuscenes.eval.common.utils import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.eval.tracking.utils import category_to_tracking_name

from SimpleTrack.data_loader.nuscenes_loader import nu_array2mot_bbox
from mot_3d.preprocessing import nms

CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

NuScenesClasses = {
    'car' : 0,
    'pedestrian' : 1,
    'bicycle' : 2,
    'bus' : 3,
    'motorcycle' : 4,
    'trailer' : 5,
    'truck' : 6,
}

box_type_3d = 'LIDAR'
box_mode_3d = get_box_type(box_type_3d)

def simpletrack_nms(frame_det_data, iou_threshold=0.1):
    boxes = np.concatenate([frame_det_data['translation'],
                            frame_det_data['size'],
                            frame_det_data['rotation'],
                            np.expand_dims(frame_det_data['score'], axis=1)],
                            axis=1)
    classes = frame_det_data['classes']
    boxes_mot = [nu_array2mot_bbox(b) for b in boxes]

    index, _ = nms(boxes_mot, classes, iou_threshold)

    return index


def transform_reference_points(reference_points, egopose, reverse=False, translation=True):
    reference_points = torch.cat([reference_points, torch.ones_like(reference_points[..., 0:1])], dim=-1)
    if reverse:
        matrix = egopose.inverse()
    else:
        matrix = egopose
    if not translation:
        matrix[..., :3, 3] = 0.0
    reference_points = (matrix.unsqueeze(1) @ reference_points.unsqueeze(-1)).squeeze(-1)[..., :3]
    return reference_points


def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix


def convert_egopose_to_matrix_numpy(rotation, translation):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix


def convert_detection_result(nusc, scenes, sequences_by_name, m='train'):
    center_point_det_val = json.load(open("center_point_det/{}.json".format(m)))
    detection_result = center_point_det_val['results']
    result = []
    for scene_id, scene_name in enumerate(tqdm.tqdm(scenes)):
        scene = sequences_by_name[scene_name]
        first_token = scene['first_sample_token']
        last_token = scene['last_sample_token']

        current_token = first_token

        scene_results = []
        tracking_id_set = set()

        while True:
            current_det_result = detection_result[current_token]
            current_sample = nusc.get('sample', current_token)

            scene_token = current_sample['scene_token']
            sample_token = current_token
            next_sample_token = current_sample['next']

            lidar_top_data = nusc.get('sample_data', current_sample['data']['LIDAR_TOP'])
            ego_pose = nusc.get('ego_pose', lidar_top_data['ego_pose_token'])
            cs_record = nusc.get(
                'calibrated_sensor',
                lidar_top_data['calibrated_sensor_token']
            )

            lidar_r = cs_record['rotation']
            lidar_t = cs_record['translation']

            global_r = ego_pose['rotation']
            global_t = ego_pose['translation']
            timestamp = ego_pose['timestamp'] / 1e6

            """
            Ground truth
            """
            frame_ann_tokens = current_sample['anns']
            gt_trans = []
            gt_size = []
            gt_yaw = []
            gt_rot = []
            gt_class = []
            gt_track_token = []
            gt_attribute = []

            gt_next_exist = []
            gt_next_trans = []
            gt_next_size = []
            gt_next_yaw = []
            gt_next_lidar = []
            gt_velocity = []

            _, box_lidar_nusc, _ = nusc.get_sample_data(lidar_top_data['token'])
            gt_lidar_boxes = {}
            for b in box_lidar_nusc:
                gt_lidar_boxes[b.token] = b

            gt_box_lidar = []
            for ann_idx, ann_token in enumerate(frame_ann_tokens):
                ann = nusc.get('sample_annotation', ann_token)
                tracking_name = category_to_tracking_name(ann['category_name'])
                if tracking_name is not None:
                    instance_token = ann['instance_token']
                    tracking_id_set.add(instance_token)

                    if len(ann['attribute_tokens']) == 0:
                        current_attr = None
                    else:
                        current_attr = nusc.get('attribute', ann['attribute_tokens'][0])['name']
                    gt_attribute.append(current_attr)
                    gt_trans.append(ann['translation'])
                    gt_size.append(ann['size'])
                    gt_yaw.append([quaternion_yaw(Quaternion(ann['rotation']))])
                    gt_rot.append(ann['rotation'])
                    gt_class.append(NuScenesClasses[tracking_name])
                    gt_track_token.append(instance_token)
                    gt_box_lidar.append(gt_lidar_boxes[ann_token])
                    next_ann_token = ann['next']
                    # get next frame translation information
                    if next_ann_token == "":
                        gt_next_exist.append(False)
                        gt_next_lidar.append(None)
                        gt_next_trans.append([0.0, 0.0, 0.0])
                        gt_next_size.append([0.0, 0.0, 0.0])
                        gt_next_yaw.append([0.0])
                        gt_velocity.append([0.0, 0.0, 0.0])
                    else:
                        gt_next_exist.append(True)
                        next_ann = nusc.get('sample_annotation', next_ann_token)

                        next_sample = nusc.get('sample', next_ann['sample_token'])
                        next_lidar = nusc.get('sample_data', next_sample['data']['LIDAR_TOP'])

                        next_lidar_box = nusc.get_sample_data(next_lidar['token'],
                                                              selected_anntokens=[next_ann['token']])[1]
                        gt_next_lidar.append(next_lidar_box[0])
                        gt_next_trans.append(next_ann['translation'])
                        gt_next_size.append(next_ann['size'])
                        gt_next_yaw.append([quaternion_yaw(Quaternion(next_ann['rotation']))])
                        gt_velocity.append(nusc.box_velocity(ann['token']).tolist())

            frame_anns_dict = {
                'translation': np.array(gt_trans, dtype=np.float32), # [M, 3]
                'size': np.array(gt_size, dtype=np.float32), # [M, 3]
                'yaw': np.array(gt_yaw, dtype=np.float32), # [M, 1]
                'rotation': np.array(gt_rot, dtype=np.float32), # [M, 4]
                'class': np.array(gt_class, dtype=np.int32), # [M]
                'tracking_id': gt_track_token, # [M]
                'next_exist': np.array(gt_next_exist, dtype=np.bool), # [M]
                'next_translation': np.array(gt_next_trans, dtype=np.float32), # [M, 3]
                'next_size': np.array(gt_next_size, dtype=np.float32), # [M, 3]
                'next_yaw': np.array(gt_next_yaw, dtype=np.float32), # [M, 1]
                'box_lidar': gt_box_lidar,
                "gt_velocity": gt_velocity,
                "gt_next_lidar": gt_next_lidar,
                "gt_attribute": gt_attribute
            }
            """
            Ground truth done.
            """

            """
            Prediction
            """
            pred_translation = [r['translation'] for r in current_det_result]
            pred_size = [r['size'] for r in current_det_result]
            pred_yaw = [quaternion_yaw(Quaternion(r['rotation'])) for r in current_det_result]
            pred_rotation = [r['rotation'] for r in current_det_result]
            pred_velocity = [r['velocity'] for r in current_det_result]
            pred_score = [r['detection_score'] for r in current_det_result]
            pred_attribute = [r['attribute_name'] for r in current_det_result]
            pred_classes = [NuScenesClasses[r['detection_name']] for r in current_det_result if
                            r['detection_name'] in NuScenesClasses]

            pred_mask = [r['detection_name'] in NuScenesClasses for r in current_det_result]

            frame_pred_dict = {
                "translation": np.array(pred_translation)[pred_mask],
                "size": np.array(pred_size)[pred_mask],
                "yaw": np.array(pred_yaw)[pred_mask],
                "rotation": np.array(pred_rotation)[pred_mask],
                "velocity": np.array(pred_velocity)[pred_mask],
                "score": np.array(pred_score)[pred_mask],
                "attribute": np.array(pred_attribute)[pred_mask],
                "classes": np.array(pred_classes),
            }
            index = simpletrack_nms(frame_pred_dict, iou_threshold=0.1)

            for ann_field in frame_pred_dict:
                frame_pred_dict[ann_field] = frame_pred_dict[ann_field][index]

            """
            Prediction done.
            """

            gt_tensor_boxes = torch.cat([
                torch.tensor(frame_anns_dict['translation']),
                torch.tensor(frame_anns_dict['size']),
                torch.tensor(frame_anns_dict['yaw'])
            ], dim=-1).view(-1, 7).to(torch.float32)

            pred_tensor_boxes = torch.cat([
                torch.tensor(frame_pred_dict['translation']),
                torch.tensor(frame_pred_dict['size']),
                torch.tensor(frame_pred_dict['yaw']).unsqueeze(1)
            ], dim=-1).view(-1, 7).to(torch.float32)


            gt_cls = torch.tensor(frame_anns_dict['class'], dtype=torch.int32)
            det_cls = torch.tensor(frame_pred_dict['classes'], dtype=torch.int32)

            cls_valid_mask = torch.eq(det_cls.unsqueeze(1), gt_cls.unsqueeze(0))

            iou = torch.FloatTensor(torch.Size((pred_tensor_boxes.shape[0], gt_tensor_boxes.shape[0]))).zero_()
            iou3d_nms_cuda.boxes_iou_bev_cpu(pred_tensor_boxes.contiguous(), gt_tensor_boxes.contiguous(), iou)

            iou_valid_mask = iou > 0

            valid_mask = torch.logical_and(cls_valid_mask, iou_valid_mask)
            invalid_mask = torch.logical_not(valid_mask)
            cost = - iou + 1e18 * invalid_mask

            cost[cost > 1e16] = 1e18

            row_ind, col_ind = linear_sum_assignment(cost)

            matches = []
            for i, j in zip(row_ind, col_ind):
                if cost[i, j] < 1e16:
                    matches.append([i, j])

            frame_result = {'prediction': frame_pred_dict,
                            'ground_truth': frame_anns_dict,
                            'num_dets': pred_tensor_boxes.shape[0], # int: N
                            'num_gts': gt_tensor_boxes.shape[0], # int: M
                            'scene_token': scene_token,
                            'sample_token': sample_token,
                            'timestamp': timestamp,
                            'matches': matches,
                            'l2e_t': lidar_t,
                            'l2e_r': lidar_r,
                            'e2g_t': global_t,
                            'e2g_r': global_r
                            }
            scene_results.append(frame_result)
            if current_token == last_token:
                break
            next_token = current_sample['next']
            current_token = next_token

        assert len(scene_results) == scene['nbr_samples']
        ## Convert instance token to tacking id for the whole scene
        tracking_token_to_id = {}
        for i, tracking_id in enumerate(tracking_id_set):
            tracking_token_to_id.update({tracking_id: i + scene_id * 1000})

        for frame_result in scene_results:
            for i, tracking_token in enumerate(frame_result['ground_truth']['tracking_id']):
                tracking_id = tracking_token_to_id[tracking_token]
                frame_result['ground_truth']['tracking_id'][i] = tracking_id
            # list to numpy
            frame_result['ground_truth']['tracking_id'] = \
                np.array(frame_result['ground_truth']['tracking_id'], dtype=np.int32)

        first_result = scene_results[0]
        first_lidar_t = first_result['l2e_t']
        first_lidar_r = first_result['l2e_r']

        first_global_t = first_result['e2g_t']
        first_global_r = first_result['e2g_r']

        first_global_r = Quaternion(first_global_r)
        first_lidar_r = Quaternion(first_lidar_r)

        first_global_r_mat = first_global_r.rotation_matrix
        first_lidar_r_mat = first_lidar_r.rotation_matrix

        predictions_ = []
        for frame_result in scene_results:
            match = frame_result['matches']
            ground_truth = frame_result['ground_truth']

            gt_next_exist = ground_truth['next_exist']
            get_next_trans = torch.tensor(ground_truth['next_translation'])
            get_next_trans = get_next_trans.view(get_next_trans.size(0), 3)

            detections = frame_result['prediction']
            num_det = len(detections['translation'])

            matches = np.array(match) #.view(-1, 2).long()

            prediction_tracking_id = - np.ones(num_det, dtype=np.int)
            gt_tracking_id = np.array(ground_truth['tracking_id'])

            if len(matches) > 0:
                matched_tracking_id = gt_tracking_id[matches[:, 1]]
                prediction_tracking_id[matches[:, 0]] = matched_tracking_id

            if len(matches) > 0:
                matched_gt_next_exist = np.array(gt_next_exist[matches[:, 1]])
                prediction_next_exist = np.zeros(num_det, dtype=np.bool)
                prediction_next_exist[matches[:, 0]] = matched_gt_next_exist

            detections['e2g_r'] = frame_result['e2g_r']
            detections['l2e_r'] = frame_result['l2e_r']
            detections['e2g_t'] = frame_result['e2g_t']
            detections['l2e_t'] = frame_result['l2e_t']

            num_gts = frame_result['num_gts']

            if num_gts > 0 and len(matches) > 0:
                gt_next_trans = np.array(ground_truth['next_translation'])
                gt_next_trans = gt_next_trans - np.array(first_global_t)
                gt_next_trans = gt_next_trans @ np.linalg.inv(first_global_r_mat).T
                gt_next_trans = gt_next_trans - np.array(first_lidar_t)
                gt_next_trans = gt_next_trans @ np.linalg.inv(first_lidar_r_mat).T

            pred_trans = np.array(detections['translation'])
            pred_trans = pred_trans - np.array(first_global_t)
            pred_trans = pred_trans @ np.linalg.inv(first_global_r_mat).T
            pred_trans = pred_trans - np.array(first_lidar_t)
            pred_trans = pred_trans @ np.linalg.inv(first_lidar_r_mat).T

            pred_orientation = [Quaternion(r) / first_global_r / first_lidar_r for r in detections['rotation']]

            pred_velocity = np.array([np.array([*v, 0.0]) for v in detections['velocity']])
            pred_velocity = [np.dot(v, np.linalg.inv(first_global_r_mat).T) for v in pred_velocity]    # velocity convert global to lidar
            pred_velocity = [np.dot(v, np.linalg.inv(first_lidar_r_mat).T) for v in pred_velocity]

            loc = pred_trans
            dim = np.array([s for s in detections['size']])
            yaw = np.array([r.yaw_pitch_roll[0:1] for r in pred_orientation])
            pred_velocities = np.array(pred_velocity)

            boxes_3d = np.concatenate([loc, dim[:, [1, 0, 2]], yaw, pred_velocities[:, :2]], axis=1)

            velo_target = np.zeros([num_det, 3], dtype=np.float64)
            velo_mask = np.zeros([num_det, ], dtype=bool)

            if num_gts > 0 and len(matches) > 0:
                velo_target[matches[:, 0]] = \
                    np.array(gt_next_trans[matches[:, 1]] -
                             pred_trans[matches[:, 0]], dtype=np.float64) * 2.0
                velo_mask[matches[:, 0]] = np.array(ground_truth['next_exist'][matches[:, 1]])

            # prediction target
            detections['velo_target'] = velo_target
            detections['velo_mask'] = velo_mask
            detections['boxes_3d'] = boxes_3d
            detections['tracking_id'] = prediction_tracking_id

            # prediction lidar
            detections['pred_lidar_translation'] = pred_trans
            detections['pred_lidar_orientation'] = pred_orientation
            detections['pred_lidar_velocity'] = pred_velocity

            gt_next_translation = np.zeros([num_det, 3], dtype=np.float64)
            if num_gts > 0 and len(matches) > 0:
                gt_next_translation[matches[:, 0]] = np.array(gt_next_trans[matches[:, 1]], dtype=np.float64)
                detections['gt_next_translation'] = gt_next_translation

            gt_translation_target = np.zeros([num_det, 3], dtype=np.float64)
            if num_gts > 0 and len(matches) > 0:
                gt_trans = np.array(ground_truth['translation'])
                gt_trans = gt_trans - np.array(first_global_t) #g2e
                gt_trans = gt_trans @ np.linalg.inv(first_global_r_mat).T
                gt_trans = gt_trans - np.array(first_lidar_t)
                gt_trans = gt_trans @ np.linalg.inv(first_lidar_r_mat).T
                gt_translation_target[matches[:, 0]] = gt_trans[matches[:, 1]]

            gt_velo_targets = np.zeros([num_det, 3], dtype=np.float64)
            if len(ground_truth['gt_velocity']) > 0 and len(matches) > 0:
                gt_velocity = np.array([np.array(v[:2] + [0.0]) for v in ground_truth['gt_velocity']])
                gt_velocity = [v @ np.linalg.inv(first_global_r_mat).T for v in gt_velocity]
                gt_velocity = np.array([v @ np.linalg.inv(first_lidar_r_mat).T for v in gt_velocity])
                gt_velo_targets[matches[:, 0]] = gt_velocity[matches[:, 1]]

            gt_attribute_targets = np.full([num_det], None)
            if len(ground_truth['gt_attribute']) > 0 and len(matches) > 0:
                gt_attribute_targets[matches[:, 0]] = np.array(ground_truth['gt_attribute'])[matches[:, 1]]

            detections['gt_attribute'] = gt_attribute_targets
            detections['gt_lidar_translation'] = gt_translation_target
            detections['gt_lidar_velocity'] = gt_velo_targets
            detections['scene_token'] = frame_result['scene_token']
            detections['sample_token'] = frame_result['sample_token']
            detections['timestamp'] = frame_result['timestamp']
            predictions_.append(detections)
        # save sequence prediction
        result.append(predictions_)
    return result

if __name__ == '__main__':
    mod = "train"
    print("Loading NuScenes")
    nusc = NuScenes(version="v1.0-trainval", dataroot="data/nuscenes", verbose=True)
    sequences_by_name = {scene["name"]: scene for scene in nusc.scene}
    splits_to_scene_names = create_splits_scenes()
    train_scenes = splits_to_scene_names[mod]

    data = convert_detection_result(nusc, train_scenes, sequences_by_name, mod)
    print("Loading prediction results")

    with open("grae_{}.pickle".format(mod), "wb") as f:
        pickle.dump(data, f)
    print("centerpoint_{}_instances.pickle".format(mod))