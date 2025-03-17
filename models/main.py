import random
import lap
import torch

from .structures import Instances
from .utils import *
from .losses import L2LossTrack


class GRAE(nn.Module):
    def __init__(self,
                 in_channels=256,
                 layers=3,
                 device="cuda:0"
                 ):
        super().__init__()
        self.d_model = in_channels
        self.delta = 5  # track age
        self.layers = layers

        self.point_range = 51.2
        self.track_weight = 2.0
        self.device = device

        self.loss_aux_track = L2LossTrack(
            neg_pos_ub=3,
            pos_margin=0,
            neg_margin=0.1,
            hard_mining=True,
            reduction='mean',
            loss_weight=1.0
        )

        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.coord_proj = MLP(18, self.d_model // 4, self.d_model)
        self.spatial_proj = MLP(19, self.d_model // 4, self.d_model)

        self.spatial_dffl = DFFL(self.d_model, 1, self.d_model)
        self.spatial_w = nn.Linear(self.d_model, 1)

        self.spatial_proj2 = nn.Linear(self.d_model, self.d_model)

        self.encoder_layer = nn.Linear(self.d_model, self.d_model)

        self.encoder = Encoder(
            in_dim=self.d_model,
            out_dim=self.d_model,
            dim_feedforward=self.d_model // 4,
            num_layer=1
        )
        self.spatial_proj3 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )

        self.temporal_proj = MLP(12, self.d_model // 4, self.d_model)
        self.temporal_dffl = DFFL(self.d_model, 1, self.d_model)
        self.temporal_proj2 = nn.Linear(self.d_model, self.d_model)

        self.gen_tau = nn.Linear(self.d_model, 8)
        self.affinity_decoder = Decoder(
            in_dims=self.d_model,
            out_dims=self.d_model,
            dim_feedforward=self.d_model // 4,
            num_layer=self.layers
        )

        temporal_fusion = MLP(self.d_model * 2, self.d_model, self.d_model)
        self.temporal_fusion = nn.ModuleList(
            [temporal_fusion for _ in range(self.layers)])

        traj_proj = MLP(
            self.d_model,
            dim_forward=self.d_model // 4,
            out_model=self.d_model
        )
        self.traj_proj = nn.ModuleList(
            [traj_proj for _ in range(self.layers)])

        traj_proj2 = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )
        self.traj_proj2 = nn.ModuleList(
                [traj_proj2 for _ in range(self.layers)])

        affinity = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, 1)
        )
        self.affinity = nn.ModuleList(
                [affinity for _ in range(self.layers)])

    def init_weights(self):
        """Initialize weights of the transformer head."""
        self._is_init = True
        nn.init.zeros_(self.spatial_w.weight)
        nn.init.uniform_(self.spatial_w.bias, 0.0, 1.0)
        self.encoder.init_weights()
        self.affinity_decoder.init_weights()

    def forward_spatial(self, coordinate_info, spatial_info, spatial_dist):
        """
        process spatial relation-aware encoder
        """
        coordinate_feature = self.coord_proj(coordinate_info) # geometry feature
        
        # spatial realation-aware encoder
        spatial_distance = spatial_dist.unsqueeze(-1).to(torch.float32)        
        spatial_feature = self.spatial_proj(spatial_info)
        spatial_feature = self.spatial_dffl(spatial_feature, spatial_distance)
        asso_weight = self.spatial_w(spatial_feature)

        det_asso_feature = spatial_feature
        det_asso_w = torch.softmax(asso_weight, dim=1)

        spatial_feature = torch.sum(det_asso_feature * det_asso_w, dim=1)
        spatial_feature = self.spatial_proj2(spatial_feature)

        # construct graph
        asso_features = 2 * spatial_feature[:, None, :] - spatial_feature[None, ...]
        asso_features = self.encoder_layer(asso_features)

        attn_mask = torch.zeros_like(spatial_dist[None, None, ...])

        # MHSA
        asso_features, w = self.encoder(asso_features, attn_mask=attn_mask)
        asso_features = torch.sum(asso_features, dim=1)

        instance_feature = asso_features + coordinate_feature
        instance_feature = self.spatial_proj3(instance_feature)
        return coordinate_feature, instance_feature, det_asso_w, w

    def forward_temporal(self, instance_feature, temporal_info, temporal_dist=None, tracked_feature=None):
        """
        Process temporal relation-aware encoder and association process
        """
        # temporal relation-aware encoder
        if tracked_feature is None:
            return instance_feature

        temporal_feature1 = self.temporal_proj(temporal_info)
        temporal_distance = temporal_dist.transpose(0, 1).unsqueeze(-1).to(torch.float32)
        temporal_feature1 = self.temporal_dffl(temporal_feature1, temporal_distance)

        temporal_feature2 = instance_feature[:, None, :] + temporal_feature1
        temporal_feature2 = self.temporal_proj2(temporal_feature2)
        temporal_feature2 = temporal_feature2.permute(1, 0, 2)
        temporal_feature1 = temporal_feature1.permute(1, 0, 2)

        attn_dist = -temporal_dist
        attn_dist = attn_dist.unsqueeze(0)
        attn_dist = attn_dist.permute(1, 0, 2)
        attn_mask = torch.zeros_like(attn_dist[..., None])

        tracked_feature = tracked_feature.unsqueeze(1)

        intermediate = []
        attention_scores = []
        affinity_scores = []
        motion_features = []

        # association process
        lvl = 0
        for layer in self.affinity_decoder.layers:
            tgt, attn = layer(temporal_feature2, tracked_feature, attn_mask.transpose(1, 3))
            tgt = torch.cat([tgt, temporal_feature1], dim=-1)
            temporal_feature2 = self.temporal_fusion[lvl](tgt)
            intermediate.append(tgt)
            attention_scores.append(attn)
            
            # predicted afffinity score
            aff_score = self.affinity[lvl](temporal_feature2)
            affinity_scores.append(aff_score)

            # association features
            motion_feature = self.traj_proj[lvl](temporal_feature2)
            motion_feature = torch.sum(motion_feature, dim=0) + instance_feature
            motion_feature = self.traj_proj2[lvl](motion_feature)

            tau = self.gen_tau(motion_feature)
            tau = tau.unsqueeze(0)
            # attention mask 
            aux_attn_mask = -aff_score
            attn_mask = tau[None, ...] * aux_attn_mask[:, None, ...]

            motion_features.append(motion_feature)
            lvl += 1

        motion_features = torch.stack(motion_features)
        affinity_scores = torch.stack(affinity_scores)
        attention_scores = torch.stack(attention_scores)
        return motion_features, affinity_scores, attention_scores

    def forward(self, coordinate_info, spatial_info, spatial_dist, temporal_info=None, temporal_dist=None, tracked_feature=None, first_frame=False):
        """
        : param
        coordinate_info : geometry cues: center positions, sizes, rotations, one-hot encoded class, and a detection socre.
        spatial_info : spatial relation grpah
        spatial_dist : spatial relation distance
        temporal_info: temporal relation graph
        tempora_dist: temporal relation distance
        tracked_feature: tracked features from previous frame
        first_frame: first frame or not

        return: 
        coordinate_feature: project coordinate_info (geometry cues) into coordinate features through MLP
        instance_feature: spatial relation-aware features
        motion_feature: spatiotemporal relation-aware feature (motion feature) by last decoder layers
        motion_features: motion features output results by decoder layers
        affinity_scores: association scores
        attention_scores: attention scores by decoder
        """
        if first_frame:
            with torch.no_grad():
                coordinate_feature, instance_feature, motion_feature = self.first_frame(coordinate_info, spatial_info, spatial_dist, temporal_info)
            return coordinate_feature, instance_feature, motion_feature

        coordinate_feature, instance_feature, det_asso_w, spatial_w = self.forward_spatial(coordinate_info, spatial_info, spatial_dist)
        motion_features, affinity_scores, attention_scores = self.forward_temporal(instance_feature, temporal_info, temporal_dist, tracked_feature)
        motion_feature = motion_features[-1]
        return coordinate_feature, instance_feature, motion_feature, motion_features, affinity_scores, attention_scores

    def first_frame(self, coordinate_info, spatial_info, spatial_dist, temporal_info):
        """
        Process spatial and temporal relation-aware encoder in first frame
        """
        coordinate_feature, instance_feature, det_asso_w, spatial_w = self.forward_spatial(coordinate_info, spatial_info, spatial_dist)
        motion_feature = self.forward_temporal(instance_feature, temporal_info)
        return coordinate_feature, instance_feature, motion_feature



def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
