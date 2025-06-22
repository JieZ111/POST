from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
from model.stratified_transformer import Stratified
from model.common import MLPWithoutResidual, KPConvResBlock, AggregatorLayer

import torch_points_kernels as tp
from util.logger import get_logger
from lib.pointops2.functions import pointops
from model.attention import SelfAttention, QGPA,MultiHeadAttention1

class COSeg(nn.Module):
    def __init__(self, args):
        super(COSeg, self).__init__()
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.n_subprototypes = args.n_subprototypes
        self.n_queries = args.n_queries
        self.n_classes = self.n_way + 1
        self.args = args
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([0.1] + [1 for _ in range(self.n_way)]),
            ignore_index=args.ignore_label,
        )
        self.criterion_base = nn.CrossEntropyLoss(
            ignore_index=args.ignore_label
        )
        self.transformer = QGPA()
        self.MultiHeadAttention1 = MultiHeadAttention1(192, 1)
        args.patch_size = args.grid_size * args.patch_size
        args.window_size = [
            args.patch_size * args.window_size * (2**i)
            for i in range(args.num_layers)
        ]
        args.grid_sizes = [
            args.patch_size * (2**i) for i in range(args.num_layers)
        ]
        args.quant_sizes = [
            args.quant_size * (2**i) for i in range(args.num_layers)
        ]

        if args.data_name == "s3dis":
            self.base_classes = 6
            if args.cvfold == 1:
                self.base_class_to_pred_label = {
                    0: 1,
                    3: 2,
                    4: 3,
                    8: 4,
                    10: 5,
                    11: 6,
                }
            else:
                self.base_class_to_pred_label = {
                    1: 1,
                    2: 2,
                    5: 3,
                    6: 4,
                    7: 5,
                    9: 6,
                }
        else:
            self.base_classes = 10
            if args.cvfold == 1:
                self.base_class_to_pred_label = {
                    2: 1,
                    3: 2,
                    5: 3,
                    6: 4,
                    7: 5,
                    10: 6,
                    12: 7,
                    13: 8,
                    14: 9,
                    19: 10,
                }
            else:
                self.base_class_to_pred_label = {
                    1: 1,
                    4: 2,
                    8: 3,
                    9: 4,
                    11: 5,
                    15: 6,
                    16: 7,
                    17: 8,
                    18: 9,
                    20: 10,
                }

        if self.main_process():
            self.logger = get_logger(args.save_path)

        self.encoder = Stratified(
            args.downsample_scale,
            args.depths,
            args.channels,
            args.num_heads,
            args.window_size,
            args.up_k,
            args.grid_sizes,
            args.quant_sizes,
            rel_query=args.rel_query,
            rel_key=args.rel_key,
            rel_value=args.rel_value,
            drop_path_rate=args.drop_path_rate,
            concat_xyz=args.concat_xyz,
            num_classes=self.args.classes // 2 + 1,
            ratio=args.ratio,
            k=args.k,
            prev_grid_size=args.grid_size,
            sigma=1.0,
            num_layers=args.num_layers,
            stem_transformer=args.stem_transformer,
            backbone=True,
            logger=get_logger(args.save_path),
        )

        self.feat_dim = args.channels[2]
        self.w = args.w
        self.sigma = args.sigma
        self.dist_method="cosine"
        self.bk_ffn = nn.Sequential(
            nn.Linear(self.feat_dim + self.feat_dim // 2, 4 * self.feat_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4 * self.feat_dim, self.feat_dim),
        )
        self.init_weights()
        self.register_buffer(
            "base_prototypes", torch.zeros(self.base_classes, self.feat_dim)
        )

    def init_weights(self):
        for name, m in self.named_parameters():
            if "class_attention.base_merge" in name:
                continue
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)

    def main_process(self):
        return not self.args.multiprocessing_distributed or (
            self.args.multiprocessing_distributed
            and self.args.rank % self.args.ngpus_per_node == 0
        )

    def forward(
        self,
        support_offset: torch.Tensor,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_offset: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        epoch: int,
        support_base_y: Optional[torch.Tensor] = None,
        query_base_y: Optional[torch.Tensor] = None,
        sampled_classes: Optional[np.array] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the COSeg model.

        Args:
            support_offset: Offset of each scene in the support set (shape: [N_way*K_shot]).
            support_x: Support point cloud inputs (shape: [N_support, in_channels]).
            support_y: Support masks (shape: [N_support]).
            query_offset: Offset of each scene in the query set (shape: [N_way]).
            query_x: Query point cloud inputs (shape: [N_query, in_channels]).
            query_y: Query labels (shape: [N_query]).
            epoch: Current training epoch.
            support_base_y: Base class labels in the support set (shape: [N_support]).
            query_base_y: Base class labels in the query set (shape: [N_query]).
            sampled_classes: The classes sampled in the current episode (shape: [N_way]).

        Returns:
            final_pred: Predicted class logits for query point clouds (shape: [1, n_way+1, N_query]).
            loss: The total loss value for this forward pass.
        """

        # get downsampled support features
        (
            support_feat,  # N_s, C
            support_x_low,  # N_s, 3
            support_offset_low,
            support_y_low,  # N_s
            _,
            support_base_y,  # N_s
        ) = self.getFeatures(
            support_x, support_offset, support_y, support_base_y
        )

        assert support_y_low.shape[0] == support_x_low.shape[0]
        # split support features and coords into list according to offset
        support_offset_low = support_offset_low[:-1].long().cpu()
        support_feat = torch.tensor_split(support_feat, support_offset_low)
        support_x_low = torch.tensor_split(support_x_low, support_offset_low)
        if support_base_y is not None:
            support_base_y = torch.tensor_split(
                support_base_y, support_offset_low
            )
        #support_feat (N,d=192)
        # get prototypes
        fg_mask = support_y_low
        bg_mask = torch.logical_not(support_y_low)
        fg_mask = torch.tensor_split(fg_mask, support_offset_low)
        bg_mask = torch.tensor_split(bg_mask, support_offset_low)
        # For k_shot, extract N_pt/k_shot per shot
        fg_prototypes = self.getPrototypes(
            support_x_low,
            support_feat,
            fg_mask,
            k=1,
        )  # N_way*N_pt, C
        bg_prototype = self.getPrototypes(
            support_x_low,
            support_feat,
            bg_mask,
            k=1,
        )  # N_way*N_pt, C
     
        # reduce the number of bg_prototypes to n_subprototypes when N_way > 1

        sparse_embeddings = torch.cat(
            [bg_prototype, fg_prototypes]
        )  # (N_way+1)*N_pt, C

        # get downsampled query features
        (
            query_feat,  # N_q, C
            query_x_low,  # N_q, 3
            query_offset_low,
            query_y_low,  # N_q
            q_base_pred,  # N_q, N_base_classes
            query_base_y,  # N_q
        ) = self.getFeatures(query_x, query_offset, query_y, query_base_y)

        # split query features into list according to offset
        query_offset_low_cpu = query_offset_low[:-1].long().cpu()
        query_feat = torch.tensor_split(query_feat, query_offset_low_cpu)
        query_x_low_list = torch.tensor_split(
            query_x_low, query_offset_low_cpu
        )
        if query_base_y is not None:
            query_base_y_list = torch.tensor_split(
                query_base_y, query_offset_low_cpu
            )
       
       # bg_prototype =(bg_prototype[0]+bg_prototype[1]/2).view(192)
       # print(len(fg_prototypes),len(bg_prototype))
        prototypes = [bg_prototype]
        for i in range(self.n_way):
            prototypes = prototypes + [fg_prototypes[i]]      
 
        #optimization prototype module
        for way in range(self.n_way):
            for shot in range(self.k_shot):
                regulize_loss_fg, loss_mask, surplus_mask, loss_count, surplus_count = self.regulize_L(prototypes, support_feat[way], fg_mask[way], bg_mask[way])
                loss_mask=()+(loss_mask.view(-1),)
                NUM=surplus_mask[0].shape[-1]
                surplus_mask=()+(surplus_mask.view(-1),)
                lossf1 = (loss_count[0, shot] / NUM).cuda()
                surf1 = (surplus_count[0, shot] / NUM).cuda()
                if lossf1>=0.3:
                    loss_support_feat = self.getMaskedFeatures(support_feat, loss_mask.cuda())
                    att_loss_prototype0 = self.MultiHeadAttention1(loss_support_feat[way, shot].view(1, 192),
                                                                  prototypes[way].view(1, 192),
                                                                  prototypes[way].view(1, 192)).view(1, 192).cuda()
                    prototypes[way] = self.w * att_loss_prototype0 + prototypes[way]
                if surf1>=0.3:
                    surplus_support_feat = self.getMaskedFeatures(support_feat, surplus_mask.cuda())
                    att_surplus_prototype0 = self.MultiHeadAttention1(surplus_support_feat[way, shot].view(1, 320),
                                                                     prototypes[way].view(1, 192),
                                                                     prototypes[way].view(1, 192)).view(1,192).cuda()
                    prototypes[way] = prototypes[way] - att_surplus_prototype0 * self.w
            prototypes[way] = prototypes[way].view(192)
        
        #bg_prototype = bg_prototype.view(192)
        #fg_prototypes = torch.stack(fg_prototypes).view(self.n_way, 192)
        #prototypes = [((bg_prototype[0]+bg_prototype[1])/2).view(192)]
        prototypes = [bg_prototype.view(192)]
        for i in range(self.n_way):
            prototypes = prototypes + [fg_prototypes[i].view(192)]
        
        self_regulize_loss = 0
        
        self_regulize_loss = self.sup_regulize_Loss(prototypes, support_feat, fg_mask, bg_mask[0])

        align_loss = 0
      
        #self-training strategy
        similarity_sts0 = [self.calculateSimilarity_trans(query_feat[0].transpose(0,1).unsqueeze(0), prototype, self.dist_method) for prototype in prototypes]
        query_pred0 = torch.stack(similarity_sts0, dim=1)
        #similarity_sts1 = [self.calculateSimilarity_trans(query_feat[1].transpose(0,1).unsqueeze(0), prototype, self.dist_method) for prototype in prototypes]
        #query_pred1 = torch.stack(similarity_sts1, dim=1)
        #query_pred=torch.cat((query_pred0,query_pred1),dim=2)
        query_pred=query_pred0
        #align_loss_epi = self.alignLoss_trans(torch.cat((query_feat[0],query_feat[1]),dim=0), query_pred, i, query_base_y, query_base_y)
        align_loss_epi = self.alignLoss_trans(query_feat[0], query_pred, i, query_base_y, query_base_y)
        align_loss += align_loss_epi
       
        
        query_pred=query_pred.squeeze(0).transpose(0,1)
        assert not torch.any(
            torch.isnan(query_pred)
        ), "torch.any(torch.isnan(query_pred))"
        loss = self.criterion(query_pred, query_y_low)
        if query_base_y is not None:
            loss += self.criterion_base(q_base_pred, query_base_y.cuda())
        
        final_pred = (
            pointops.interpolation(
                query_x_low,
                query_x[:, :3].cuda().contiguous(),
                query_pred.contiguous(),
                query_offset_low,
                query_offset.cuda(),
            )
            .transpose(0, 1)
            .unsqueeze(0)
        )  # 1, n_way+1, N_query

        
        return final_pred, loss+align_loss+self_regulize_loss
    def regulize_L(self, prototype_supp, supp_fts, fore_mask,back_mask):
        n_ways, n_shots = self.n_way, self.k_shot
        Num=len(fore_mask)
        loss_mask = torch.ones(1, n_shots, Num)
        surplus_mask = torch.ones(1, n_shots, Num)
        A = torch.ones(1, Num)
        B = torch.zeros(1, Num)
        count1=torch.ones(1, n_shots)
        count2=torch.ones(1, n_shots)
        fore_mask=fore_mask.view(1,n_shots,-1)
        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            prototypes = [prototype_supp[way + 1],prototype_supp[way + 1]]
            for shot in range(n_shots):
                img_fts = supp_fts[way, shot].unsqueeze(0)
                supp_dist = [self.calculateSimilarity(img_fts, prototype, self.dist_method) for prototype in prototypes]
                supp_pred = torch.stack(supp_dist, dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[0, shot], 255, device=img_fts.device).long()

                supp_label[fore_mask[0, shot] == 1] = 1
                supp_label[fore_mask[0, shot] == 0] = 0
                supp_label1 = torch.where(supp_dist[0] >self.sigma), torch.ones_like(supp_dist[0]), torch.zeros_like(supp_dist[0]))

                ###
                count1[0,shot] = (fore_mask[0, shot] - supp_label1 == 1).sum().item()
                count2[0,shot] = (fore_mask[0, shot] - supp_label1 == -1).sum().item()

                loss_mask[0, shot] = torch.where(fore_mask[0, shot].cpu() - supp_label1.cpu() == 1, A, B)
                surplus_mask[0, shot] = torch.where(fore_mask[0, shot].cpu() - supp_label1.cpu() == -1, A, B)
        return loss, loss_mask, surplus_mask,count1,count2
    def getFeatures(self, ptclouds, offset, gt, query_base_y=None):
        """
        Get the features of one point cloud from backbone network.

        Args:
            ptclouds: Input point clouds with shape (N_pt, 6), where N_pt is the number of points.
            offset: Offset tensor with shape (b), where b is the number of query scenes.
            gt: Ground truth labels. shape (N_pt).
            query_base_y: Optional base class labels for input point cloud. shape (N_pt).

        Returns:
            feat: Features from backbone with shape (N_down, C), where C is the number of channels.
            coord: Point coords. Shape (N_down, 3).
            offset: Offset for each scene. Shape (b).
            gt: Ground truth labels. Shape (N_down).
            base_pred: Base class predictions from backbone. Shape (N_down, N_base_classes).
            query_base_y: Base class labels for input point cloud. Shape (N_down).
        """
        coord, feat = (
            ptclouds[:, :3].contiguous(),
            ptclouds[:, 3:6].contiguous(),  # rgb color
        )  # (N_pt, 3), (N_pt, 3)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat(
            [torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0
        ).long()  # N_pt

        sigma = 1.0
        radius = 2.5 * self.args.grid_size * sigma
        batch = batch.to(coord.device)
        neighbor_idx = tp.ball_query(
            radius,
            self.args.max_num_neighbors,
            coord,
            coord,
            mode="partial_dense",
            batch_x=batch,
            batch_y=batch,
        )[
            0
        ]  # (N_pt, max_num_neighbors)

        coord, feat, offset, gt = (
            coord.cuda(non_blocking=True),
            feat.cuda(non_blocking=True),
            offset.cuda(non_blocking=True),
            gt.cuda(non_blocking=True),
        )
        batch = batch.cuda(non_blocking=True)
        neighbor_idx = neighbor_idx.cuda(non_blocking=True)
        assert batch.shape[0] == feat.shape[0]

        if self.args.concat_xyz:
            feat = torch.cat([feat, coord], 1)  # N_pt, 6
        # downsample the input point clouds
        feat, coord, offset, gt, base_pred, query_base_y = self.encoder(
            feat, coord, offset, batch, neighbor_idx, gt, query_base_y
        )  # (N_down, C_bc) (N_down, 3) (b), (N_down), (N_down, N_base_classes), (N_down)

        feat = self.bk_ffn(feat)  # N_down, C
        return feat, coord, offset, gt, base_pred, query_base_y

    def getPrototypes(self, coords, feats, masks, k=100):
        """
        Extract k prototypes for each scene.

        Args:
            coords: Point coordinates. List of (N_pt, 3).
            feats: Point features. List of (N_pt, C).
            masks: Target class masks. List of (N_pt).
            k: Number of prototypes extracted in each shot (default: 100).

        Return:
            prototypes: Shape (n_way*k_shot*k, C).
        """
        prototypes = []
        for i in range(0, self.n_way * self.k_shot):
            coord = coords[i][:, :3]  # N_pt, 3
            feat = feats[i]  # N_pt, C
            mask = masks[i].bool()  # N_pt

            coord_mask = coord[mask]
            feat_mask = feat[mask]
            protos = self.getMutiplePrototypes(
                coord_mask, feat_mask, k
            )  # k, C
            prototypes.append(protos)

        prototypes = torch.cat(prototypes)  # n_way*k_shot*k, C
        return prototypes
    def getPrototypes1(self, coords, feats, masks, k=100):
        """
        Extract k prototypes for each scene.

        Args:
            coords: Point coordinates. List of (N_pt, 3).
            feats: Point features. List of (N_pt, C).
            masks: Target class masks. List of (N_pt).
            k: Number of prototypes extracted in each shot (default: 100).

        Return:
            prototypes: Shape (n_way*k_shot*k, C).
        """
        prototypes = []
       
        coord = coords[:, :3]  # N_pt, 3
        feat = feats # N_pt, C
        mask = masks[0].bool()  # N_pt

        coord_mask = coord[mask]
        feat_mask = feat[mask]
        protos = self.getMutiplePrototypes(
            coord_mask, feat_mask, k
        )  # k, C
        prototypes.append(protos)

        prototypes = torch.cat(prototypes)  # n_way*k_shot*k, C
        return prototypes
    def getMutiplePrototypes(self, coord, feat, num_prototypes):
        """
        Extract k prototypes using furthest point samplling

        Args:
            coord: Point coordinates. Shape (N_pt, 3)
            feat: Point features. Shape (N_pt, C).
            num_prototypes: Number of prototypes to extract.
        Return:
            prototypes: Extracted prototypes. Shape: (num_prototypes, C).
        """
        # when the number of points is less than the number of prototypes, pad the points with zero features
        if feat.shape[0] <= num_prototypes:
            no_feats = feat.new_zeros(
                1,
                self.feat_dim,
            ).expand(num_prototypes - feat.shape[0], -1)
            feat = torch.cat([feat, no_feats])
            return feat

        # sample k seeds  by Farthest Point Sampling
        fps_index = pointops.furthestsampling(
            coord,
            torch.cuda.IntTensor([coord.shape[0]]),
            torch.cuda.IntTensor([num_prototypes]),
        ).long()  # (num_prototypes,)

        # use the k seeds as initial centers and compute the point-to-seed distance
        num_prototypes = len(fps_index)
        farthest_seeds = feat[fps_index]  # (num_prototypes, feat_dim)
        distances = torch.linalg.norm(
            feat[:, None, :] - farthest_seeds[None, :, :], dim=2
        )  # (N_pt, num_prototypes)

        # clustering the points to the nearest seed
        assignments = torch.argmin(distances, dim=1)  # (N_pt,)

        # aggregating each cluster to form prototype
        prototypes = torch.zeros(
            (num_prototypes, self.feat_dim), device="cuda"
        )
        for i in range(num_prototypes):
            selected = torch.nonzero(assignments == i).squeeze(
                1
            )  # (N_selected,)
            selected = feat[selected, :]  # (N_selected, C)
            if (
                len(selected) == 0
            ):  # exists same prototypes (coord not same), points are assigned to the prior prototype
                # simple use the seed as the prototype here
                prototypes[i] = feat[fps_index[i]]
                if self.main_process():
                    self.logger.info("len(selected) == 0")
            else:
                prototypes[i] = selected.mean(0)  # (C,)

        return prototypes

    def vis(
        self,
        query_offset,
        query_x,
        query_y,
        support_offset,
        support_x,
        support_y,
        final_pred,
    ):
        query_offset_cpu = query_offset[:-1].long().cpu()
        query_x_splits = torch.tensor_split(query_x, query_offset_cpu)
        query_y_splits = torch.tensor_split(query_y, query_offset_cpu)
        vis_pred = torch.tensor_split(final_pred, query_offset_cpu, dim=-1)
        support_offset_cpu = support_offset[:-1].long().cpu()
        vis_mask = torch.tensor_split(support_y, support_offset_cpu)

        sp_nps, sp_fgs = [], []
        for i, support_x_split in enumerate(
            torch.tensor_split(support_x, support_offset_cpu)
        ):
            sp_np = (
                support_x_split.detach().cpu().numpy()
            )  # num_points, in_channels
            sp_np[:, 3:6] = sp_np[:, 3:6] * 255.0
            sp_fg = np.concatenate(
                (
                    sp_np[:, :3],
                    vis_mask[i].unsqueeze(-1).detach().cpu().numpy(),
                ),
                axis=-1,
            )
            sp_nps.append(sp_np)
            sp_fgs.append(sp_fg)

        qu_s, qu_gts, qu_pds = [], [], []
        for i, query_x_split in enumerate(query_x_splits):
            qu = (
                query_x_split.detach().cpu().numpy()
            )  # num_points, in_channels
            qu[:, 3:6] = qu[:, 3:6] * 255.0
            result_tensor = torch.where(
                query_y_splits[i] == 255,
                torch.tensor(0, device=query_y.device),
                query_y_splits[i],
            )
            qu_gt = np.concatenate(
                (
                    qu[:, :3],
                    result_tensor.unsqueeze(-1).detach().cpu().numpy(),
                ),
                axis=-1,
            )
            q_prd = np.concatenate(
                (
                    qu[:, :3],
                    vis_pred[i]
                    .squeeze(0)
                    .max(0)[1]
                    .unsqueeze(-1)
                    .detach()
                    .cpu()
                    .numpy(),
                ),
                axis=-1,
            )

            qu_s.append(qu)
            qu_gts.append(qu_gt)
            qu_pds.append(q_prd)

        wandb.log(
            {
                "Support": [
                    wandb.Object3D(sp_nps[i]) for i in range(len(sp_nps))
                ],
                "Support_fg": [
                    wandb.Object3D(sp_fgs[i]) for i in range(len(sp_fgs))
                ],
                "Query": [wandb.Object3D(qu_s[i]) for i in range(len(qu_s))],
                "Query_pred": [
                    wandb.Object3D(qu_pds[i]) for i in range(len(qu_pds))
                ],
                "Query_GT": [
                    wandb.Object3D(qu_gts[i]) for i in range(len(qu_gts))
                ],
            }
        )
    def sup_regulize_Loss(self, prototype_supp, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype suppoort self alignment branch

        Args:
            prototypes: embedding features for query images
                expect shape: N x C x num_points
            supp_fts: embedding features for support images
                expect shape: (Wa x Shot) x C x num_points
            fore_mask: foreground masks for support images
                expect shape: (way x shot) x num_points
            back_mask: background masks for support images
                expect shape: (way x shot) x num_points
        """
        n_ways, n_shots = self.n_way, self.k_shot

        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            prototypes = [prototype_supp[0], prototype_supp[way + 1]]
            for shot in range(n_shots):
                supp_fts1=supp_fts[shot]
                fore_mask1=fore_mask[shot]
                fore_mask1=fore_mask1.view(1, 1 ,-1)
                supp_fts1=supp_fts1.view( 1, 1 ,-1,192)
                img_fts = supp_fts1[0, 0].transpose(0,1).unsqueeze(0)
              
                supp_dist = [self.calculateSimilarity_trans(img_fts, prototype, 'cosine') for prototype in prototypes]
                
                supp_pred = torch.stack(supp_dist, dim=1)
                
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask1[0, 0], 255, device=img_fts.device).long()

                supp_label[fore_mask1[0, 0] == 1] = 1
                supp_label[fore_mask1[0, 0] == 0] = 0
                # Compute Loss
            
                loss = loss + F.cross_entropy(supp_pred, supp_label.unsqueeze(0), ignore_index=255) / n_shots / n_ways
        return loss
    def calculateSimilarity(self, feat,  prototype, method='cosine', scaler=10):
        """
        Calculate the Similarity between query point-level features and prototypes

        Args:
            feat: input query point-level features
                  shape: (n_queries, feat_dim, num_points)
            prototype: prototype of one semantic class
                       shape: (feat_dim,)
            method: 'cosine' or 'euclidean', different ways to calculate similarity
            scaler: used when 'cosine' distance is computed.
                    By multiplying the factor with cosine distance can achieve comparable performance
                    as using squared Euclidean distance (refer to PANet [ICCV2019])
        Return:
            similarity: similarity between query point to prototype
                        shape: (n_queries, 1, num_points)
        """
        if method == 'cosine':
            similarity = F.cosine_similarity(feat, prototype[None, ..., None], dim=1) * scaler
        elif method == 'euclidean':
            similarity = - F.pairwise_distance(feat, prototype[None, ..., None], p=2)**2
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' %method)
        return similarity

    def calculateSimilarity_trans(self, feat,  prototype, method='cosine', scaler=10):
        """
        Calculate the Similarity between query point-level features and prototypes

        Args:
            feat: input query point-level features
                  shape: (n_queries, feat_dim, num_points)
            prototype: prototype of one semantic class
                       shape: (feat_dim,)
            method: 'cosine' or 'euclidean', different ways to calculate similarity
            scaler: used when 'cosine' distance is computed.
                    By multiplying the factor with cosine distance can achieve comparable performance
                    as using squared Euclidean distance (refer to PANet [ICCV2019])
        Return:
            similarity: similarity between query point to prototype
                        shape: (n_queries, 1, num_points)
        """
        if method == 'cosine':
            similarity = F.cosine_similarity(feat, prototype[..., None], dim=1) * scaler
        elif method == 'euclidean':
            similarity = - F.pairwise_distance(feat, prototype[..., None], p=2)**2
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' %method)
        return similarity

    def computeCrossEntropyLoss(self, query_logits, query_labels):
        """ Calculate the CrossEntropy Loss for query set
        """
        return F.cross_entropy(query_logits, query_labels)
    def focalloss(self,query_logits,query_labels):
        ce=F.cross_entropy(query_logits,query_labels)
        pt=torch.exp(-ce)
        return (1-pt)**2*ce
    def alignLoss_trans(self, qry_fts, pred, i, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x num_points
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x num_points
            supp_fts: embedding features for support images
                expect shape: (Wa x Shot) x C x num_points
            fore_mask: foreground masks for support images
                expect shape: (way x shot) x num_points
            back_mask: background masks for support images
                expect shape: (way x shot) x num_points
        """
        if  fore_mask is None:
            return 0
        n_ways, n_shots = self.n_way, 1
        qry_fts=qry_fts.transpose(0,1).unsqueeze(0)
      

        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'

        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3)) / (pred_mask.sum(dim=(0, 3)) + 1e-5)
    
        # Compute the support loss
        loss = 0
        
            # Get the query prototypes
        #prototypes = [qry_prototypes[0], qry_prototypes[1], qry_prototypes[2]]
        prototypes = [qry_prototypes[0], qry_prototypes[1]]
        for shot in range(n_shots):
            img_fts = qry_fts

            qry_dist = [self.calculateSimilarity(img_fts, prototype, self.dist_method) for prototype in prototypes]
            qry_pred = torch.stack(qry_dist, dim=1)
            # Construct the support Ground-Truth segmentation
            qry_label = torch.full_like(fore_mask, 255, device=img_fts.device).long()

            qry_label[fore_mask == 1] = 1
            qry_label[fore_mask == 0] = 0
            #qry_label[fore_mask == 2] = 2
            # Compute Loss
            loss = loss + F.cross_entropy(qry_pred, qry_label.unsqueeze(0), ignore_index=255) / n_shots / n_ways
        return loss
