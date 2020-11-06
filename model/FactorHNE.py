import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.layers import FactorGNN


class FactorHNELayer(nn.ModuleList):
    def __init__(self,
                 num_latent,
                 in_dim,
                 hidden_dim,
                 num_heads,
                 attr_drop,
                 feat_drop):
        super(FactorHNELayer, self).__init__()
        self.g = None
        self.factorLayer = FactorGNN(in_dim, hidden_dim, hidden_dim, num_latent, num_heads, attr_drop, feat_drop)

    def forward(self, g, feat, target_idx):
        self.g = g
        embedding = self.factorLayer(self.g, feat)
        return embedding[target_idx]

    def get_loss(self):
        return self.factorLayer.merge_loss(self.factorLayer.compute_disentangle_loss())


class FactorHNE(nn.Module):
    def __init__(self,
                 num_paths,
                 num_latent,
                 in_dim,
                 hidden_dim,
                 num_heads,
                 attn_drop,
                 feat_drop):
        super(FactorHNE, self).__init__()
        self.num_path = num_paths
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.layers = nn.ModuleList()
        for i in range(num_paths):
            self.layers.append(FactorHNELayer(num_latent, in_dim, hidden_dim, num_heads, attn_drop, feat_drop))

        # metapath-level attention
        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=False)

        # weight initialization
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def get_loss(self):
        loss = 0
        for i in range(self.num_path):
            loss += self.layers[i].get_loss()
        return loss

    def forward(self, g_list, feat_list, target_idx_list):
        metapath_outs = [node_aggr_layer(g, feat, target_idx).view(-1, self.hidden_dim)
                         for node_aggr_layer, g, feat, target_idx in zip(
                self.layers, g_list, feat_list, target_idx_list)]

        beta = []
        for metapath_out in metapath_outs:
            fc1 = torch.tanh(self.fc1(metapath_out))
            fc1_mean = torch.mean(fc1, dim=0)
            fc2 = self.fc2(fc1_mean)
            beta.append(fc2)
        beta = torch.cat(beta, dim=0)
        beta = F.softmax(beta, dim=0)
        beta = torch.unsqueeze(beta, dim=-1)
        beta = torch.unsqueeze(beta, dim=-1)
        metapath_outs = [torch.unsqueeze(metapath_out, dim=0) for metapath_out in metapath_outs]
        metapath_outs = torch.cat(metapath_outs, dim=0)
        h = torch.sum(beta * metapath_outs, dim=0)
        return h


class FactorHNE_lp(nn.Module):
    def __init__(self,
                 num_paths_list,
                 feats_dim_list,
                 num_latent,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_heads,
                 attn_drop,
                 feat_drop):
        super(FactorHNE_lp, self).__init__()
        self.hidden_dim = hidden_dim

        # ntype-specific transformation
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        # feature dropout after trainsformation
        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        self.gene_layer = FactorHNE(num_paths_list[0],
                                    num_latent,
                                    in_dim,
                                    hidden_dim,
                                    num_heads,
                                    attn_drop,
                                    feat_drop)
        self.dis_layer = FactorHNE(num_paths_list[1],
                                   num_latent,
                                   in_dim,
                                   hidden_dim,
                                   num_heads,
                                   attn_drop,
                                   feat_drop)

        self.fc_gene = nn.Linear(hidden_dim, out_dim, bias=True)
        self.fc_dis = nn.Linear(hidden_dim, out_dim, bias=True)
        nn.init.xavier_normal_(self.fc_gene.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_dis.weight, gain=1.414)

    def get_loss(self):
        loss = 0
        loss += self.gene_layer.get_loss()
        loss += self.dis_layer.get_loss()
        return loss

    def forward(self, g_lists, features_list, type_mask, node_idx_lists, target_idx_lists):
        # ntype-specific transformation
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=features_list[0].device)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features_list[i])
        transformed_features = self.feat_drop(transformed_features)

        feat_list = [[], []]
        for mode in range(len(feat_list)):
            for node_idx_list in node_idx_lists[mode]:
                feat_list[mode].append(transformed_features[node_idx_list])

        logits_gene = self.fc_gene(self.gene_layer(g_lists[0], feat_list[0], target_idx_lists[0]))
        logits_dis = self.fc_dis(self.dis_layer(g_lists[1], feat_list[1], target_idx_lists[1]))
        return logits_gene, logits_dis
