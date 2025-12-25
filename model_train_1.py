import scanpy as sc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sklearn
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.sparse import issparse
from scipy.special import comb
import pandas as pd
from collections import defaultdict, deque
import warnings
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import datetime
import os

warnings.filterwarnings('ignore')

# 设置中文字体和绘图样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


class MemoryEfficientMultiModalDataset(Dataset):
    def __init__(self, atac_data, rna_data, common_prefixes, atac_prefix_to_raw, rna_prefix_to_raw,
                 atac_samples_raw, rna_samples_raw, transform=None, augmentation=True):
        self.atac_data = atac_data
        self.rna_data = rna_data
        self.common_prefixes = common_prefixes
        self.atac_prefix_to_raw = atac_prefix_to_raw
        self.rna_prefix_to_raw = rna_prefix_to_raw
        self.atac_samples_raw = atac_samples_raw
        self.rna_samples_raw = rna_samples_raw
        self.transform = transform
        self.augmentation = augmentation

        # 预先计算特征维度，但不加载所有数据
        self._precompute_dimensions()

    def _precompute_dimensions(self):
        """预先计算特征维度"""
        sample_prefix = self.common_prefixes[0]
        atac_raw_name = self.atac_prefix_to_raw[sample_prefix]
        rna_raw_name = self.rna_prefix_to_raw[sample_prefix]

        atac_idx = self.atac_samples_raw.index(atac_raw_name)
        rna_idx = self.rna_samples_raw.index(rna_raw_name)

        atac_feat = self.atac_data.X[atac_idx]
        rna_feat = self.rna_data.X[rna_idx]

        if issparse(atac_feat):
            atac_feat = atac_feat.toarray().flatten()
        else:
            atac_feat = atac_feat.flatten()

        if issparse(rna_feat):
            rna_feat = rna_feat.toarray().flatten()
        else:
            rna_feat = rna_feat.flatten()

        self.atac_dim = atac_feat.shape[0]
        self.rna_dim = rna_feat.shape[0]

        print(f"ATAC特征维度: {self.atac_dim}")
        print(f"RNA特征维度: {self.rna_dim}")

        del atac_feat, rna_feat
        gc.collect()

    def _augment_features(self, features, modality='atac'):
        """数据增强"""
        if not self.augmentation:
            return features

        # 高斯噪声
        noise_std = 0.05 if modality == 'atac' else 0.02
        noise = torch.randn_like(features) * noise_std * features.std()

        # 随机mask
        mask_prob = 0.1
        mask = torch.rand_like(features) > mask_prob
        features = features * mask + noise * (~mask)

        # 随机缩放
        scale = torch.FloatTensor(1).uniform_(0.9, 1.1)
        features = features * scale

        return features

    def __len__(self):
        return len(self.common_prefixes)

    def __getitem__(self, idx):
        sample_prefix = self.common_prefixes[idx]
        atac_raw_name = self.atac_prefix_to_raw[sample_prefix]
        rna_raw_name = self.rna_prefix_to_raw[sample_prefix]

        atac_idx = self.atac_samples_raw.index(atac_raw_name)
        rna_idx = self.rna_samples_raw.index(rna_raw_name)

        atac_feat = self.atac_data.X[atac_idx]
        rna_feat = self.rna_data.X[rna_idx]

        if issparse(atac_feat):
            atac_feat = atac_feat.toarray().flatten()
        else:
            atac_feat = atac_feat.flatten()

        if issparse(rna_feat):
            rna_feat = rna_feat.toarray().flatten()
        else:
            rna_feat = rna_feat.flatten()

        atac_feat = torch.FloatTensor(atac_feat)
        rna_feat = torch.FloatTensor(rna_feat)

        if self.transform:
            atac_feat = self.transform(atac_feat)
            rna_feat = self.transform(rna_feat)

        # 应用数据增强
        atac_aug = self._augment_features(atac_feat.clone(), 'atac')
        rna_aug = self._augment_features(rna_feat.clone(), 'rna')

        return {
            'atac': atac_feat,
            'rna': rna_feat,
            'atac_aug': atac_aug,
            'rna_aug': rna_aug,
            'sample_name': sample_prefix
        }


class LightweightCLIPModel(nn.Module):
    def __init__(self, atac_dim, rna_dim, embedding_dim=64, hidden_dim=128):
        super(LightweightCLIPModel, self).__init__()

        self.atac_encoder = nn.Sequential(
            nn.Linear(atac_dim, hidden_dim),
            # nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.rna_encoder = nn.Sequential(
            nn.Linear(rna_dim, hidden_dim),
            # nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, atac, rna):
        atac_embedding = self.atac_encoder(atac)
        rna_embedding = self.rna_encoder(rna)

        atac_embedding = atac_embedding / atac_embedding.norm(dim=1, keepdim=True)
        rna_embedding = rna_embedding / rna_embedding.norm(dim=1, keepdim=True)

        return atac_embedding, rna_embedding


class MoCoModel(nn.Module):
    """引入MoCo思路的改进模型"""

    def __init__(self, atac_dim, rna_dim, embedding_dim=64, hidden_dim=128,
                 moco_dim=256, K=4096, m=0.999, T=0.07):
        super(MoCoModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.moco_dim = moco_dim
        self.K = K
        self.m = m
        self.T = T

        # 在线编码器 (query encoder)
        self.atac_encoder_q = nn.Sequential(
            nn.Linear(atac_dim, hidden_dim),
            # nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.rna_encoder_q = nn.Sequential(
            nn.Linear(rna_dim, hidden_dim),
            # nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # 动量编码器 (key encoder)
        self.atac_encoder_k = nn.Sequential(
            nn.Linear(atac_dim, hidden_dim),
            # nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.rna_encoder_k = nn.Sequential(
            nn.Linear(rna_dim, hidden_dim),
            # nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # MoCo投影头
        self.atac_projection_q = nn.Sequential(
            nn.Linear(embedding_dim, moco_dim),
            nn.ReLU(),
            nn.Linear(moco_dim, moco_dim)
        )

        self.rna_projection_q = nn.Sequential(
            nn.Linear(embedding_dim, moco_dim),
            nn.ReLU(),
            nn.Linear(moco_dim, moco_dim)
        )

        self.atac_projection_k = nn.Sequential(
            nn.Linear(embedding_dim, moco_dim),
            nn.ReLU(),
            nn.Linear(moco_dim, moco_dim)
        )

        self.rna_projection_k = nn.Sequential(
            nn.Linear(embedding_dim, moco_dim),
            nn.ReLU(),
            nn.Linear(moco_dim, moco_dim)
        )

        # 初始化动量编码器与在线编码器相同
        for param_q, param_k in zip(self.atac_encoder_q.parameters(), self.atac_encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.rna_encoder_q.parameters(), self.rna_encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.atac_projection_q.parameters(), self.atac_projection_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.rna_projection_q.parameters(), self.rna_projection_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # 创建队列
        self.register_buffer("atac_queue", torch.randn(moco_dim, K))
        self.atac_queue = nn.functional.normalize(self.atac_queue, dim=0)
        self.register_buffer("rna_queue", torch.randn(moco_dim, K))
        self.rna_queue = nn.functional.normalize(self.rna_queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """动量更新key编码器"""
        for param_q, param_k in zip(self.atac_encoder_q.parameters(), self.atac_encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.rna_encoder_q.parameters(), self.rna_encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.atac_projection_q.parameters(), self.atac_projection_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.rna_projection_q.parameters(), self.rna_projection_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, atac_keys, rna_keys):
        """更新队列"""
        batch_size = atac_keys.shape[0]

        ptr = int(self.queue_ptr)

        # 替换队列中的样本
        if ptr + batch_size > self.K:
            batch_size = self.K - ptr
            atac_keys = atac_keys[:batch_size]
            rna_keys = rna_keys[:batch_size]

        self.atac_queue[:, ptr:ptr + batch_size] = atac_keys.T
        self.rna_queue[:, ptr:ptr + batch_size] = rna_keys.T
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def forward(self, atac_q, rna_q, atac_k=None, rna_k=None, is_eval=False):
        """前向传播"""
        if is_eval:
            # 评估模式，只使用query编码器
            atac_embedding = self.atac_encoder_q(atac_q)
            rna_embedding = self.rna_encoder_q(rna_q)

            atac_embedding = atac_embedding / atac_embedding.norm(dim=1, keepdim=True)
            rna_embedding = rna_embedding / rna_embedding.norm(dim=1, keepdim=True)

            return atac_embedding, rna_embedding

        # 训练模式
        # Query编码
        atac_q_embed = self.atac_encoder_q(atac_q)
        rna_q_embed = self.rna_encoder_q(rna_q)

        atac_q_proj = self.atac_projection_q(atac_q_embed)
        rna_q_proj = self.rna_projection_q(rna_q_embed)

        atac_q_proj = nn.functional.normalize(atac_q_proj, dim=1)
        rna_q_proj = nn.functional.normalize(rna_q_proj, dim=1)

        # Key编码
        with torch.no_grad():
            self._momentum_update_key_encoder()

            atac_k_embed = self.atac_encoder_k(atac_k)
            rna_k_embed = self.rna_encoder_k(rna_k)

            atac_k_proj = self.atac_projection_k(atac_k_embed)
            rna_k_proj = self.rna_projection_k(rna_k_embed)

            atac_k_proj = nn.functional.normalize(atac_k_proj, dim=1)
            rna_k_proj = nn.functional.normalize(rna_k_proj, dim=1)

        # 计算logits
        # 正样本对
        l_pos_atac = torch.einsum('nc,nc->n', [atac_q_proj, rna_k_proj]).unsqueeze(-1)
        l_pos_rna = torch.einsum('nc,nc->n', [rna_q_proj, atac_k_proj]).unsqueeze(-1)

        # 负样本对 (从队列中获取)
        l_neg_atac = torch.einsum('nc,ck->nk', [atac_q_proj, self.rna_queue.clone().detach()])
        l_neg_rna = torch.einsum('nc,ck->nk', [rna_q_proj, self.atac_queue.clone().detach()])

        # 合并正负样本
        logits_atac = torch.cat([l_pos_atac, l_neg_atac], dim=1)
        logits_rna = torch.cat([l_pos_rna, l_neg_rna], dim=1)

        logits_atac /= self.T
        logits_rna /= self.T

        # 标签：正样本在位置0
        labels = torch.zeros(logits_atac.shape[0], dtype=torch.long).to(atac_q.device)

        # 更新队列
        self._dequeue_and_enqueue(atac_k_proj, rna_k_proj)

        return logits_atac, logits_rna, labels, atac_q_embed, rna_q_embed


def moco_loss(logits_atac, logits_rna, labels):
    """MoCo对比损失"""
    loss_atac = nn.CrossEntropyLoss()(logits_atac, labels)
    loss_rna = nn.CrossEntropyLoss()(logits_rna, labels)

    return (loss_atac + loss_rna) / 2


def clip_loss(atac_embeddings, rna_embeddings, logit_scale):
    """CLIP对比损失"""
    logit_scale = logit_scale.exp()
    logits_per_atac = logit_scale * atac_embeddings @ rna_embeddings.t()
    logits_per_rna = logits_per_atac.t()

    batch_size = atac_embeddings.size(0)
    labels = torch.arange(batch_size).to(device)

    loss_atac = nn.CrossEntropyLoss()(logits_per_atac, labels)
    loss_rna = nn.CrossEntropyLoss()(logits_per_rna, labels)

    return (loss_atac + loss_rna) / 2


class CombinedLoss(nn.Module):
    """组合损失函数：CLIP损失 + MoCo损失"""

    def __init__(self, alpha=0.7):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha

    def forward(self, clip_logit_scale, atac_emb_clip, rna_emb_clip,
                logits_atac_moco, logits_rna_moco, labels_moco):
        # CLIP损失
        loss_clip = clip_loss(atac_emb_clip, rna_emb_clip, clip_logit_scale)

        # MoCo损失
        loss_moco = moco_loss(logits_atac_moco, logits_rna_moco, labels_moco)

        # 组合损失
        total_loss = self.alpha * loss_clip + (1 - self.alpha) * loss_moco

        return total_loss, loss_clip, loss_moco


# 保持原有的ComprehensiveMetricCalculator和VisualizationGenerator类不变
class ComprehensiveMetricCalculator:
    def __init__(self, atac_embeddings, rna_embeddings, sample_names,
                 atac_cell_types=None, rna_cell_types=None):
        self.atac_embeddings = atac_embeddings
        self.rna_embeddings = rna_embeddings
        self.sample_names = sample_names
        self.n_samples = len(sample_names)

        self.atac_cell_types = atac_cell_types if atac_cell_types is not None else list(range(self.n_samples))
        self.rna_cell_types = rna_cell_types if rna_cell_types is not None else list(range(self.n_samples))

        if len(self.atac_cell_types) != self.n_samples or len(self.rna_cell_types) != self.n_samples:
            print("警告: 细胞类型标签数量与样本数量不匹配，使用伪标签")
            self.atac_cell_types = list(range(self.n_samples))
            self.rna_cell_types = list(range(self.n_samples))

    def calculate_foscttm(self, n_neighbors=None):
        """计算FOSCTTM指标 - 越小越好"""
        if n_neighbors is None:
            n_neighbors = min(100, self.n_samples - 1)
        else:
            n_neighbors = min(n_neighbors, self.n_samples - 1)

        nbrs_atac_to_rna = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', n_jobs=1)
        nbrs_atac_to_rna.fit(self.rna_embeddings)
        _, indices_atac_to_rna = nbrs_atac_to_rna.kneighbors(self.atac_embeddings)

        nbrs_rna_to_atac = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', n_jobs=1)
        nbrs_rna_to_atac.fit(self.atac_embeddings)
        _, indices_rna_to_atac = nbrs_rna_to_atac.kneighbors(self.rna_embeddings)

        foscttm_values = []

        for i in range(self.n_samples):
            if i in indices_atac_to_rna[i]:
                rank_atac = np.where(indices_atac_to_rna[i] == i)[0][0]
                fraction_atac = rank_atac / (n_neighbors - 1)
            else:
                fraction_atac = 1.0

            if i in indices_rna_to_atac[i]:
                rank_rna = np.where(indices_rna_to_atac[i] == i)[0][0]
                fraction_rna = rank_rna / (n_neighbors - 1)
            else:
                fraction_rna = 1.0

            foscttm_values.append((fraction_atac + fraction_rna) / 2)

        return np.mean(foscttm_values)

    def calculate_lisi(self, embeddings, labels, n_neighbors=None):
        """计算LISI指标 - 评估局部多样性"""
        if n_neighbors is None:
            n_neighbors = min(90, len(embeddings) - 1)

        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', n_jobs=1)
        nbrs.fit(embeddings)
        _, indices = nbrs.kneighbors(embeddings)

        lisi_scores = []

        for i in range(len(embeddings)):
            neighbor_indices = indices[i]
            neighbor_labels = [labels[j] for j in neighbor_indices]

            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            proportions = counts / len(neighbor_labels)
            simpson_index = np.sum(proportions ** 2)

            if simpson_index > 0:
                inverse_simpson = 1 / simpson_index
            else:
                inverse_simpson = len(unique_labels)

            lisi_scores.append(inverse_simpson)

        return np.mean(lisi_scores)

    def calculate_ari(self):
        """计算ARI指标 - 越高越好"""
        from sklearn.cluster import MiniBatchKMeans

        n_clusters = min(10, max(2, self.n_samples // 50))

        kmeans_atac = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=100)
        kmeans_rna = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=100)

        atac_labels = kmeans_atac.fit_predict(self.atac_embeddings)
        rna_labels = kmeans_rna.fit_predict(self.rna_embeddings)

        ari_score = adjusted_rand_score(atac_labels, rna_labels)
        return ari_score

    def calculate_asw_omics(self):
        """计算跨组学层ASW指标 - 越高越好"""
        labels = np.array([0] * self.n_samples + [1] * self.n_samples)
        combined_embeddings = np.vstack([self.atac_embeddings, self.rna_embeddings])

        if len(combined_embeddings) > 2000:
            indices = np.random.choice(len(combined_embeddings), 2000, replace=False)
            asw_score = silhouette_score(combined_embeddings[indices], labels[indices],
                                         metric='cosine', random_state=42)
        else:
            asw_score = silhouette_score(combined_embeddings, labels, metric='cosine')

        return asw_score

    def calculate_all_metrics(self):
        """计算所有四个评估指标"""
        metrics = {}

        print("1. 计算FOSCTTM (配对保真度)...")
        metrics['FOSCTTM'] = self.calculate_foscttm()

        print("2. 计算LISI指标...")
        batch_labels = list(range(self.n_samples))
        combined_embeddings = np.vstack([self.atac_embeddings, self.rna_embeddings])
        combined_batch_labels = batch_labels + batch_labels

        metrics['iLISI'] = self.calculate_lisi(combined_embeddings, combined_batch_labels)

        metrics['cLISI_ATAC'] = self.calculate_lisi(self.atac_embeddings, self.atac_cell_types)
        metrics['cLISI_RNA'] = self.calculate_lisi(self.rna_embeddings, self.rna_cell_types)
        metrics['cLISI_combined'] = (metrics['cLISI_ATAC'] + metrics['cLISI_RNA']) / 2

        print("3. 计算ARI (聚类一致性)...")
        metrics['ARI'] = self.calculate_ari()

        print("4. 计算ASW (跨模态一致性)...")
        metrics['ASW'] = self.calculate_asw_omics()

        return metrics

    def print_metric_interpretation(self, metrics):
        """打印指标解释和评估结果"""
        print("\n" + "=" * 60)
        print("模型性能评估结果")
        print("=" * 60)

        print(f"1. FOSCTTM (配对保真度): {metrics['FOSCTTM']:.4f}")
        print("   → 越小越好，表示更好的跨模态配对准确性")
        print(f"   当前值表示约 {metrics['FOSCTTM'] * 100:.1f}% 的样本比真实配对更接近")

        print(f"\n2. LISI指标:")
        print(f"   iLISI (批次混合): {metrics['iLISI']:.4f}")
        print("   → 越大越好，表示更好的批次混合效果")
        print(f"   cLISI (细胞类型保护): {metrics['cLISI_combined']:.4f}")
        print("   → 越小越好，表示更好的细胞类型结构保持")

        print(f"\n3. ARI (聚类一致性): {metrics['ARI']:.4f}")
        print("   → 越大越好，表示ATAC和RNA模态聚类结果更一致")
        print(f"   当前值表示聚类一致性为 {metrics['ARI'] * 100:.1f}%")

        print(f"\n4. ASW (跨模态一致性): {metrics['ASW']:.4f}")
        print("   → 越大越好，表示同一细胞在不同模态中的嵌入更紧密")
        print(f"   当前值: {metrics['ASW']:.4f} (范围: -1 到 1)")

        print("\n总体评估:")
        good_foscttm = metrics['FOSCTTM'] < 0.3
        good_ari = metrics['ARI'] > 0.5
        good_asw = metrics['ASW'] > 0.2

        if good_foscttm and good_ari and good_asw:
            print("✅ 模型表现优秀！")
        elif (good_foscttm or good_ari) and good_asw:
            print("✅ 模型表现良好！")
        else:
            print("⚠️ 模型有改进空间，建议调整超参数或增加训练轮数")

    def save_metrics_to_txt(self, metrics, file_path):
        """将指标结果保存到txt文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("模型性能评估结果\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"评估时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"样本数量: {self.n_samples}\n\n")

            f.write("主要指标:\n")
            f.write("-" * 40 + "\n")
            f.write(f"FOSCTTM (配对保真度): {metrics['FOSCTTM']:.6f}\n")
            f.write(f"ARI (聚类一致性): {metrics['ARI']:.6f}\n")
            f.write(f"ASW (跨模态一致性): {metrics['ASW']:.6f}\n")
            f.write(f"iLISI (批次混合): {metrics['iLISI']:.6f}\n")
            f.write(f"cLISI_combined (细胞类型保护): {metrics['cLISI_combined']:.6f}\n\n")

            f.write("详细LISI指标:\n")
            f.write("-" * 40 + "\n")
            f.write(f"cLISI_ATAC: {metrics['cLISI_ATAC']:.6f}\n")
            f.write(f"cLISI_RNA: {metrics['cLISI_RNA']:.6f}\n\n")

            f.write("指标解释:\n")
            f.write("-" * 40 + "\n")
            f.write("FOSCTTM: 越小越好，表示更好的跨模态配对准确性\n")
            f.write("ARI: 越大越好，表示ATAC和RNA模态聚类结果更一致\n")
            f.write("ASW: 越大越好，表示同一细胞在不同模态中的嵌入更紧密\n")
            f.write("iLISI: 越大越好，表示更好的批次混合效果\n")
            f.write("cLISI: 越小越好，表示更好的细胞类型结构保持\n\n")

            f.write("性能评估:\n")
            f.write("-" * 40 + "\n")
            good_foscttm = metrics['FOSCTTM'] < 0.3
            good_ari = metrics['ARI'] > 0.5
            good_asw = metrics['ASW'] > 0.2

            if good_foscttm and good_ari and good_asw:
                f.write("✅ 模型表现优秀！\n")
            elif (good_foscttm or good_ari) and good_asw:
                f.write("✅ 模型表现良好！\n")
            else:
                f.write("⚠️ 模型有改进空间，建议调整超参数或增加训练轮数\n")

        print(f"指标结果已保存至: {file_path}")


class SeparateVisualizationGenerator:
    def __init__(self, atac_embeddings, rna_embeddings, sample_names, metrics):
        self.atac_embeddings = atac_embeddings
        self.rna_embeddings = rna_embeddings
        self.sample_names = sample_names
        self.metrics = metrics
        self.n_samples = len(sample_names)

    def create_tsne_visualization_separate(self):
        """分别为ATAC和RNA创建t-SNE降维可视化"""
        print("分别为ATAC和RNA生成t-SNE可视化...")

        # 为ATAC数据生成t-SNE
        pca_atac = PCA(n_components=min(50, self.atac_embeddings.shape[1]))
        atac_pca = pca_atac.fit_transform(self.atac_embeddings)
        tsne_atac = TSNE(n_components=2, random_state=42, perplexity=min(30, self.n_samples - 1))
        atac_2d = tsne_atac.fit_transform(atac_pca)

        # 为RNA数据生成t-SNE
        pca_rna = PCA(n_components=min(50, self.rna_embeddings.shape[1]))
        rna_pca = pca_rna.fit_transform(self.rna_embeddings)
        tsne_rna = TSNE(n_components=2, random_state=42, perplexity=min(30, self.n_samples - 1))
        rna_2d = tsne_rna.fit_transform(rna_pca)

        return atac_2d, rna_2d

    def create_pca_visualization_separate(self):
        """分别为ATAC和RNA创建PCA降维可视化"""
        print("分别为ATAC和RNA生成PCA可视化...")

        # 为ATAC数据生成PCA
        pca_atac = PCA(n_components=2)
        atac_2d = pca_atac.fit_transform(self.atac_embeddings)

        # 为RNA数据生成PCA
        pca_rna = PCA(n_components=2)
        rna_2d = pca_rna.fit_transform(self.rna_embeddings)

        return atac_2d, rna_2d

    def plot_separate_visualizations(self, save_dir="separate_visualizations"):
        """创建单独的可视化图，分别显示ATAC和RNA的嵌入"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 生成降维结果
        atac_tsne, rna_tsne = self.create_tsne_visualization_separate()
        atac_pca, rna_pca = self.create_pca_visualization_separate()

        # 1. 单独显示ATAC的t-SNE
        plt.figure(figsize=(10, 8))
        plt.scatter(atac_tsne[:, 0], atac_tsne[:, 1], alpha=0.7, c='red', s=30)
        plt.title('ATAC Embeddings - t-SNE Visualization')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        atac_tsne_path = os.path.join(save_dir, "atac_tsne.png")
        plt.savefig(atac_tsne_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"ATAC t-SNE图已保存至: {atac_tsne_path}")

        # 2. 单独显示RNA的t-SNE
        plt.figure(figsize=(10, 8))
        plt.scatter(rna_tsne[:, 0], rna_tsne[:, 1], alpha=0.7, c='blue', s=30)
        plt.title('RNA Embeddings - t-SNE Visualization')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        rna_tsne_path = os.path.join(save_dir, "rna_tsne.png")
        plt.savefig(rna_tsne_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"RNA t-SNE图已保存至: {rna_tsne_path}")

        # 3. 单独显示ATAC的PCA
        plt.figure(figsize=(10, 8))
        plt.scatter(atac_pca[:, 0], atac_pca[:, 1], alpha=0.7, c='red', s=30)
        plt.title('ATAC Embeddings - PCA Visualization')
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        atac_pca_path = os.path.join(save_dir, "atac_pca.png")
        plt.savefig(atac_pca_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"ATAC PCA图已保存至: {atac_pca_path}")

        # 4. 单独显示RNA的PCA
        plt.figure(figsize=(10, 8))
        plt.scatter(rna_pca[:, 0], rna_pca[:, 1], alpha=0.7, c='blue', s=30)
        plt.title('RNA Embeddings - PCA Visualization')
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        rna_pca_path = os.path.join(save_dir, "rna_pca.png")
        plt.savefig(rna_pca_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"RNA PCA图已保存至: {rna_pca_path}")

        # 5. 创建对比图（可选）
        self._create_comparison_plot(atac_tsne, rna_tsne, atac_pca, rna_pca, save_dir)

        return {
            'atac_tsne': atac_tsne_path,
            'rna_tsne': rna_tsne_path,
            'atac_pca': atac_pca_path,
            'rna_pca': rna_pca_path
        }

    def _create_comparison_plot(self, atac_tsne, rna_tsne, atac_pca, rna_pca, save_dir):
        """创建对比图，在同一图中显示ATAC和RNA但用不同颜色"""
        # t-SNE对比图
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(atac_tsne[:, 0], atac_tsne[:, 1], alpha=0.7, c='red', s=30, label='ATAC')
        plt.scatter(rna_tsne[:, 0], rna_tsne[:, 1], alpha=0.7, c='blue', s=30, label='RNA')
        plt.title('ATAC vs RNA - t-SNE Comparison')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.scatter(atac_pca[:, 0], atac_pca[:, 1], alpha=0.7, c='red', s=30, label='ATAC')
        plt.scatter(rna_pca[:, 0], rna_pca[:, 1], alpha=0.7, c='blue', s=30, label='RNA')
        plt.title('ATAC vs RNA - PCA Comparison')
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        comparison_path = os.path.join(save_dir, "comparison_plot.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"对比图已保存至: {comparison_path}")


def train_with_moco(model, train_loader, val_loader, num_epochs=30, learning_rate=1e-4):
    """带MoCo的训练函数"""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = CombinedLoss(alpha=0.7)

    train_losses = []
    train_clip_losses = []
    train_moco_losses = []
    best_val_foscttm = float('inf')
    best_model_state = None
    all_val_metrics = []

    for epoch in range(num_epochs):
        model.train()
        epoch_total_loss = 0
        epoch_clip_loss = 0
        epoch_moco_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            atac = batch['atac'].to(device)
            rna = batch['rna'].to(device)
            atac_aug = batch['atac_aug'].to(device)
            rna_aug = batch['rna_aug'].to(device)

            optimizer.zero_grad()

            # MoCo前向传播
            logits_atac_moco, logits_rna_moco, labels_moco, atac_emb, rna_emb = model(
                atac_aug, rna_aug, atac, rna
            )

            # CLIP风格的前向传播（用于组合损失）
            atac_emb_clip, rna_emb_clip = model.atac_encoder_q(atac), model.rna_encoder_q(rna)
            atac_emb_clip = atac_emb_clip / atac_emb_clip.norm(dim=1, keepdim=True)
            rna_emb_clip = rna_emb_clip / rna_emb_clip.norm(dim=1, keepdim=True)

            # 组合损失
            total_loss, loss_clip, loss_moco = criterion(
                model.atac_encoder_q[0].weight.mean(),  # 使用一个参数作为logit_scale的替代
                atac_emb_clip, rna_emb_clip,
                logits_atac_moco, logits_rna_moco, labels_moco
            )

            total_loss.backward()
            optimizer.step()

            epoch_total_loss += total_loss.item()
            epoch_clip_loss += loss_clip.item()
            epoch_moco_loss += loss_moco.item()

            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_total_loss = epoch_total_loss / len(train_loader)
        avg_clip_loss = epoch_clip_loss / len(train_loader)
        avg_moco_loss = epoch_moco_loss / len(train_loader)

        train_losses.append(avg_total_loss)
        train_clip_losses.append(avg_clip_loss)
        train_moco_losses.append(avg_moco_loss)

        # 验证阶段
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                all_atac_emb = []
                all_rna_emb = []
                all_names = []

                for batch in val_loader:
                    atac = batch['atac'].to(device)
                    rna = batch['rna'].to(device)

                    atac_emb, rna_emb = model(atac, rna, is_eval=True)
                    all_atac_emb.append(atac_emb.cpu().numpy())
                    all_rna_emb.append(rna_emb.cpu().numpy())
                    all_names.extend(batch['sample_name'])

                all_atac_emb = np.vstack(all_atac_emb)
                all_rna_emb = np.vstack(all_rna_emb)

                metric_calculator = ComprehensiveMetricCalculator(all_atac_emb, all_rna_emb, all_names)
                val_metrics = metric_calculator.calculate_all_metrics()
                all_val_metrics.append(val_metrics)

                print(f'\nEpoch [{epoch + 1}/{num_epochs}]')
                print(
                    f'Total Loss: {avg_total_loss:.4f}, CLIP Loss: {avg_clip_loss:.4f}, MoCo Loss: {avg_moco_loss:.4f}')
                metric_calculator.print_metric_interpretation(val_metrics)

                if val_metrics['FOSCTTM'] < best_val_foscttm:
                    best_val_foscttm = val_metrics['FOSCTTM']
                    best_model_state = model.state_dict().copy()
                    torch.save(best_model_state, 'moco_clip_model_best.pth')
                    print("💾 保存最佳模型!")

            del all_atac_emb, all_rna_emb
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        scheduler.step()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, train_clip_losses, train_moco_losses, all_val_metrics


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("正在读取数据...")
    #atac_path = r"/media/wjw/16T/wjw/clip_moco_10x Multiome_pbmcs_human/10x Multiome_pbmcs_human/scglue_wugeo_10x-Multiome-Pbmc10k-ATAC.h5ad"
    #rna_path = r"/media/wjw/16T/wjw/clip_moco_10x Multiome_pbmcs_human/10x Multiome_pbmcs_human/scglue_wugeo_10x-Multiome-Pbmc10k-RNA.h5ad"
    atac_path = r"/mnt/e/BaiduNetdiskDownload/mRNA_ATAC(1)/mRNA_ATAC/clip_10x Multiome_pbmcs_human/10x Multiome_pbmcs_human/scglue_wugeo_10x-Multiome-Pbmc10k-ATAC.h5ad"
    rna_path = r"/mnt/e/BaiduNetdiskDownload/mRNA_ATAC(1)/mRNA_ATAC/clip_10x Multiome_pbmcs_human/10x Multiome_pbmcs_human/scglue_wugeo_10x-Multiome-Pbmc10k-RNA.h5ad"

    atac_data = sc.read_h5ad(atac_path, backed='r')
    rna_data = sc.read_h5ad(rna_path, backed='r')

    print(f"ATAC数据形状: {atac_data.shape}")
    print(f"RNA数据形状: {rna_data.shape}")

    def extract_sample_prefix(sample_name):
        if sample_name.endswith('_ATAC'):
            return sample_name[:-5]
        elif sample_name.endswith('_RNA'):
            return sample_name[:-4]
        else:
            return sample_name

    def find_sample_names(adata):
        return list(adata.obs.index)

    atac_samples_raw = find_sample_names(atac_data)
    rna_samples_raw = find_sample_names(rna_data)

    atac_prefixes = [extract_sample_prefix(sample) for sample in atac_samples_raw]
    rna_prefixes = [extract_sample_prefix(sample) for sample in rna_samples_raw]

    common_prefixes = list(set(atac_prefixes) & set(rna_prefixes))
    print(f"公共样本前缀数量: {len(common_prefixes)}")

    if len(common_prefixes) == 0:
        print("没有找到公共样本，无法进行训练")
        return

    if len(common_prefixes) > 5000:
        print(f"样本数量较多 ({len(common_prefixes)})，进行子采样...")
        common_prefixes = np.random.choice(common_prefixes, 5000, replace=False).tolist()
        print(f"子采样后样本数量: {len(common_prefixes)}")

    atac_prefix_to_raw = {prefix: raw for prefix, raw in zip(atac_prefixes, atac_samples_raw)}
    rna_prefix_to_raw = {prefix: raw for prefix, raw in zip(rna_prefixes, rna_samples_raw)}

    np.random.shuffle(common_prefixes)
    train_size = int(0.7 * len(common_prefixes))
    val_size = int(0.15 * len(common_prefixes))

    train_prefixes = common_prefixes[:train_size]
    val_prefixes = common_prefixes[train_size:train_size + val_size]
    test_prefixes = common_prefixes[:]  # train_size + val_size

    print(f"训练集大小: {len(train_prefixes)}")
    print(f"验证集大小: {len(val_prefixes)}")
    print(f"测试集大小: {len(test_prefixes)}")

    train_dataset = MemoryEfficientMultiModalDataset(atac_data, rna_data, train_prefixes,
                                                     atac_prefix_to_raw, rna_prefix_to_raw,
                                                     atac_samples_raw, rna_samples_raw, augmentation=True)
    val_dataset = MemoryEfficientMultiModalDataset(atac_data, rna_data, val_prefixes,
                                                   atac_prefix_to_raw, rna_prefix_to_raw,
                                                   atac_samples_raw, rna_samples_raw, augmentation=False)
    test_dataset = MemoryEfficientMultiModalDataset(atac_data, rna_data, test_prefixes,
                                                    atac_prefix_to_raw, rna_prefix_to_raw,
                                                    atac_samples_raw, rna_samples_raw, augmentation=False)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    atac_dim = train_dataset.atac_dim
    rna_dim = train_dataset.rna_dim

    print(f"创建MoCo模型: ATAC维度={atac_dim}, RNA维度={rna_dim}")
    model = MoCoModel(atac_dim, rna_dim, embedding_dim=64, hidden_dim=128,
                      moco_dim=256, K=2048, m=0.999, T=0.07).to(device)

    print("开始训练模型...")
    trained_model, train_losses, train_clip_losses, train_moco_losses, all_val_metrics = train_with_moco(
        model, train_loader, val_loader, num_epochs=100, learning_rate=1e-4
    )

    print("\n" + "=" * 60)
    print("最终测试集评估和可视化")
    print("=" * 60)

    trained_model.eval()
    with torch.no_grad():
        all_atac_emb = []
        all_rna_emb = []
        all_names = []

        for batch in test_loader:
            atac = batch['atac'].to(device)
            rna = batch['rna'].to(device)
            atac_emb, rna_emb = trained_model(atac, rna, is_eval=True)
            all_atac_emb.append(atac_emb.cpu().numpy())
            all_rna_emb.append(rna_emb.cpu().numpy())
            all_names.extend(batch['sample_name'])

        all_atac_emb = np.vstack(all_atac_emb)
        all_rna_emb = np.vstack(all_rna_emb)

        test_metric_calculator = ComprehensiveMetricCalculator(all_atac_emb, all_rna_emb, all_names)
        test_metrics = test_metric_calculator.calculate_all_metrics()
        test_metric_calculator.print_metric_interpretation(test_metrics)

        # 保存指标结果到txt文件
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = f"moco_test_metrics_{timestamp}.txt"
        test_metric_calculator.save_metrics_to_txt(test_metrics, metrics_file)

        # 生成单独的可视化
        print("\n生成单独的可视化图...")
        visualizer = SeparateVisualizationGenerator(all_atac_emb, all_rna_emb, all_names, test_metrics)
        saved_paths = visualizer.plot_separate_visualizations("moco_separate_visualizations")

    print("训练和评估完成!")


if __name__ == "__main__":
    main()