import scanpy as sc
import numpy as np
from collections import defaultdict
import re

# 读取两个h5ad文件
atac_path = r"/media/wjw/16T/wjw/cluster_ai/SNARE_seq_cortex_mouse/Chen-2019-ATAC.h5ad"
rna_path = r"/media/wjw/16T/wjw/cluster_ai/SNARE_seq_cortex_mouse/Chen-2019-RNA.h5ad"

print("正在读取ATAC数据...")
atac_data = sc.read_h5ad(atac_path)
print("正在读取RNA数据...")
rna_data = sc.read_h5ad(rna_path)

print(f"ATAC数据形状: {atac_data.shape}")
print(f"RNA数据形状: {rna_data.shape}")


def extract_sample_prefix(sample_name):
    """提取样本前缀（去掉_ATAC或_RNA后缀）"""
    # 去掉_ATAC或_RNA后缀
    if sample_name.endswith('_ATAC'):
        return sample_name[:-5]  # 去掉最后5个字符"_ATAC"
    elif sample_name.endswith('_RNA'):
        return sample_name[:-4]  # 去掉最后4个字符"_RNA"
    else:
        return sample_name


def find_sample_names(adata):
    """查找样本名称"""
    # 使用obs的index作为样本名称
    sample_names = list(adata.obs.index)
    return sample_names


# 获取原始样本名称
atac_samples_raw = find_sample_names(atac_data)
rna_samples_raw = find_sample_names(rna_data)

print(f"ATAC样本数量: {len(atac_samples_raw)}")
print(f"RNA样本数量: {len(rna_samples_raw)}")

# 提取样本前缀
atac_prefixes = [extract_sample_prefix(sample) for sample in atac_samples_raw]
rna_prefixes = [extract_sample_prefix(sample) for sample in rna_samples_raw]

print(f"ATAC样本前缀示例: {atac_prefixes[:5]}")
print(f"RNA样本前缀示例: {rna_prefixes[:5]}")

# 查找公共样本前缀
common_prefixes = set(atac_prefixes) & set(rna_prefixes)
print(f"公共样本前缀数量: {len(common_prefixes)}")

if len(common_prefixes) > 0:
    # 创建映射字典：前缀 -> 原始样本名
    atac_prefix_to_raw = {prefix: raw for prefix, raw in zip(atac_prefixes, atac_samples_raw)}
    rna_prefix_to_raw = {prefix: raw for prefix, raw in zip(rna_prefixes, rna_samples_raw)}

    # 创建结果字典
    result_dict = defaultdict(dict)

    # 取前5个公共样本作为示例
    prefix_subset = list(common_prefixes)[:]

    for prefix in prefix_subset:
        atac_raw_name = atac_prefix_to_raw[prefix]
        rna_raw_name = rna_prefix_to_raw[prefix]

        # 获取样本在各自数据集中的索引
        atac_idx = atac_samples_raw.index(atac_raw_name)
        rna_idx = rna_samples_raw.index(rna_raw_name)

        # 获取ATAC表达数据
        if hasattr(atac_data.X, 'toarray'):
            atac_expression = atac_data.X[atac_idx:atac_idx + 1, :]  # 保持稀疏格式
            result_dict[prefix]['ATAC'] = {
                'original_name': atac_raw_name,
                'data_type': 'sparse_matrix',
                'shape': atac_expression.shape,
                'nnz': atac_expression.nnz if hasattr(atac_expression, 'nnz') else 'dense',
                'preview': f"稀疏矩阵: {atac_expression.shape}, 非零元素: {atac_expression.nnz}",
                "content": atac_expression.toarray(),
                'density': f"{atac_expression.nnz / (atac_expression.shape[1]):.4f}" if hasattr(atac_expression,
                                                                                                'nnz') else "1.0000"
            }
        else:
            atac_expression = atac_data.X[atac_idx:atac_idx + 1, :]  # 稠密矩阵
            result_dict[prefix]['ATAC'] = {
                'original_name': atac_raw_name,
                'data_type': 'dense_matrix',
                'shape': atac_expression.shape,
                'preview': f"稠密矩阵: {atac_expression.shape}",
                'non_zero_count': np.count_nonzero(atac_expression),
                'density': f"{np.count_nonzero(atac_expression) / atac_expression.size:.4f}"
            }

        # 获取RNA表达数据
        if hasattr(rna_data.X, 'toarray'):
            rna_expression = rna_data.X[rna_idx:rna_idx + 1, :]  # 保持稀疏格式
            result_dict[prefix]['RNA'] = {
                'original_name': rna_raw_name,
                'data_type': 'sparse_matrix',
                'shape': rna_expression.shape,
                'nnz': rna_expression.nnz if hasattr(rna_expression, 'nnz') else 'dense',
                'preview': f"稀疏矩阵: {rna_expression.shape}, 非零元素: {rna_expression.nnz}",
                "content" : rna_expression.toarray(),
                'density': f"{rna_expression.nnz / (rna_expression.shape[1]):.4f}" if hasattr(rna_expression,
                                                                                              'nnz') else "1.0000"
            }
        else:
            rna_expression = rna_data.X[rna_idx:rna_idx + 1, :]  # 稠密矩阵
            result_dict[prefix]['RNA'] = {
                'original_name': rna_raw_name,
                'data_type': 'dense_matrix',
                'shape': rna_expression.shape,
                'preview': f"稠密矩阵: {rna_expression.shape}",
                'non_zero_count': np.count_nonzero(rna_expression),
                'density': f"{np.count_nonzero(rna_expression) / rna_expression.size:.4f}"
            }


print(result_dict)

#     # 显示结果
#     print("\n" + "=" * 60)
#     print("公共样本表达数据预览 (基于样本前缀匹配):")
#     print("=" * 60)
#
#     for prefix, data in list(result_dict.items())[:10]:  # 显示前10个
#         print(f"\n样本前缀: {prefix}")
#         if 'ATAC' in data:
#             atac_info = data['ATAC']
#             print(f"  ATAC原始名: {atac_info['original_name']}")
#             print(f"  ATAC数据: {atac_info['preview']}")
#             print(f"  数据密度: {atac_info['density']}")
#
#         if 'RNA' in data:
#             rna_info = data['RNA']
#             print(f"  RNA原始名: {rna_info['original_name']}")
#             print(f"  RNA数据: {rna_info['preview']}")
#             print(f"  数据密度: {rna_info['density']}")
#
#     # 显示完整字典结构（前3个样本）
#     print("\n" + "=" * 60)
#     print("完整字典结构 (前3个样本):")
#     print("=" * 60)
#
#     final_dict = dict(list(result_dict.items())[:3])
#     for prefix, modalities in final_dict.items():
#         print(f"\n{prefix}:")
#         for modality, info in modalities.items():
#             print(f"  {modality}:")
#             for key, value in info.items():
#                 print(f"    {key}: {value}")
#
#     print(f"\n总共找到 {len(common_prefixes)} 个公共样本")
#
# else:
#     print("未找到公共样本前缀！")
#     print("可能的原因：")
#     print("1. 样本命名规则不同")
#     print("2. 数据来自不同的实验批次")
#     print("3. 需要不同的前缀提取方法")
#
# print(result_dict)
#
#
# # 显示数据集的更多信息
# print("\n" + "=" * 60)
# print("数据集详细信息:")
# print("=" * 60)
# print("ATAC数据:")
# print(f"  细胞数: {atac_data.n_obs}, 特征数: {atac_data.n_vars}")
# print(f"  obs列: {list(atac_data.obs.columns)}")
# if atac_data.obs.shape[1] > 0:
#     print("  obs前5行:")
#     print(atac_data.obs.head())
#
# print("\nRNA数据:")
# print(f"  细胞数: {rna_data.n_obs}, 特征数: {rna_data.n_vars}")
# print(f"  obs列: {list(rna_data.obs.columns)}")
# if rna_data.obs.shape[1] > 0:
#     print("  obs前5行:")
#     print(rna_data.obs.head())
#
# # 检查矩阵类型
# print("\n矩阵类型信息:")
# print(f"ATAC矩阵类型: {type(atac_data.X)}")
# print(f"RNA矩阵类型: {type(rna_data.X)}")