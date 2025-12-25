# draw_embed.py
import scanpy as sc
import matplotlib.pyplot as plt
import os

# 1. 读取嵌入（已含细胞类型）
tmp_dir = "tmp_checkpoint"
adata_ATAC = sc.read_h5ad(f"{tmp_dir}/adata_ATAC_embed.h5ad")
adata_RNA  = sc.read_h5ad(f"{tmp_dir}/adata_RNA_embed.h5ad")
adata_all  = sc.read_h5ad(f"{tmp_dir}/adata_combined_embed.h5ad")

# 2. 统一输出目录
out_dir = "embed_plots"
os.makedirs(out_dir, exist_ok=True)

# 3. 单模态 t-SNE / PCA（含细胞类型图例）
for mod, adata in [('ATAC', adata_ATAC), ('RNA', adata_RNA)]:
    # t-SNE
    sc.tl.tsne(adata, random_state=42)
    sc.pl.tsne(adata, color='cell_type', show=False, frameon=False,
               title=f"{mod} t-SNE (cell_type)")
    plt.savefig(f"{out_dir}/{mod.lower()}_tsne_celltype.png", dpi=300, bbox_inches='tight')
    plt.close()

    # PCA
    sc.pp.pca(adata)
    sc.pl.pca(adata, color='cell_type', show=False, frameon=False,
              title=f"{mod} PCA (cell_type)")
    plt.savefig(f"{out_dir}/{mod.lower()}_pca_celltype.png", dpi=300, bbox_inches='tight')
    plt.close()

# 4. 合并 t-SNE / PCA（双图：cell_type + entity）
sc.tl.tsne(adata_all, random_state=42)
sc.pp.pca(adata_all)

for color_by in ['cell_type', 'entity']:
    # t-SNE
    sc.pl.tsne(adata_all, color=color_by, show=False, frameon=False,
               title=f"Combined t-SNE ({color_by})")
    plt.savefig(f"{out_dir}/combined_tsne_{color_by}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # PCA
    sc.pl.pca(adata_all, color=color_by, show=False, frameon=False,
              title=f"Combined PCA ({color_by})")
    plt.savefig(f"{out_dir}/combined_pca_{color_by}.png", dpi=300, bbox_inches='tight')
    plt.close()

print(">>> 所有嵌入图已保存 →", out_dir)