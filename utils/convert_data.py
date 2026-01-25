# Script to convert h5ad data files to TSV format for STdGCN
import scanpy as sc
import pandas as pd
import os

# Load the data
print("[INFO] Loading STARmap single-cell data...")
sc_adata = sc.read_h5ad('data/starmap/starmap_sc_adata.h5ad')
print(f"  Shape: {sc_adata.shape}")

print("[INFO] Loading STARmap spatial data...")
st_adata = sc.read_h5ad('data/starmap/starmap_st_adata.h5ad')
print(f"  Shape: {st_adata.shape}")

# Create output directory
output_dir = 'data/starmap_tsv'
os.makedirs(output_dir, exist_ok=True)

# Save single-cell expression matrix
print("[INFO] Saving sc_data.tsv...")
sc_df = pd.DataFrame(sc_adata.X.toarray() if hasattr(sc_adata.X, 'toarray') else sc_adata.X,
                     index=sc_adata.obs_names, columns=sc_adata.var_names)
sc_df.to_csv(f'{output_dir}/sc_data.tsv', sep='\t')

# Save single-cell labels
print("[INFO] Saving sc_label.tsv...")
# Get cell type column - try common names
cell_type_col = None
for col in ['cell_type', 'celltype', 'CellType', 'cluster', 'leiden', 'louvain']:
    if col in sc_adata.obs.columns:
        cell_type_col = col
        break

if cell_type_col is None:
    print(f"  Available columns: {sc_adata.obs.columns.tolist()}")
    cell_type_col = sc_adata.obs.columns[0]  # Use first column as fallback

sc_label = pd.DataFrame({'cell': sc_adata.obs_names, 'cell_type': sc_adata.obs[cell_type_col].values})
sc_label.to_csv(f'{output_dir}/sc_label.tsv', sep='\t', index=False)

# Save spatial expression matrix
print("[INFO] Saving ST_data.tsv...")
st_df = pd.DataFrame(st_adata.X.toarray() if hasattr(st_adata.X, 'toarray') else st_adata.X,
                     index=st_adata.obs_names, columns=st_adata.var_names)
st_df.to_csv(f'{output_dir}/ST_data.tsv', sep='\t')

# Save coordinates
print("[INFO] Saving coordinates.csv...")
if 'spatial' in st_adata.obsm:
    coords = pd.DataFrame(st_adata.obsm['spatial'], index=st_adata.obs_names, columns=['x', 'y'])
elif st_adata.obs.shape[1] >= 2:
    # Try to find x,y columns
    x_col = None
    y_col = None
    for col in st_adata.obs.columns:
        if 'x' in col.lower():
            x_col = col
        if 'y' in col.lower():
            y_col = col
    if x_col and y_col:
        coords = st_adata.obs[[x_col, y_col]].copy()
        coords.columns = ['x', 'y']
    else:
        coords = st_adata.obs.iloc[:, :2].copy()
        coords.columns = ['x', 'y']
else:
    raise ValueError("Cannot find spatial coordinates!")

coords.to_csv(f'{output_dir}/coordinates.csv')

print("[SUCCESS] Data conversion complete!")
print(f"Files saved to: {output_dir}/")
