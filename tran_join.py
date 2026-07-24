import os
import glob
import anndata as ad

file_pattern = "Tran/tran_*.h5ad"
h5ad_files = sorted(glob.glob(file_pattern))
adata_list = [ad.read_h5ad(file) for file in h5ad_files]
combined_adata = ad.concat(adata_list, axis=0, join="inner", label="batch")
combined_adata.write_h5ad("Tran/Tran.h5ad")
