# Importing libraries
import scanpy as sc
import os
import sys 
import scvi
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Main code


# Opening file
print(os.getcwd())
print(sys.executable)
adata = sc.read_csv("D:\DATA\GSE171524_RAW\GSM5226574_C51ctr_raw_counts.csv\GSM5226574_C51ctr_raw_counts.csv").T   # Reading the data transposed (because scanpy requires rows to be cells)
adata
adata.obs #Number of observations (cells ids)
adata.var #Number of variables¡s (genes)
adata.X.shape #Numpy array

# Doublet removal (avoid the errors that come from two or more cells joined together into the same droplet)
sc.pp.filter_genes(adata, min_cells = 10) # Keep genes that are found at least in 10 of the cells.
sc.pp.highly_variable_genes(adata, n_top_genes = 2000, subset = True, flavor = 'seurat_v3')  # Keep the 2000 most relevant genes.
adata
scvi.model.SCVI.setup_anndata(adata) # SCVI model of the droplets
vae = scvi.model.SCVI(adata)
vae.train(accelerator="gpu", devices=1) 
solo = scvi.external.SOLO.from_scvi_model(vae)
solo.train(accelerator="gpu", devices=1)
df = solo.predict() #The highest score is the prediction, whter it is identified as a doublet or as a singlet cell
df['prediction'] = solo.predict(soft = False) #Added the predicted label with probabilities
# df.index = df.index.map(lambda x: x[:-1]) #Pandas gets rid of the last 2 characters of each cell sequence ID. Not necessary anymore for this dataset.
df.groupby('prediction').count()
df['dif'] = df.doublet - df.singlet
df
g = sns.displot(df[df["prediction"] == "doublet"], x="dif")
plt.show() 
# doublets = df[(df.prediction == 'doublet') & (df.dif >1)]  #This command filters out the arbitrary singlets identified as this category but with a prediction rate below 1
doublets = df[(df.prediction == 'doublet') & (df.dif >0.5)]  #This command filters out the arbitrary singlets identified as this category but with a prediction rate below 0.5
doublets
adata = sc.read_csv("D:\DATA\GSE171524_RAW\GSM5226574_C51ctr_raw_counts.csv\GSM5226574_C51ctr_raw_counts.csv").T   # Reading the data transposed (because scanpy requires rows to be cells)
adata.obs['doublet'] = adata.obs.index.isin(doublets.index) # We identify which cells are droplets (TRUE) or not (FALSE)
adata = adata[~adata.obs.doublet] # The  ~ means that you keep the False
adata

# Preprocessing
adata.var['mt'] = adata.var.index.str.startswith('MT-') # Show only the human mitochondrial genes (Sometimes in mice is Mt)
adata.var
ribo_url = "http://software.broadinstitute.org/gsea/msigdb/download_geneset.jsp?geneSetName=KEGG_RIBOSOME&fileType=txt" # Description of total human ribosomal genes
ribo_genes = pd.read_table(ribo_url, skiprows=2, header = None)
ribo_genes
ribo_genes[0].values
ribo_genes[0].value_counts
adata.var['ribo'] = adata.var_names.isin(ribo_genes[0].values) 
adata.var
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt','ribo'], log1p=False, percent_top=None, inplace=True) #Permite calcular cuántas veces aparece un gen en una célula y su promedio de aparición
adata.var.sort_values('n_cells_by_counts')
adata.obs
sc.pp.filter_genes(adata, min_cells = 3) # Keep genes that are found at least in 3 of the cells.
adata.var.sort_values('n_cells_by_counts')
# sc.pp.filter_cells(adata, min_genes = 200) # Keep cells that have at least 200 genes. Not necessary for this dataset
sc.pl.violin(adata, ['n_genes_by_counts','total_counts', 'pct_counts_mt','pct_counts_ribo'], jitter = 0.4, multi_panel=True) #Qc metrics to get rid of outliers. If a cell has significantly higher genes than the average, there's a chance that there's some artifact
upper_lim = np.quantile(adata.obs.n_genes_by_counts.values, .98) #Eliminar las células que tengan una cantidad de genes por encima del percentil 98
upper_lim
adata = adata[adata.obs.n_genes_by_counts < upper_lim] 
adata.obs
adata = adata[adata.obs.pct_counts_mt < 20]  #Maintain cells with a mitochondrial gene count of less than 20 genes.
adata = adata[adata.obs.pct_counts_ribo < 2]  #Maintain cells with a ribosomal gene count of less than 2 genes.

### Normalization 

# #Crucial step, because of the inter and intracell high variations (due to sequencing biases...etc)
adata.X.sum(axis = 1)
sc.pp.normalize_total(adata, target_sum=1e4) #Normalize every cell to 10000 UMI
adata.X.sum(axis = 1)
sc.pp.log1p(adata) #Change to log counts
adata.X.sum(axis = 1)
adata.raw = adata

### Clustering
 
# You can skip to integration step if you have more than one sample
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata.var
sc.pl.highly_variable_genes(adata) #For reducing the number of dimensions of the dataset
adata = adata[:, adata.var.highly_variable] #We keep the 2000 less variable genes
adata 
sc.pp.regress_out(adata,['total_counts','pct_counts_mt','pct_counts_ribo']) #Get rid of some of the variations of the data due to processing and sample quality (eg.sequencing artifacts)
sc.pp.scale(adata, max_value=10) #normalize each gene for the unit variance of that gene
sc.tl.pca(adata, svd_solver='arpack') #Principal Component Analysis to further reduce the dimensionality of the data
sc.pl.pca_variance_ratio(adata, log=True,n_pcs=50) #You would expect to see more difference as long as there's a bigger number of PC (PC50 more variance than PC10)
sc.pp.neighbors(adata, n_pcs = 30) #We pick 30 because in the PCA we see a relative plateau at PC30
adata.obsp['connectivities'].toarray() #We get a cell to cell matrix
adata.obsp['distances'].toarray() #We get a cell to cell matrix
sc.tl.umap(adata) #Project data from 30 dimensions to 2D
sc.pl.umap(adata)
sc.tl.leiden(adata, resolution = 0.5) #1 is the maximum number of clusters, 0 is the minimum number
adata.obs
sc.pl.umap(adata, color = ['leiden'])

#Integration
# You can skip integration step if you only have one sample
def pp(csv_path):
    adata = sc.read_csv(csv_path).T
    adata.X = sp.csr_matrix(adata.X)
    adata.X.data = adata.X.data.astype(np.int32, copy=False)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.filter_genes(adata, min_cells = 10)
    sc.pp.highly_variable_genes(adata, n_top_genes = 2000, subset = True, flavor = 'seurat_v3')
    scvi.model.SCVI.setup_anndata(adata)
    vae = scvi.model.SCVI(adata)
    vae.train()
    solo = scvi.external.SOLO.from_scvi_model(vae)
    solo.train()
    df = solo.predict()
    df['prediction'] = solo.predict(soft = False)
    df.index = df.index.map(lambda x: x[:-2])
    df['dif'] = df.doublet - df.singlet
    doublets = df[(df.prediction == 'doublet') & (df.dif > 0.5)]
    
    adata = sc.read_csv(csv_path).T
    adata.X = sp.csr_matrix(adata.X)
    adata.X.data = adata.X.data.astype(np.int32, copy=False)
    adata.layers["counts"] = adata.X.copy()
    adata.obs['Sample'] = csv_path.split('_')[2] #'raw_counts/GSM5226574_C51ctr_raw_counts.csv' Your data will likely be different. You need a different ID for each sample in the Sample column. In this case I used the .csv path returning the second item  divided by '_' character (the C51ctr part, in this case)  
    adata.obs['doublet'] = adata.obs.index.isin(doublets.index)
    adata = adata[~adata.obs.doublet]
    
    
    sc.pp.filter_cells(adata, min_genes=200) #get rid of cells with fewer than 200 genes
    #sc.pp.filter_genes(adata, min_cells=3) #get rid of genes that are found in fewer than 3 cells
    adata.var['mt'] = adata.var_names.str.startswith('mt-')  # annotate the group of mitochondrial genes as 'mt'
    adata.var['ribo'] = adata.var_names.isin(ribo_genes[0].values) # annotate the group of ribosomal genes as 'ribo'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt', 'ribo'], percent_top=None, log1p=False, inplace=True)
    #Filtering out the outliers
    upper_lim = np.quantile(adata.obs.n_genes_by_counts.values, .98) 
    adata = adata[adata.obs.n_genes_by_counts < upper_lim]
    adata = adata[adata.obs.pct_counts_mt < 20]
    adata = adata[adata.obs.pct_counts_ribo < 2] 

    return adata
out=[]
os.listdir("D:/DATA\GSE171524_RAW/Decompressed/")
for file in os.listdir("D:/DATA\GSE171524_RAW/Decompressed/"):
    out.append(pp("D:/DATA\GSE171524_RAW/Decompressed/"+file))
out[2]   #List of adata objects (26 in this case)
out
for i, ad in enumerate(out):
    ad.X = ad.X.tocsr() if sp.issparse(ad.X) else sp.csr_matrix(ad.X)
    if ad.X.dtype != np.int32:
        ad.X.data = ad.X.data.astype(np.int32, copy=False)
    ad.layers["counts"] = ad.layers.get("counts", ad.X.copy())
    ad.var_names_make_unique(); ad.obs_names_make_unique()
    ad.obs["batch"] = i
    ad.obsm.clear(); ad.varm.clear()  # evita merges densos

adata = sc.concat(
    out,
    join="outer",      # switch to inner if you only want the intersecting genes
    label="batch",
    merge="same",
    uns_merge="same",
    pairwise=False,
    index_unique=None,
)
# test = out[:11]  # o cualquier subconjunto pequeño
# adata_test = sc.concat(test, join="inner", label="batch", merge="same", uns_merge="same", pairwise=False)
# print(adata_test.X.shape, getattr(adata_test.X, "nnz", None))
# adata = sc.concat(out)
adata
adata.obs

sc.pp.filter_genes(adata, min_cells = 50) # Keep genes that are found at least in 50 of the cells.
# sc.pp.highly_variable_genes(adata, n_top_genes = 2000, subset = True, flavor = 'seurat_v3')  # Keep the 2000 most relevant genes.
adata
# adata.X = csr_matrix(adata.X)
adata.X
adata.obs.groupby('Sample').count() # Group data by sample
adata.layers['counts'] = adata.X.copy() # We save the raw data without integration or further preprocessing
sc.pp.normalize_total(adata, target_sum=1e4) #Normalize every cell to 10000 UMI
sc.pp.log1p(adata) #Change to log counts
adata.raw = adata
adata
# If the number of cells is less than 2 times the number of genes, you can apply the following lines:
    # sc.pp.highly_variable_genes(adata, n_top_genes = 3000, subset = True, layer = 'counts', flavor = 'seurat_v3', batch_key='Sample')  # Keep the 3000 most relevant genes. No batch key if one sample

adata.X = adata.X.tocsr()
if "counts" not in adata.layers:
    adata.layers["counts"] = adata.X.copy()
adata.obs["Sample"] = adata.obs["Sample"].astype("category")

# SCVI
scvi.model.SCVI.setup_anndata(
    adata,
    layer="counts",
    categorical_covariate_keys=["Sample"],
    continuous_covariate_keys=["pct_counts_mt", "total_counts", "pct_counts_ribo"]
)
model = scvi.model.SCVI(adata)
model.train(accelerator="gpu", devices=1)

# Latents
adata.obsm["X_scVI"] = model.get_latent_representation()

# # Normalized: safe version (subset of genes)
# hvg = adata.var.get("highly_variable", None)
# if hvg is not None and hvg.sum() > 0:
#     gene_list = list(adata.var_names[hvg])
# else:
#     gene_list = list(adata.var_names[:2000])  # fallback


norm = model.get_normalized_expression(library_size=1e4, return_numpy=True)
norm
adata.layers['scvi_normalized'] = norm
sc.pp.neighbors(adata, use_rep = 'X_scVI') #We pick the latent representation to calculate the neighbours
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution = 0.5)
sc.pl.umap(adata, color = ['leiden', 'Sample'], frameon = False) #Leiden is the label for the clusters
adata.write_h5ad('integrated.h5ad')




