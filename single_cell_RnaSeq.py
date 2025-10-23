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

# Normalization
