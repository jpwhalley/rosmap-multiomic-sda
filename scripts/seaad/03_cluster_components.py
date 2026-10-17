"""03_cluster_components.py: SEA-AD consensus components from the ten SDA runs (as deposited in 2023).

Pools the non-zero components of `sda/results1..10/it3000` (sample scores A, cell-type scores B1,
gene loadings X1, PIPs S1), clusters the sample scores by complete linkage on 1 - |Pearson r| cut
at 0.4, and keeps clusters that reach half the runs; members are sign-aligned to the first and averaged
(PIPs: median). Writes `sda/{samples,treatment,genes,pip}_final.csv` and two plots in `sda/figs/`.
The run count here carries an off-by-one in the component labels; scripts/round2/seaad/01_recluster_seaad.py
reproduces these files and applies the corrected count (141 components, used in the manuscript). Paths: `_paths.py`.

From the original analysis (dlpfc2_cluster.py, 2023; pandas 1.x). Changes: paths, `DataFrame.append`
written as the equivalent `pd.concat`, the matplotlib import restored (it was commented out, so the
original stopped at the plots after writing the tables) and the unused natsort import removed.
"""
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import scipy.cluster.hierarchy as hac
from scipy import stats
from collections import Counter
from os import path, makedirs
from _paths import SEAAD_DIR
SDA = SEAAD_DIR + '/sda/'
makedirs(SDA + 'figs', exist_ok=True)

def my_metric(x, y):
    r = abs(stats.pearsonr(x, y)[0])
    return 1 - r # absolute correlation to distance: range 0 to 1
    
# Get the clinical data in and lets restrict to COVID-19
samps_in_common = open(SDA + 'md_samples.txt').read().splitlines()
cell_types = open(SDA + 'cell_types.txt').read().splitlines()
rna_genes = open(SDA + 'genes.txt').read().splitlines()

        
# expr_info =  pd.read_csv('/Volumes/jpw/coding/new/b_extreme_response_new_data/data/Homo_sapiens.GRCh38.100.txt',index_col=0,sep=',')
# expr_info['ProbeID'] = expr_info.index.values
# expr_info = expr_info.loc[rna.columns.values]
        

# Now do all
indices = []
start = True
directory = SDA + 'results'
no_of_iterations = range(1,11)

for i in no_of_iterations:
    if start:
        # Get the names for samples
        samples = pd.read_csv(directory+str(i)+'/it3000/A', sep=' ', header=None)
        samples = samples.T
        samples.columns = samps_in_common
        sind_to_del = samples[(samples.T == 0).all()].index.values
        sind_to_keep = samples[(samples.T != 0).any()].index.values
        samples = samples[(samples.T != 0).any()]
        index = list(range(len(samples))) # Set up the correct index for all
        samples.index = index
        indices.append(index)
        
        # genes
        genes = pd.read_csv(directory+str(i)+'/it3000/X1', sep=' ', header=None)
        genes.columns = rna_genes
        # print(j, len(genes))
        # genes = genes.reindex(corr_info['ProbeID'], axis=1)
        gind_to_del = genes[(genes.T == 0).all()].index.values
        # ind_to_keep = (genes.T != 0).any()
        genes = genes.loc[sind_to_keep]
        genes.index = index
        # pip
        pip = pd.read_csv(directory+str(i)+'/it3000/S1', sep=' ', header=None)
        pip.columns = rna_genes
        # expr_info = expr_info.reindex(index=order_by_index(expr_info.index, index_natsorted(zip(expr_info.chrom, expr_info.chromStart))))
        # pip = pip.reindex(expr_info['ProbeID'], axis=1)
        pip = pip.loc[sind_to_keep]
        pip.index = index
        
        treatment = pd.read_csv(directory+str(i)+'/it3000/B1', sep=' ', header=None)
        treatment = treatment.T
        treatment.columns = cell_types
        tind_to_del = treatment[(treatment.T == 0).all()].index.values
        treatment = treatment[(treatment.T != 0).any()]
        treatment.index = index
        
        print(i)
        print('Results: ' +str([len(sind_to_del),len(gind_to_del), len(tind_to_del)]) + ' zero rows dropped')
        print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
        # To keep on adding to these dataframes
        start = False
    else:
        # samples
        samp_temp = pd.read_csv(directory+str(i)+'/it3000/A', sep=' ', header=None)
        samp_temp = samp_temp.T
        samp_temp.columns = samps_in_common
        sind_to_del = samp_temp[(samp_temp.T == 0).all()].index.values
        sind_to_keep = samp_temp[(samp_temp.T != 0).any()].index.values
        samp_temp = samp_temp[(samp_temp.T != 0).any()]
        index = list(range(max(samples.index),max(samples.index)+len(samp_temp)))
        samp_temp.index = index
        samples = pd.concat([samples, samp_temp])
        indices.append(index)
        
        # genes
        genes_temp = pd.read_csv(directory+str(i)+'/it3000/X1', sep=' ', header=None)
        genes_temp.columns = rna_genes
        # genes_temp = genes_temp.reindex(corr_info['ProbeID'], axis=1)
        gind_to_del = genes_temp[(genes_temp.T == 0).all()].index.values
        # ind_to_keep = (genes_temp.T != 0).any()
        genes_temp = genes_temp.loc[sind_to_keep]
        genes_temp.index = index
        genes = pd.concat([genes, genes_temp])
        # pip
        pip_temp = pd.read_csv(directory+str(i)+'/it3000/S1', sep=' ', header=None)
        pip_temp.columns = rna_genes
        # expr_info = expr_info.reindex(index=order_by_index(expr_info.index, index_natsorted(zip(expr_info.chrom, expr_info.chromStart))))
        # pip_temp = pip_temp.reindex(expr_info['ProbeID'], axis=1)
        pip_temp = pip_temp.loc[sind_to_keep]
        pip_temp.index = index
        pip = pd.concat([pip, pip_temp])
        
        # cell_type
        treat_temp = pd.read_csv(directory+str(i)+'/it3000/B1', sep=' ', header=None)
        treat_temp = treat_temp.T
        treat_temp.columns = cell_types
        tind_to_del = treat_temp[(treat_temp.T == 0).all()].index.values
        treat_temp = treat_temp[(treat_temp.T != 0).any()]
        treat_temp.index = index
        treatment = pd.concat([treatment, treat_temp])
    
        print(i)
        print('Results: ' +str([len(sind_to_del),len(gind_to_del), len(tind_to_del)]) + ' zero rows dropped')
        print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')


Z = hac.linkage(samples, method='complete', metric=my_metric) # 12min 12s
clusters = hac.fcluster(Z,0.4,criterion='distance')
clust_size = Counter(clusters)

samples_final = pd.DataFrame(columns=samples.columns.values)
treatment_final = pd.DataFrame(columns=treatment.columns.values)
genes_final = pd.DataFrame(columns=genes.columns.values)
pip_final = pd.DataFrame(columns=genes.columns.values)
        
index = 1
for clust in clust_size:
    if clust_size[clust] >= len(no_of_iterations)//2:
        counted = 0
        for ind in indices:
            if len(set(np.nonzero(clusters == clust)[0]).intersection(ind)) > 0:
                counted += 1
        if counted >= len(no_of_iterations)//2:
            temp_samples = samples[pd.Series(clusters==clust, index=samples.index.values)]
            for ind in temp_samples.index.values:
                # print(stats.pearsonr(temp_samples.iloc[0], temp_samples.loc[ind])[0])
                if stats.pearsonr(temp_samples.iloc[0], temp_samples.loc[ind])[0] < 0:
                    temp_samples.loc[ind] = temp_samples.loc[ind] * -1
            samples_final.loc[index] = temp_samples.mean(axis=0)
            
            temp_treatment = treatment[pd.Series(clusters==clust, index=treatment.index.values)]
            for ind in temp_treatment.index.values:
                # print(stats.pearsonr(temp_treatment.iloc[0], temp_treatment.loc[ind])[0])
                if stats.pearsonr(temp_treatment.iloc[0], temp_treatment.loc[ind])[0] < 0:
                    temp_treatment.loc[ind] = temp_treatment.loc[ind] * -1
            treatment_final.loc[index] = temp_treatment.mean(axis=0)
            
            temp_genes = {}
            temp_genes = genes[pd.Series(clusters==clust, index=samples.index.values)]
            for ind in temp_genes.index.values:
                # print(stats.pearsonr(temp_genes.iloc[0], temp_genes.loc[ind])[0])
                if stats.pearsonr(temp_genes.iloc[0], temp_genes.loc[ind])[0] < 0:
                    temp_genes.loc[ind] = temp_genes.loc[ind] * -1
                genes_final.loc[index] = temp_genes.mean(axis=0)
                pip_final.loc[index] = pip[pd.Series(clusters==clust, index=samples.index.values)].median(axis=0)
            index += 1
        else:
            print("Not enough clusters for: " + str(clust))
    
# Save the data for further use
samples_final.to_csv(SDA + 'samples_final.csv')
treatment_final.to_csv(SDA + 'treatment_final.csv')
genes_final.to_csv(SDA + 'genes_final.csv')
pip_final.to_csv(SDA + 'pip_final.csv')
#
# Plot dendogram - too big to plot?
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.yticks(np.arange(0,2.01,0.2))
hac.dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
plt.axhline(y=0.4, c='k')
plt.savefig(SDA + 'figs/sample_components_cluster.pdf')
plt.show()

# Plot frequency of clusters bar chart
sizes = []
for clust in clust_size:
    sizes.append(clust_size[clust])

sizes = pd.Series(sizes)
vc = sizes.value_counts()
vc = vc.sort_index()
vc.plot(kind='bar',colormap='ocean')
plt.xlabel('Cluster size')
plt.ylabel('Frequency')
plt.savefig(SDA + 'figs/no_of_clusters.pdf')
plt.show()
