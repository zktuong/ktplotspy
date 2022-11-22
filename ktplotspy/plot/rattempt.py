#!/usr/bin/env python
# coding: utf-8

# ## Tutorial

# Welcome to `ktplotspy`! This is a python library to help visualise `CellphoneDB` results, ported from the original [ktplots R package](https://www.github.com/zktuong/ktplots) (which still has several other visualisation options). Here, we will go through a quick tutorial on how to use the functions in this package.

# **Import libraries**

# In[16]:


import os
import anndata as ad
import pandas as pd
import ktplotspy as kpy


# **Prepare input**

# We will need 3 files to use this package, the h5ad file used for `CelllphoneDB` and the `means.txt` and `pvalues.txt` output.

# In[17]:


os.chdir('/Users/kt16/Documents/Github/ktplotspy')

# read in the files
# 1) .h5ad file used for performing cellphonedb
adata = ad.read_h5ad('data/kidneyimmune.h5ad')

# 2) output from cellphonedb
means = pd.read_csv('data/out/means.txt', sep = '\t')
pvals = pd.read_csv('data/out/pvalues.txt', sep = '\t')


# ### Heatmap

# The original heatmap plot from `CellphoneDB` can be achieved with this reimplemented function.

# In[3]:


kpy.plot_cpdb_heatmap(
        adata=adata,
        pvals=pvals,
        celltype_key="celltype",
        figsize = (5,5),
        title = "Sum of significant interactions"
    )


# ### Dot plot

# A simple usage of `plot_cpdb` is like as follows:

# In[4]:


# TODO: How to specify the default plot resolution??
kpy.plot_cpdb(
        adata=adata,
        cell_type1="B cell",
        cell_type2=".", # this means all cell-types
        means=means,
        pvals=pvals,
        celltype_key="celltype",
        genes=["PTPRC", "TNFSF13"],
        figsize = (10,2),
        title = "interacting interactions!"
    )


# You can also specify a `gene_family`.

# In[5]:


kpy.plot_cpdb(
        adata=adata,
        cell_type1=".",
        cell_type2=".",
        means=means,
        pvals=pvals,
        celltype_key="celltype",
        gene_family = "chemokines",
        figsize = (20,4)
    )


# Or don't specify either and it will try to plot all significant interactions.

# In[6]:


kpy.plot_cpdb(
        adata=adata,
        cell_type1="B cell",
        cell_type2="pDC|T",
        means=means,
        pvals=pvals,
        celltype_key="celltype",
        highlight_size = 1,
        figsize = (4, 5)
    )


# If you prefer, you can also use the `squidpy` inspired plotting style:

# In[7]:


kpy.plot_cpdb(
        adata=adata,
        cell_type1="B cell",
        cell_type2=".",
        means=means,
        pvals=pvals,
        celltype_key="celltype",
        genes=["PTPRC", "CD40", "CLEC2D"],
        default_style = False,
        figsize = (10,2)
    )


# That's it for now! Please check out the original [ktplots R package](https://www.github.com/zktuong/ktplots) if you are after other kinds of visualisations.

# In[7]:


import pandas as pd
import holoviews as hv
from holoviews import opts, dim
from bokeh.sampledata.les_mis import data

hv.extension('bokeh')
hv.output(size=200)


# In[8]:


links = pd.DataFrame(data['links'])
print(links.head(3))


# In[9]:


hv.Chord(links)


# In[10]:


nodes = hv.Dataset(pd.DataFrame(data['nodes']), 'index')
nodes.data.head()


# In[12]:


links


# In[14]:


nodes.data


# In[11]:


chord = hv.Chord((links, nodes)).select(value=(5, None))
chord.opts(
    opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=dim('source').str(), 
               labels='name', node_color=dim('index').str()))


# In[18]:


# lr_interactions = plot_cpdb(cell_type1 = cell_type1, cell_type2 = cell_type2,
#         scdata = scdata, idents = idents, split.by = split.by, means = means, pvals = pvals,
#         keep_significant_only = keep_significant_only, standard_scale = standard_scale,
#         return_table = TRUE, degs_analysis = degs_analysis, ...)
lr_interactions = kpy.plot_cpdb(
        adata=adata,
        cell_type1="B cell",
        cell_type2=".", # this means all cell-types
        means=means,
        pvals=pvals,
        celltype_key="celltype",
        genes=["PTPRC", "TNFSF13"],
        return_table = True
    )


# In[19]:


lr_interactions


# In[22]:


try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

def flatten(l: list[list[str]]) -> list[str]:
    """
    Flatten a list-in-list-in-list.

    Parameters
    ----------
    l : list
        a list-in-list list

    Yields
    ------
    list
        a flattened list.
    """
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


# In[24]:


sep = '-'
subset_clusters = list(set(flatten([x.split(sep) for x in lr_interactions.celltype_group])))
subset_clusters


# In[25]:


idents = celltype_key = "celltype"
adata_subset = adata[adata.obs[idents].isin(subset_clusters)].copy()


# In[31]:


import re
interactions = means[["interacting_pair", "gene_a", "gene_b", "partner_a", "partner_b", "receptor_a", "receptor_b"]].copy()
interactions["converted"] = [re.sub('-', " ", x) for x in interactions.interacting_pair]
interactions["converted"] = [re.sub('_', "-", x) for x in interactions.interacting_pair]
interactions_subset = interactions[interactions["converted"].isin(list(lr_interactions.interaction_group))].copy()
interactions_subset


# In[57]:


tm0 = [x.split('_') for x in interactions_subset.interacting_pair]
tm0 = pd.DataFrame(tm0)
tm0.columns = ["id_a", "id_b"]
interactions_subset = pd.concat([interactions_subset.reset_index(), tm0],axis = 1)
dictionary = interactions_subset[["gene_a", "gene_b", "partner_a", "partner_b", "id_a", "id_b", "receptor_a", "receptor_b"]]


# In[58]:


dictionary


# In[68]:


geneid = list(set(list(interactions_subset.id_a) + list(interactions_subset.id_b)))
if not all([g in adata_subset.var.index for g in geneid]):
    geneid = list(set(list(interactions_subset.gene_a) + list(interactions_subset.gene_b)))


# In[71]:


adata_subset_tmp = adata_subset[:, adata_subset.var_names.isin(geneid)].copy()


# In[72]:


meta = adata_subset_tmp.obs.copy()


# In[74]:


adata_list = {}
adata_list_alt = {}
for x in list(set(meta[idents])):
    adata_list[x] = adata_subset_tmp[adata_subset_tmp.obs[idents] == x].copy()
    adata_list_alt[x] = adata_subset[adata_subset.obs[idents] == x].copy()


# In[91]:


from scipy.sparse import csr_matrix
def celltype_means(adata, layer = None):
    if layer is None:
        if isinstance(adata.X, csr_matrix):
            return np.mean(adata.X.toarray(), axis = 0)
        else: # assume it's numpy array
            return np.mean(adata.X, axis = 0)
    else:
        if isinstance(adata.layer[layer], csr_matrix):
            return np.mean(adata.layer[layer].toarray(), axis = 0)
        else:
            return np.mean(adata.layer[layer], axis = 0)

def celltype_fraction(adata, layer = None):
    if layer is None:
        if isinstance(adata.X, csr_matrix):
            return np.mean(adata.X.toarray() > 0, axis = 0)
        else: # assume it's numpy array
            return np.mean(adata.X > 0, axis = 0)
    else:
        if isinstance(adata.layer[layer], csr_matrix):
            return np.mean(adata.layer[layer].toarray() > 0, axis = 0)
        else:
            return np.mean(adata.layer[layer] > 0, axis = 0)


# In[116]:


layer = None
adata_list2, adata_list3 = {}, {}
for x in adata_list:
    adata_list2[x] = celltype_means(adata_list[x], layer)
    adata_list3[x] = celltype_fraction(adata_list[x], layer)
adata_list2 = pd.DataFrame(adata_list2, index = adata_subset_tmp.var_names)
adata_list3 = pd.DataFrame(adata_list3, index = adata_subset_tmp.var_names)


# In[117]:


def present(x):
    """Utility function to check if x is not null or blank."""
    return pd.notnull(x) and x != ""


# In[118]:


id_dict = {k:r for k,r in zip(dictionary.gene_a, dictionary.id_a) if present(r)}
id_b_dict = {k:r for k,r in zip(dictionary.gene_b, dictionary.id_b) if present(r)}
id_dict.update(id_b_dict)
id_dict


# In[119]:


adata_list3


# In[123]:


id_dict


# In[124]:


# humanreadablename = []
# for i in adata_list3.index:
#     humanreadablename.append(id_dict[i])
# humanreadablename


# In[125]:


expr_df = adata_list2
fraction_df = adata_list3


# In[158]:


from itertools import combinations
cells_test = list(set(meta[idents]))
remove_self=True
if remove_self:
    cell_type_grid = pd.DataFrame({c:[[cc for cc in cells_test if cc !=c]] for c in cells_test}).T
else:
    cell_type_grid = pd.DataFrame({c:[cells_test] for c in cells_test}).T
cell_type_grid[0] = cell_type_grid[0].apply(set)    
cell_type_grid = pd.DataFrame(
    data=list(combinations(cell_type_grid.index.tolist(), 2)), 
    columns=['source', 'target'])
cell_type_grid


# In[162]:


sep = ">@<"
ligand = list(interactions_subset.id_a)
receptor = list(interactions_subset.id_b)
pair = list(interactions_subset.interacting_pair)
converted_pair = list(interactions_subset.converted)
receptor_a = list(interactions_subset.receptor_a)
receptor_b = list(interactions_subset.receptor_b)
producers = list(cell_type_grid.source)
receivers = list(cell_type_grid.target)
barcodes = [a + sep + b for a,b in zip(lr_interactions.celltype_group, lr_interactions.interaction_group)]


# In[169]:


pp = producers
rc = receivers
cell_type_means = expr_df
cell_type_fractions = fraction_df
sce = adata_subset
sce_alt = adata_list_alt


# In[176]:


producer_expression = pd.DataFrame()
producer_fraction = pd.DataFrame()
receiver_expression = pd.DataFrame()
receiver_fraction = pd.DataFrame()
# adata_altx = adata_list_alt


# In[177]:


for i in pp:
    for j in ligand:
        if any([re.search("^"+j+"$", v) for v in cell_type_means.index]):
            x = cell_type_means.loc[j, i]
            y = cell_type_fractions.loc[j, i]
        else:
            if any([re.search("^"+j+"$", v) for v in sce.var_names]):
                pass
            else:
                x = 0
                y = 0
        producer_expression.at[j, i] = x
        producer_fraction.at[j, i] = y
for i in rc:
    for j in receptor:
        if any([re.search("^"+j+"$", v) for v in cell_type_means.index]):
            x = cell_type_means.loc[j, i]
            y = cell_type_fractions.loc[j, i]
        else:
            if any([re.search("^"+j+"$", v) for v in sce.var_names]):
                pass
            else:
                x = 0
                y = 0
        receiver_expression.at[j, i] = x
        receiver_fraction.at[j, i] = y


# In[180]:


test_df = []
for i in range(0, len(pp)):
    px = pp[i]
    rx = rc[i]
    for j in range(0, len(pair)):
        lg = ligand[j]
        rcp = receptor[j]
        ra = receptor_a[j]
        rb = receptor_b[j]
        pr = pair[j]
        out = pd.Series([lg, rcp, ra, rb, pr, px, rx, producer_expression.at[lg, px], producer_fraction.at[lg, px], receiver_expression.at[rcp, rx], receiver_fraction.at[rcp, rx]])
        test_df.append(out)


# In[202]:


df_


# In[203]:


df_ = pd.DataFrame(test_df)
df_.columns = ["ligand", "receptor", "receptor_a", "receptor_b", "pair", "producer", "receiver", "producer_expression", "producer_fraction", "receiver_expression", "receiver_fraction"]
df_['from'] = [p + sep + lg for p,lg in zip(df_.producer, df_.ligand)]
df_['to'] = [r + sep + rc for r,rc in zip(df_.receiver, df_.receptor)]
df_['barcode'] = [p + '-' + r + sep + cp for p,r,cp in zip(df_.producer, df_.receiver, df_.pair)]
df_


# In[207]:


interactions_names = [x + sep + y for x,y in zip(lr_interactions.celltype_group, lr_interactions.interaction_group)]
interactions_items = dict(zip(interactions_names, lr_interactions.scaled_means))
pval_items = dict(zip(interactions_names, lr_interactions.pvals))
interactions_names


# In[208]:


tmp_dfx.barcode
tmp_dfx = df_.copy()
tmp_dfx['pair_swap'] = tmp_dfx["pair"]
tmp_dfx["value"] = [interactions_items[b] for b in tmp_dfx.barcode]
tmp_dfx["pval"] = [pvals_items[b] for b in tmp_dfx.barcode]


# In[ ]:




