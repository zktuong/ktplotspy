#!/usr/bin/env python
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import defaultdict
from itertools import combinations
from matplotlib.lines import Line2D
from pycircos import Garc, Gcircle
from typing import Optional, Tuple, Dict, Union

from ktplotspy.utils.settings import DEFAULT_SEP
from ktplotspy.utils.support import celltype_fraction, celltype_means, find_complex, flatten, generate_df, present
from ktplotspy.plot import plot_cpdb


def plot_cpdb_chord(
    adata: "AnnData",
    means: pd.DataFrame,
    pvals: pd.DataFrame,
    deconvoluted: pd.DataFrame,
    celltype_key: str,
    remove_self: bool = True,
    gap: int = 2,
    scale_line: int = 10,
    layer: Optional[str] = None,
    size: int = 50,
    interspace: int = 2,
    raxis_range: Tuple[int, int] = (950, 1000),
    labelposition: int = 80,
    label_visible: bool = True,
    col_dict: Dict[str, str] = None,
    figsize: Tuple[Union[int, float], Union[int, float]] = (8, 8),
    *args,
    **kwargs
):
    # assert splitby = False
    splitby_key, return_table = None, True
    # run plot_cpdb
    lr_interactions = plot_cpdb(
        adata=adata,
        means=means,
        pvals=pvals,
        celltype_key=celltype_key,
        *args,
        **kwargs,
        return_table=return_table,
        splitby_key=splitby_key
    )
    # do some name wrangling
    subset_clusters = list(set(flatten([x.split("-") for x in lr_interactions.celltype_group])))
    adata_subset = adata[adata.obs[celltype_key].isin(subset_clusters)].copy()
    interactions = means[["interacting_pair", "gene_a", "gene_b", "partner_a", "partner_b", "receptor_a", "receptor_b"]].copy()
    interactions["converted"] = [re.sub("-", " ", x) for x in interactions.interacting_pair]
    interactions["converted"] = [re.sub("_", "-", x) for x in interactions.interacting_pair]
    barcodes = lr_interactions["barcode"] = [
        a + DEFAULT_SEP + b for a, b in zip(lr_interactions.celltype_group, lr_interactions.interaction_group)
    ]
    interactions_subset = interactions[interactions["converted"].isin(list(lr_interactions.interaction_group))].copy()
    # handle complexes gently
    tm0 = {kx: rx.split("_") for kx, rx in interactions_subset.interacting_pair.items()}
    if any([len(x) > 2 for x in tm0.values()]):
        complex_id, simple_id = [], []
        for i, j in tm0.items():
            if len(j) > 2:
                complex_id.append(i)
            elif len(j) == 2:
                simple_id.append(i)
        _interactions_subset = interactions_subset.loc[complex_id].copy()
        _interactions_subset_simp = interactions_subset.loc[simple_id].copy()
        complex_idx1 = [i for i, j in _interactions_subset.partner_b.items() if re.search("complex:", j)]
        complex_idx2 = [i for i, j in _interactions_subset.partner_a.items() if re.search("complex:", j)]
        # complex_idx
        simple_1 = list(_interactions_subset.loc[complex_idx1, "interacting_pair"])
        simple_2 = list(_interactions_subset.loc[complex_idx2, "interacting_pair"])
        partner_1 = [re.sub("complex:", "", b) for b in _interactions_subset.loc[complex_idx1, "partner_b"]]
        partner_2 = [re.sub("complex:", "", a) for a in _interactions_subset.loc[complex_idx2, "partner_a"]]
        for i, _ in enumerate(simple_1):
            simple_1[i] = re.sub(partner_1[i] + "_|_" + partner_1[i], "", simple_1[i])
        for i, _ in enumerate(simple_2):
            simple_2[i] = re.sub(partner_2[i] + "_|_" + partner_2[i], "", simple_2[i])
        tmpdf = pd.concat([pd.DataFrame(zip(simple_1, partner_1)), pd.DataFrame(zip(partner_2, simple_2))])
        tmpdf.index = complex_id
        tmpdf.columns = ["id_a", "id_b"]
        _interactions_subset = pd.concat([_interactions_subset, tmpdf], axis=1)
        simple_tm0 = pd.DataFrame(
            [rx.split("_") for rx in _interactions_subset_simp.interacting_pair],
            columns=["id_a", "id_b"],
            index=_interactions_subset_simp.index,
        )
        _interactions_subset_simp = pd.concat([_interactions_subset_simp, simple_tm0], axis=1)
        interactions_subset = pd.concat([_interactions_subset_simp, _interactions_subset], axis=0)
    else:
        tm0 = pd.DataFrame(tm0).T
        tm0.columns = ["id_a", "id_b"]
        interactions_subset = pd.concat([interactions_subset, tm0], axis=1)

    # keep only useful genes
    geneid = list(set(list(interactions_subset.id_a) + list(interactions_subset.id_b)))
    if not all([g in adata_subset.var.index for g in geneid]):
        geneid = list(set(list(interactions_subset.gene_a) + list(interactions_subset.gene_b)))
    # create a subet anndata
    adata_subset_tmp = adata_subset[:, adata_subset.var_names.isin(geneid)].copy()
    meta = adata_subset_tmp.obs.copy()
    adata_list, adata_list_alt = {}, {}
    for x in list(set(meta[celltype_key])):
        adata_list[x] = adata_subset_tmp[adata_subset_tmp.obs[celltype_key] == x].copy()
        adata_list_alt[x] = adata_subset[adata_subset.obs[celltype_key] == x].copy()

    # create expression and fraction dataframes.
    adata_list2, adata_list3 = {}, {}
    for x in adata_list:
        adata_list2[x] = celltype_means(adata_list[x], layer)
        adata_list3[x] = celltype_fraction(adata_list[x], layer)
    adata_list2 = pd.DataFrame(adata_list2, index=adata_subset_tmp.var_names)
    adata_list3 = pd.DataFrame(adata_list3, index=adata_subset_tmp.var_names)

    decon_subset = deconvoluted[deconvoluted.complex_name.isin(find_complex(interactions_subset))].copy()

    # if any interactions are actually complexes, extract them from the deconvoluted dataframe.
    if decon_subset.shape[0] > 0:
        decon_subset_expr = decon_subset.groupby("complex_name").apply(lambda r: r[adata_list2.columns].apply(np.mean, axis=0))
        cellfrac = defaultdict(dict)
        zgenes = list(set(decon_subset_expr.index))
        for ct, adat in adata_list_alt.items():
            for zg in zgenes:
                cellfrac[ct][zg] = np.mean(adat[:, adata.var_names.isin(zg.split("_"))].X > 0)
        decon_subset_fraction = pd.DataFrame(cellfrac)
        expr_df = pd.concat([adata_list2, decon_subset_expr])
        fraction_df = pd.concat([adata_list3, decon_subset_fraction])
    else:
        expr_df = adata_list2.copy()
        fraction_df = adata_list3.copy()

    # create edge list
    cells_test = list(set(meta[celltype_key]))
    if remove_self:
        cell_type_grid = pd.DataFrame({c: [[cc for cc in cells_test if cc != c]] for c in cells_test}).T
    else:
        cell_type_grid = pd.DataFrame({c: [cells_test] for c in cells_test}).T
    cell_type_grid[0] = cell_type_grid[0].apply(set)
    cell_type_grid = pd.DataFrame(data=list(combinations(cell_type_grid.index.tolist(), 2)), columns=["source", "target"])

    # create the final dataframe for plotting
    dfx = generate_df(
        interactions_subset=interactions_subset,
        cell_type_grid=cell_type_grid,
        cell_type_means=expr_df,
        cell_type_fractions=fraction_df,
        keep_barcodes=barcodes,
        sep=DEFAULT_SEP,
    )
    # ok form the table for pyCircos
    int_value = dict(zip(lr_interactions.barcode, lr_interactions.y_means))
    dfx["interaction_value"] = [int_value[y] if y in int_value else np.nan for y in dfx["barcode"]]
    tmpdf = dfx[["producer", "receiver", "converted_pair", "interaction_value"]].copy()
    tmpdf["interaction_celltype"] = [
        DEFAULT_SEP.join(sorted([a, b, c])) for a, b, c in zip(tmpdf.producer, tmpdf.receiver, tmpdf.converted_pair)
    ]
    celltypes = sorted(list(set(list(tmpdf.producer) + list(tmpdf.receiver))))
    celltype_start_dict = {r: k * gap for k, r in enumerate(celltypes)}
    celltype_end_dict = {r: k + gap for k, r in enumerate(celltypes)}
    interactions = sorted(list(set(tmpdf["interaction_celltype"])))
    interaction_start_dict = {r: k * gap for k, r in enumerate(interactions)}
    interaction_end_dict = {r: k + gap for k, r in enumerate(interactions)}
    tmpdf["from"] = [celltype_start_dict[x] for x in tmpdf.producer]
    tmpdf["to"] = [celltype_end_dict[x] for x in tmpdf.receiver]
    tmpdf["interaction_value"] = [
        j * scale_line + interaction_start_dict[x] if pd.notnull(j) else None
        for j, x in zip(tmpdf.interaction_value, tmpdf.interaction_celltype)
    ]
    tmpdf["start"] = round(tmpdf["interaction_value"] + tmpdf["from"])
    tmpdf["end"] = round(tmpdf["interaction_value"] + tmpdf["to"])
    if col_dict is None:
        uni_interactions = list(set(tmpdf.converted_pair))
        cmap = plt.cm.nipy_spectral
        col_step = 1 / len(uni_interactions)
        start_step = 0
        col_dict = {}
        for i in uni_interactions:
            col_dict[i] = cmap(start_step)
            start_step += col_step
    circle = Gcircle(figsize=figsize)
    circle.set_garcs(-180, 180)
    for i, j in tmpdf.iterrows():
        name = j["producer"]
        arc = Garc(
            arc_id=name, size=size, interspace=interspace, raxis_range=raxis_range, labelposition=labelposition, label_visible=label_visible
        )
        circle.add_garc(arc)
    circle.set_garcs(-180, 180)
    for i, j in tmpdf.iterrows():
        if pd.notnull(j["interaction_value"]):
            lr = j["converted_pair"]
            start_size = j["start"] + j["interaction_value"] / scale_line
            end_size = j["end"] + j["interaction_value"] / scale_line
            start_size = 1 if start_size < 1 else start_size
            end_size = 1 if end_size < 1 else end_size
            source = (j["producer"], j["start"] - 1, start_size, raxis_range[0] - size)
            destination = (j["receiver"], j["end"] - 1, end_size, raxis_range[0] - size)
            circle.chord_plot(source, destination, col_dict[lr])

    custom_lines = [Line2D([0], [0], color=val, lw=4) for val in col_dict.values()]
    circle.figure.legend(custom_lines, col_dict.keys(), frameon=False)
    return circle, dfx
