#!/usr/bin/env python
import re
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from collections import defaultdict
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from pycircos import Garc, Gcircle
from typing import Optional, Tuple, Dict, Union

from ktplotspy.utils.settings import DEFAULT_SEP  # DEFAULT_PAL
from ktplotspy.utils.support import celltype_fraction, celltype_means, find_complex, flatten, generate_df, present
from ktplotspy.plot import plot_cpdb


def plot_cpdb_chord(
    adata: "AnnData",
    means: pd.DataFrame,
    pvals: pd.DataFrame,
    deconvoluted: pd.DataFrame,
    celltype_key: str,
    face_col_dict: Optional[Dict[str, str]] = None,
    edge_col_dict: Optional[Dict[str, str]] = None,
    edge_cmap: LinearSegmentedColormap = plt.cm.nipy_spectral,
    remove_self: bool = True,
    gap: Union[int, float] = 2,
    scale_lw: Union[int, float] = 10,
    size: Union[int, float] = 50,
    interspace: Union[int, float] = 2,
    raxis_range: Tuple[int, int] = (950, 1000),
    labelposition: Union[int, float] = 80,
    label_visible: bool = True,
    figsize: Tuple[Union[int, float], Union[int, float]] = (8, 8),
    legend_params: Dict = {"loc": "center left", "bbox_to_anchor": (1, 1), "frameon": False},
    layer: Optional[str] = None,
    **kwargs
) -> Gcircle:
    """Plotting cellphonedb results as a chord diagram.

    Parameters
    ----------
    adata : AnnData
        `AnnData` object with the `.obs` storing the `celltype_key` with or without `splitby_key`.
        The `.obs_names` must match the first column of the input `meta.txt` used for `cellphonedb`.
    means : pd.DataFrame
        Dataframe corresponding to `means.txt` from cellphonedb.
    pvals : pd.DataFrame
        Dataframe corresponding to `pvalues.txt` or `relevant_interactions.txt` from cellphonedb.
    deconvoluted : pd.DataFrame
        Dataframe corresponding to `deconvoluted.txt` from cellphonedb.
    celltype_key : str
        Column name in `adata.obs` storing the celltype annotations.
        Values in this column should match the second column of the input `meta.txt` used for `cellphonedb`.
    face_col_dict : Optional[Dict[str, str]], optional
        dictionary of celltype : face colours.
        If not provided, will try and use `.uns` from `adata` if correct slot is present.
    edge_col_dict : Optional[Dict[str, str]], optional
        Dictionary of interactions : edge colours. Otherwise, will use edge_cmap option.
    edge_cmap : LinearSegmentedColormap, optional
        a `LinearSegmentedColormap` to generate edge colors.
    remove_self : bool, optional
        whether to remove self edges.
    gap : Union[int, float], optional
        relative size of gaps between edges on arc.
    scale_lw : Union[int, float], optional
        numeric value to scale width of lines.
    size : Union[int, float], optional
        Width of the arc section. If record is provided, the value is
        instead set by the sequence length of the record. In reality
        the actual arc section width in the resultant circle is determined
        by the ratio of size to the combined sum of the size and interspace
        values of the Garc class objects in the Gcircle class object.
    interspace : Union[int, float], optional
        Distance angle (deg) to the adjacent arc section in clockwise
        sequence. The actual interspace size in the circle is determined by
        the actual arc section width in the resultant circle is determined
        by the ratio of size to the combined sum of the size and interspace
        values of the Garc class objects in the Gcircle class object.
    raxis_range : Tuple[int, int], optional
        Radial axis range where line plot is drawn.
    labelposition : Union[int, float], optional
        Relative label height from the center of the arc section.
    label_visible : bool, optional
        Font size of the label. The default is 10.
    figsize : Tuple[Union[int, float], Union[int, float]], optional
        size of figure.
    legend_params : Dict, optional
        additional arguments for `plt.legend`.
    layer : Optional[str], optional
        slot in `AnnData.layers` to access. If `None`, uses `.X`.
    **kwargs
        passed to `plot_cpdb`.

    Returns
    -------
    Gcircle
        a `Gcircle` object from `pycircos`.
    """
    # assert splitby = False
    splitby_key, return_table = None, True
    # run plot_cpdb
    lr_interactions = plot_cpdb(
        adata=adata,
        means=means,
        pvals=pvals,
        celltype_key=celltype_key,
        return_table=return_table,
        splitby_key=splitby_key,
        **kwargs,
    )
    # do some name wrangling
    subset_clusters = list(set(flatten([x.split("-") for x in lr_interactions.celltype_group])))
    adata_subset = adata[adata.obs[celltype_key].isin(subset_clusters)].copy()
    interactions = means[
        ["id_cp_interaction", "interacting_pair", "gene_a", "gene_b", "partner_a", "partner_b", "receptor_a", "receptor_b"]
    ].copy()
    interactions["use_interaction_name"] = [
        x + DEFAULT_SEP * 3 + y for x, y in zip(interactions.id_cp_interaction, interactions.interacting_pair)
    ]
    # interactions["converted"] = [re.sub("-", " ", x) for x in interactions.use_interaction_name]
    interactions["converted"] = [re.sub("_", "-", x) for x in interactions.use_interaction_name]
    lr_interactions["barcode"] = [a + DEFAULT_SEP + b for a, b in zip(lr_interactions.celltype_group, lr_interactions.interaction_group)]
    interactions_subset = interactions[interactions["converted"].isin(list(lr_interactions.interaction_group))].copy()
    # handle complexes gently
    tm0 = {kx: rx.split("_") for kx, rx in interactions_subset.use_interaction_name.items()}
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
        tm0.id_a = [x.split(DEFAULT_SEP * 3)[1] for x in tm0.id_a]
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
                zg_mask = adata.var_names.isin(zg.split("_"))
                cellfrac[ct][zg] = np.mean(adat[:, zg_mask].X > 0) if np.sum(zg_mask) > 0 else 0
        decon_subset_fraction = pd.DataFrame(cellfrac)
        expr_df = pd.concat([adata_list2, decon_subset_expr])
        fraction_df = pd.concat([adata_list3, decon_subset_fraction])
    else:
        expr_df = adata_list2.copy()
        fraction_df = adata_list3.copy()

    # create edge list
    cells_test = list(set(meta[celltype_key]))
    cell_comb = []
    for c1 in cells_test:
        for c2 in cells_test:
            if remove_self:
                if c1 != c2:
                    cell_comb.append((c1, c2))
            else:
                cell_comb.append((c1, c2))
    cell_comb = list(set(cell_comb))
    cell_type_grid = pd.DataFrame(cell_comb, columns=["source", "target"])
    # create the final dataframe for plotting
    dfx = generate_df(
        interactions_subset=interactions_subset,
        cell_type_grid=cell_type_grid,
        cell_type_means=expr_df,
        cell_type_fractions=fraction_df,
        sep=DEFAULT_SEP,
    )
    # ok form the table for pyCircos
    int_value = dict(zip(lr_interactions.barcode, lr_interactions.y_means))
    int_value = {k: r for k, r in int_value.items() if pd.notnull(r)}
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
        j * scale_lw + interaction_start_dict[x] if pd.notnull(j) else np.nan
        for j, x in zip(tmpdf.interaction_value, tmpdf.interaction_celltype)
    ]
    tmpdf["start"] = round(tmpdf["interaction_value"] + tmpdf["from"])
    tmpdf["end"] = round(tmpdf["interaction_value"] + tmpdf["to"])
    if edge_col_dict is None:
        uni_interactions = list(set(tmpdf.converted_pair))
        col_step = 1 / len(uni_interactions)
        start_step = 0
        edge_col_dict = {}
        for i in uni_interactions:
            edge_col_dict[i] = edge_cmap(start_step)
            start_step += col_step
    circle = Gcircle(figsize=figsize)
    if face_col_dict is None:
        if celltype_key + "_colors" in adata.uns:
            if adata.obs[celltype_key].dtype.name == "category":
                face_col_dict = dict(zip(adata.obs[celltype_key].cat.categories, adata.uns[celltype_key + "_colors"]))
            else:
                face_col_dict = dict(zip(list(set(adata.obs[celltype_key])), adata.uns[celltype_key + "_colors"]))
    for i, j in tmpdf.iterrows():
        name = j["producer"]
        if face_col_dict is None:
            col = None
        else:
            # col = face_col_dict[name] if name in face_col_dict else next(DEFAULT_PAL) # cycle through the default palette
            col = face_col_dict[name] if name in face_col_dict else "#e7e7e7"  # or just make them grey?
        arc = Garc(
            arc_id=name,
            size=size,
            interspace=interspace,
            raxis_range=raxis_range,
            labelposition=labelposition,
            label_visible=label_visible,
            facecolor=col,
        )
        circle.add_garc(arc)
    circle.set_garcs(-180, 180)
    for i, j in tmpdf.iterrows():
        if pd.notnull(j["interaction_value"]):
            lr = j["converted_pair"]
            start_size = j["start"] + j["interaction_value"] / scale_lw
            end_size = j["end"] + j["interaction_value"] / scale_lw
            start_size = 1 if start_size < 1 else start_size
            end_size = 1 if end_size < 1 else end_size
            source = (j["producer"], j["start"] - 1, start_size, raxis_range[0] - size)
            destination = (j["receiver"], j["end"] - 1, end_size, raxis_range[0] - size)
            circle.chord_plot(source, destination, edge_col_dict[lr] if lr in edge_col_dict else "#f7f7f700")

    custom_lines = [Line2D([0], [0], color=val, lw=4) for val in edge_col_dict.values()]
    circle.figure.legend(custom_lines, edge_col_dict.keys(), **legend_params)
    return circle
