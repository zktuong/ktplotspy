#!/usr/bin/env python
import re

import numpy as np
import pandas as pd

from collections import defaultdict
from itertools import product
from matplotlib.lines import Line2D
from pycirclize import Circos

from ktplotspy.plot import plot_cpdb
from ktplotspy.utils.settings import DEFAULT_PAL, DEFAULT_PAL_CYCLER, DEFAULT_SEP
from ktplotspy.utils.support import celltype_fraction, celltype_means, find_complex, flatten, generate_df


def plot_cpdb_chord(
    adata: "AnnData",
    means: pd.DataFrame,
    pvals: pd.DataFrame,
    deconvoluted: pd.DataFrame,
    celltype_key: str,
    interaction: str | list[str] | None = None,
    cell_type1: str | None = None,
    cell_type2: str | None = None,
    keep_celltypes: list[str] | None = None,
    remove_self: bool = True,
    layer: str | None = None,
    sector_colors: dict[str, str] | None = None,
    sector_text_kwargs: dict = {"color": "black", "size": 8, "r": 105, "adjust_rotation": True},
    sector_radius_limit: tuple[float, float] = (95, 100),
    equal_sector_size: bool = False,
    same_producer_colors: bool = False,
    link_colors: str | dict[str, str] | None = None,
    link_offset: float = 0,
    link_kwargs: dict = {"direction": 1, "allow_twist": True, "r1": 95, "r2": 90},
    legend_width: float = 2,
    legend_kwargs: dict = {"loc": "center", "bbox_to_anchor": (1, 1), "fontsize": 8},
    **plot_cpdb_kwargs,
):
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
    interaction : str | list[str] | None, optional
        Interaction(s) to plot. If None, all interactions will be plotted. If provided as a '-' separated string between L-R pair, only the specific interaction(s) will be plotted (more than 1 pair is accepted).
        If provided as a list of strings without '-' separator, all relevant interactions in the list will be plotted.
    cell_type1 : str | None, optional
        Name of cell type 1. Accepts regex pattern. None defaults to "." i.e. all cell types. If both cell_type1 and cell_type2 are provided, only interactions contained in these cell types will be plotted.
    cell_type2 : str | None, optional
        Name of cell type 2. Accepts regex pattern. None defaults to "." i.e. all cell types. If both cell_type1 and cell_type2 are provided, only interactions contained in these cell types will be plotted.
    keep_celltypes : list[str] | None, optional
        If provided, only interactions to/from these celltypes will be plotted.
    remove_self : bool, optional
        whether to remove self edges.
    layer : str | None, optional
        slot in `AnnData.layers` to access. If `None`, uses `.X`.
    sector_colors : dict[str, str] | None, optional
        dictionary of celltype (sector) colours.
        If not provided, will try and use `.uns` from `adata` if correct slot is present.
    sector_text_kwargs : dict, optional
        additional arguments for `track.text`. See https://moshi4.github.io/pyCirclize/api-docs/sector/#pycirclize.sector.Sector.text
    sector_radius_limit : tuple[float, float], optional
        radius limits for the sectors.
    equal_sector_size : bool, optional
        whether to use equal sector sizes.
    same_producer_colors : bool, optional
        whether to use the same colours for sector and links for outgoing interactions.
    link_colors : dict[str, str] | None, optional
        String or dictionary of L-R interaction colours. If not provided, will use a default colour, will use a random
        colour for each unique interaction.
    link_offset : float, optional
        offset to add to each interaction value.
    link_kwargs : dict, optional
        additional arguments for `circos.link`. See https://moshi4.github.io/pyCirclize/api-docs/circos/#pycirclize.circos.Circos.link
    legend_width : float, optional
        width of the legend line.
    legend_kwargs : dict, optional
        additional arguments for `Axes.Legend`. See https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html
    **plot_cpdb_kwargs
        passed to `plot_cpdb`.

    Returns
    -------
    Gcircle
        a `Gcircle` object from `pycircos`.
    """
    # assert splitby = False
    splitby_key, return_table = None, True
    if interaction is not None:
        if isinstance(interaction, str):
            # check if there's '-' in the interaction
            if "-" in interaction:
                lr_intx = interaction.split("-")
            else:
                lr_intx = [interaction]
        elif isinstance(interaction, list):
            lr_intx = []
            for intx in interaction:
                if "-" in intx:
                    tmp_lr_intx = intx.split("-")
                    for tmp_intx in tmp_lr_intx:
                        lr_intx.append(tmp_intx)
                else:
                    lr_intx.append(intx)
    else:
        lr_intx = None
    cell_type1 = cell_type1 if cell_type1 is not None else "."
    cell_type2 = cell_type2 if cell_type2 is not None else "."
    # run plot_cpdb
    lr_interactions = plot_cpdb(
        adata=adata,
        means=means,
        pvals=pvals,
        genes=lr_intx,
        celltype_key=celltype_key,
        return_table=return_table,
        splitby_key=splitby_key,
        cell_type1=cell_type1,
        cell_type2=cell_type2,
        **plot_cpdb_kwargs,
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
        complex_idx2 = [i for i, j in _interactions_subset.partner_a.items() if re.search("complex:", j) and i not in complex_idx1]
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
    celltypes = (
        adata.obs[celltype_key].cat.categories if adata.obs[celltype_key].dtype.name == "category" else list(set(adata.obs[celltype_key]))
    )
    interactions = sorted(list(set(tmpdf["interaction_celltype"])))
    interaction_start_dict = {r: k for k, r in enumerate(interactions)}
    tmpdf["interaction_value"] = [
        j + interaction_start_dict[x] if pd.notnull(j) else np.nan for j, x in zip(tmpdf.interaction_value, tmpdf.interaction_celltype)
    ]
    # filter to specific interactions if interaction is provided with '-'
    if interaction is not None:
        if isinstance(interaction, str):
            if "-" in interaction:
                tmpdf = tmpdf[tmpdf.converted_pair.isin([interaction])]
            else:
                tmpdf = tmpdf[tmpdf.converted_pair.str.contains(interaction)]
        elif isinstance(interaction, list):
            tmpint = []
            for intx in interaction:
                if "-" in interaction:
                    tmpint.append(interaction)
            if len(tmpint) > 0:
                tmpdf = tmpdf[tmpdf.converted_pair.isin([tmpint])]
            else:
                tmpdf = tmpdf[tmpdf.converted_pair.str.contains("|".join(lr_intx))]
    if link_colors is None:
        uni_interactions = list(set(tmpdf.converted_pair))
        if len(uni_interactions) > len(DEFAULT_PAL):
            link_colors = dict(zip(uni_interactions, [next(DEFAULT_PAL_CYCLER) for i in range(len(uni_interactions))]))
        else:
            link_colors = dict(zip(uni_interactions, [DEFAULT_PAL[i] for i in range(len(uni_interactions))]))

    celltypes = sorted(list(set(list(tmpdf.producer) + list(tmpdf.receiver))))
    # Sort the 'converted_pair' values alphabetically
    tmpdf = tmpdf.sort_values("converted_pair")
    # add optional offset to each interaction_value and rounding to the nearest integer
    tmpdf["interaction_value"] = round(tmpdf["interaction_value"].fillna(0) + link_offset)
    # For each celltype ('producer'), create a cumulative interaction value tracker
    cumulative_interaction_dict = defaultdict(int)
    start_values, end_values = [], []
    for _, row in tmpdf.iterrows():
        start_values.append(cumulative_interaction_dict[row["producer"]])
        if row["interaction_value"] > 0:
            cumulative_interaction_dict[row["producer"]] += row["interaction_value"]
        end_values.append(cumulative_interaction_dict[row["producer"]])
    # Add the 'start' and 'end' columns
    tmpdf["start"] = start_values
    tmpdf["end"] = end_values
    # Set the starting value for the receiver
    rec_start = {}
    for celltype in celltypes:
        rec_start[celltype] = max(tmpdf[tmpdf.producer == celltype].end)
    # For each row in the dataframe, calculate the 'start' and 'end' values for the receiver
    start_values, end_values = [], []
    for _, r in tmpdf.iterrows():
        start_values.append(rec_start[r.receiver])
        rec_start[r.receiver] += r.interaction_value
        end_values.append(rec_start[r.receiver])
    # This is the position of the end of the links (i.e. the target)
    tmpdf["start_2"] = start_values
    tmpdf["end_2"] = end_values
    sectors = {}
    # the maximum size of each sector is the maximum value of 'end_2' for each celltype
    for celltype in celltypes:
        tmp = tmpdf[tmpdf.receiver == celltype]
        if tmp.shape[0] > 0:
            sectors[celltype] = max(tmp.end_2)
        else:
            tmp = tmpdf[tmpdf.producer == celltype]
            if tmp.shape[0] > 0:
                sectors[celltype] = max(tmp.end_2)
    if equal_sector_size:
        max_sector = max(sectors.values())
        sectors = {k: max_sector for k in sectors}
    else:
        # offset the sector by 10 if any of the values are 0
        for sector in sectors:
            if sectors[sector] == 0:
                sectors[sector] = 10
    tmpdf = tmpdf[tmpdf.interaction_value > link_offset]
    tmpdf = tmpdf.reset_index(drop=True)
    if keep_celltypes is not None:
        tmpdf["celltype_test1"] = tmpdf["producer"] + DEFAULT_SEP + tmpdf["receiver"]
        tmpdf["celltype_test2"] = tmpdf["receiver"] + DEFAULT_SEP + tmpdf["producer"]
        if len(keep_celltypes) == 2:
            test_celltype = keep_celltypes[0] + DEFAULT_SEP + keep_celltypes[1]
            tmpdf = tmpdf[tmpdf["celltype_test1"].isin([test_celltype]) | tmpdf["celltype_test2"].isin([test_celltype])]
            tmpdf = tmpdf.drop(["celltype_test1", "celltype_test2"], axis=1)
        elif len(keep_celltypes) > 2:
            tests = list(product(keep_celltypes, keep_celltypes))
            test2 = []
            for test in tests:
                test2.append(test[0] + DEFAULT_SEP + test[1])
            tmpdf = tmpdf[tmpdf["celltype_test1"].isin(test2) | tmpdf["celltype_test2"].isin(test2)]
            tmpdf = tmpdf.drop(["celltype_test1", "celltype_test2"], axis=1)
        else:
            keep_celltypes = [keep_celltypes] if isinstance(keep_celltypes, str) else keep_celltypes
            tmpdf = tmpdf[tmpdf.producer.isin(keep_celltypes) | tmpdf.receiver.isin(keep_celltypes)]
    if sector_colors is None:
        if celltype_key + "_colors" in adata.uns:
            if adata.obs[celltype_key].dtype.name == "category":
                sector_colors = dict(zip(adata.obs[celltype_key].cat.categories, adata.uns[celltype_key + "_colors"]))
            else:
                sector_colors = dict(zip(list(set(adata.obs[celltype_key])), adata.uns[celltype_key + "_colors"]))
        else:
            if len(celltypes) > len(DEFAULT_PAL):
                sector_colors = dict(zip(celltypes, [next(DEFAULT_PAL_CYCLER) for i in range(len(celltypes))]))
            else:
                sector_colors = dict(zip(celltypes, [DEFAULT_PAL[i] for i in range(len(celltypes))]))
    for celltype in celltypes:
        if celltype not in sector_colors:
            sector_colors[celltype] = "#f7f7f7"  # just make it off-white if not present
    circos = Circos(sectors, space=5)
    for sector in circos.sectors:
        track = sector.add_track(r_lim=sector_radius_limit)
        track.axis(fc=sector_colors[sector.name])
        track.text(text=sector.name, **sector_text_kwargs)
    for _, r in tmpdf.iterrows():
        if not same_producer_colors:
            if link_colors is not None:
                if isinstance(link_colors, dict):
                    link_color = link_colors[r.converted_pair] if r.converted_pair in link_colors else "#f7f7f700"
                elif isinstance(link_colors, str):
                    link_color = link_colors
        else:
            link_color = sector_colors[r.producer]
        circos.link(
            sector_region1=(r.producer, r.start, r.end), sector_region2=(r.receiver, r.start_2, r.end_2), color=link_color, **link_kwargs
        )
    circos.plotfig()
    # plot legend
    if isinstance(link_colors, dict):
        line_handles = [Line2D([], [], color=value, label=key, linewidth=legend_width) for key, value in link_colors.items()]
        line_legend = circos.ax.legend(handles=line_handles, **legend_kwargs)
        circos.ax.add_artist(line_legend)

    return circos
