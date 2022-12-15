#!/usr/bin/env python
import numpy as np
import pandas as pd
import re

from plotnine import (
    aes,
    element_blank,
    element_rect,
    element_text,
    geom_point,
    ggplot,
    ggtitle,
    guide_colourbar,
    guide_legend,
    guides,
    options,
    scale_colour_continuous,
    scale_colour_manual,
    scale_fill_continuous,
    scale_fill_manual,
    scale_size_continuous,
    theme,
    theme_bw,
)
from typing import List, Literal, Optional, Union, Tuple, Dict

from ktplotspy.utils.settings import DEFAULT_SEP, DEFAULT_SPEC_PAT
from ktplotspy.utils.support import (
    ensure_categorical,
    filter_interaction_and_celltype,
    hclust,
    prep_celltype_query,
    prep_query_group,
    prep_table,
    set_x_stroke,
    sub_pattern,
)


def plot_cpdb(
    adata: "AnnData",
    cell_type1: str,
    cell_type2: str,
    means: pd.DataFrame,
    pvals: pd.DataFrame,
    celltype_key: str,
    degs_analysis: bool = False,
    splitby_key: Optional[str] = None,
    alpha: float = 0.05,
    keep_significant_only: bool = True,
    genes: Optional[str] = None,
    gene_family: Optional[Literal["chemokines", "th1", "th2", "th17", "treg", "costimulatory", "coinhibitory"]] = None,
    custom_gene_family: Optional[Dict[str, List[str]]] = None,
    standard_scale: bool = True,
    cluster_rows: bool = True,
    cmap_name: str = "viridis",
    max_size: int = 8,
    max_highlight_size: int = 3,
    default_style: bool = True,
    highlight_col: str = "#d62728",
    highlight_size: Optional[int] = None,
    special_character_regex_pattern: Optional[str] = None,
    exclude_interactions: Optional[Union[List, str]] = None,
    title: str = "",
    return_table: bool = False,
    figsize: Tuple[Union[int, float], Union[int, float]] = (6.4, 4.8),
) -> Union[ggplot, pd.DataFrame]:
    """Plotting cellphonedb results as a dot plot.

    Parameters
    ----------
    adata : AnnData
        `AnnData` object with the `.obs` storing the `celltype_key` with or without `splitby_key`.
        The `.obs_names` must match the first column of the input `meta.txt` used for `cellphonedb`.
    cell_type1 : str
        Name of cell type 1. Accepts regex pattern.
    cell_type2 : str
        Name of cell type 1. Accepts regex pattern.
    means : pd.DataFrame
        Dataframe corresponding to `means.txt` from cellphonedb.
    pvals : pd.DataFrame
        Dataframe corresponding to `pvalues.txt` or `relevant_interactions.txt` from cellphonedb.
    celltype_key : str
        Column name in `adata.obs` storing the celltype annotations.
        Values in this column should match the second column of the input `meta.txt` used for `cellphonedb`.
    degs_analysis : bool, optional
        Whether `cellphonedb` was run in `deg_analysis` mode.
    splitby_key : Optional[str], optional
        If provided, will attempt to split the output plot/table by groups.
        In order for this to work, the second column of the input `meta.txt` used for `cellphonedb` MUST be this format: {splitby}_{celltype}.
    alpha : float, optional
        P value threshold value for significance.
    keep_significant_only : bool, optional
        Whether or not to trim to significant (p<0.05) hits.
    genes : Optional[str], optional
        If provided, will attempt to plot only interactions containing the specified gene(s).
    gene_family : Optional[Literal["chemokines", "th1", "th2", "th17", "treg", "costimulatory", "coinhibitory"]], optional
        If provided, will attempt to plot a predetermined set of chemokines or genes associated with Th1, Th2, Th17, Treg, costimulatory or coinhibitory molecules.
    custom_gene_family : Optional[Dict[str, List[str]]], optional
        If provided, will update the gene_family dictionary with this custom dictionary.
        Both `gene_family` (name of the custom family) and `custom_gene_family` (dictionary holding this new family)
        must be specified for this to work.
    standard_scale : bool, optional
        Whether or not to scale the mean interaction values from 0 to 1 per receptor-ligand variable.
    cluster_rows : bool, optional
        Whether or not to cluster the rows (interactions).
    cmap_name : str, optional
        Matplotlib built-in colormap names.
    max_size : int, optional
        Maximum size of points in plot.
    max_highlight_size : int, optional
        Maximum highlight size of points in plot.
    default_style : bool, optional
        Whether or not to plot in default style or inspired from `squidpy`'s plotting style.
    highlight_col : str, optional
        Colour of highlights marking significant hits.
    highlight_size : Optional[int], optional
        Size of highlights marking significant hits.
    special_character_regex_pattern : Optional[str], optional
        Regex string pattern to perform substitution.
        This option should not realy be used unless there is really REALLY special characters that you really REALLY want to keep.
        Rather than using this option, the easiest way is to not your celltypes with weird characters.
        Just use alpha numeric characters and underscores if necessary.
    exclude_interactions : Optional[Union[List, str]], optional
        If provided, the interactions will be removed from the output.
    title : str, optional
        Plot title.
    return_table : bool, optional
        Whether or not to return the results as a dataframe.
    figsize : Tuple[Union[int, float], Union[int, float]], optional
        Figure size.

    Returns
    -------
    Union[ggplot, pd.DataFrame]
        Either a plotnine `ggplot` plot or a pandas `Dataframe` holding the results.

    Raises
    ------
    KeyError
        If genes and gene_family are both provided, or wrong key for gene family provided, the error will occur.
    """
    if special_character_regex_pattern is None:
        special_character_regex_pattern = DEFAULT_SPEC_PAT
    swapr = True if (cell_type1 == ".") or (cell_type2 == ".") else False
    # prepare data
    metadata = adata.obs.copy()
    means_mat = prep_table(data=means)
    pvals_mat = prep_table(data=pvals)
    if degs_analysis:
        pvals_mat.iloc[:, 12 : pvals_mat.shape[1]] = 1 - pvals_mat.iloc[:, 12 : pvals_mat.shape[1]]
    # ensure celltypes are ok
    cell_type1 = sub_pattern(cell_type=cell_type1, pattern=special_character_regex_pattern)
    cell_type2 = sub_pattern(cell_type=cell_type2, pattern=special_character_regex_pattern)
    # check for query
    if genes is None:
        if gene_family is not None:
            query_group = prep_query_group(means_mat, custom_gene_family)
            if isinstance(gene_family, list):
                query = []
                for gf in gene_family:
                    if gf.lower() in query_group:
                        for gfg in query_group[gf.lower()]:
                            query.append(gfg)
                    else:
                        raise KeyError("gene_family needs to be one of the following: {}".format(query_group.keys()))
                query = list(set(query))
            else:
                if gene_family.lower() in query_group:
                    query = query_group[gene_family.lower()]
                else:
                    raise KeyError("gene_family needs to be one of the following: {}".format(query_group.keys()))
        else:
            query = [i for i in means_mat.interacting_pair if re.search("", i)]
    elif genes is not None:
        if gene_family is not None:
            raise KeyError("Please specify either genes or gene_family, not both.")
        else:
            query = [i for i in means_mat.interacting_pair if re.search("|".join(genes), i)]
    metadata = ensure_categorical(meta=metadata, key=celltype_key)
    # prepare regex query for celltypes
    if splitby_key is not None:
        metadata = ensure_categorical(meta=metadata, key=splitby_key)
        groups = list(metadata[splitby_key].cat.categories)
        metadata["_labels"] = [s + "_" + c for s, c in zip(metadata[splitby_key], metadata[celltype_key])]
        metadata["_labels"] = metadata["_labels"].astype("category")
        cat_orders = []
        for s in metadata[splitby_key].cat.categories:
            for c in metadata[celltype_key].cat.categories:
                cat_orders.append(s + "_" + c)
        cat_orders = [x for x in cat_orders if x in list(metadata._labels)]
        metadata["_labels"] = metadata["_labels"].cat.reorder_categories(cat_orders)
        celltype = prep_celltype_query(
            meta=metadata,
            cell_type1=cell_type1,
            cell_type2=cell_type2,
            pattern=special_character_regex_pattern,
            split_by=splitby_key,
        )
    else:
        metadata["_labels"] = metadata[celltype_key]
        celltype = prep_celltype_query(
            meta=metadata,
            cell_type1=cell_type1,
            cell_type2=cell_type2,
            pattern=special_character_regex_pattern,
            split_by=splitby_key,
        )
    cell_type = "|".join(celltype)
    # keep cell types
    if swapr:
        ct_columns = [ct for ct in means_mat.columns if re.search(ct, cell_type)]
    else:
        ct_columns = [ct for ct in means_mat.columns if re.search(cell_type, ct)]
    # filter
    means_matx = filter_interaction_and_celltype(data=means_mat, genes=query, celltype_pairs=ct_columns)
    pvals_matx = filter_interaction_and_celltype(data=pvals_mat, genes=query, celltype_pairs=ct_columns)
    # reorder the columns
    col_order = []
    if splitby_key is not None:
        for g in groups:
            for c in means_matx.columns:
                if re.search(g, c):
                    col_order.append(c)
    else:
        col_order = means_matx.columns
    means_matx = means_matx[col_order]
    pvals_matx = pvals_matx[col_order]
    # whether or not to fillter to only significant hits
    if keep_significant_only:
        keep_rows = pvals_matx.apply(lambda r: any(r < alpha), axis=1)
        keep_rows = [r for r, k in keep_rows.iteritems() if k]
        if len(keep_rows) > 0:
            pvals_matx = pvals_matx.loc[keep_rows]
            means_matx = means_matx.loc[keep_rows]
    # reun hierarchical clustering on the rows based on interaction value.
    if cluster_rows:
        if means_matx.shape[0] > 2:
            h_order = hclust(means_matx, axis=0)
            means_matx = means_matx.loc[h_order]
            pvals_matx = pvals_matx.loc[h_order]
    if standard_scale:
        means_matx = means_matx.apply(lambda r: (r - np.min(r)) / (np.max(r) - np.min(r)), axis=1)
    means_matx.fillna(0, inplace=True)
    # prepare final table
    colm = "scaled_means" if standard_scale else "means"
    df = means_matx.melt(ignore_index=False).reset_index()
    df.columns = ["interaction_group", "celltype_group", colm]
    df_pvals = pvals_matx.melt(ignore_index=False).reset_index()
    df_pvals.columns = ["interaction_group", "celltype_group", "pvals"]
    df.celltype_group = [re.sub(DEFAULT_SEP, "-", c) for c in df.celltype_group]
    df["pvals"] = df_pvals["pvals"]
    # set factors
    df.celltype_group = df.celltype_group.astype("category")
    # prepare for non-default style plotting
    for i in df.index:
        if df.at[i, colm] == 0:
            df.at[i, colm] = np.nan
    df["x_means"] = df[colm]
    df["y_means"] = df[colm]
    for i in df.index:
        if df.at[i, "pvals"] < alpha:
            df.at[i, "x_means"] = np.nan
            if df.at[i, "pvals"] == 0:
                df.at[i, "pvals"] = 0.001
        if df.at[i, "pvals"] >= alpha:
            if keep_significant_only:
                df.at[i, "y_means"] = np.nan
    df["x_stroke"] = df["x_means"]
    set_x_stroke(df=df, isnull=False, stroke=0)
    set_x_stroke(df=df, isnull=True, stroke=highlight_size)
    if exclude_interactions is not None:
        if not isinstance(exclude_interactions, list):
            exclude_interactions = [exclude_interactions]
        df = df[~df.interaction_group.isin(exclude_interactions)]
    df["neglog10p"] = abs(-1 * np.log10(df.pvals))
    df["neglog10p"] = [0 if x >= 0.05 else j for x, j in zip(df["pvals"], df["neglog10p"])]
    df["significant"] = ["yes" if x < alpha else np.nan for x in df.pvals]
    if all(pd.isnull(df["significant"])):
        df["significant"] = "no"
        highlight_col = "#FFFFFF"
    if return_table:
        return df
    else:
        # set global figure size
        options.figure_size = figsize
        if highlight_size is not None:
            max_highlight_size = highlight_size
            stroke = df.x_stroke
        else:
            stroke = df.neglog10p
        # plotting
        if default_style:
            g = ggplot(df, aes(x="celltype_group", y="interaction_group", colour="significant", fill=colm, size=colm, stroke=stroke))
        else:
            if all(df["significant"] == "no"):
                g = ggplot(df, aes(x="celltype_group", y="interaction_group", colour="significant", fill=colm, size=colm, stroke=stroke))
                default_style = True
            else:
                highlight_col = "#FFFFFF"  # enforce this
                g = ggplot(df, aes(x="celltype_group", y="interaction_group", colour=colm, fill="significant", size=colm, stroke=stroke))
        g = (
            g
            + geom_point(na_rm=True)
            + theme_bw()
            + theme(
                axis_text_x=element_text(angle=90, hjust=0, colour="#000000"),
                axis_text_y=element_text(colour="#000000"),
                axis_ticks=element_blank(),
                axis_title_x=element_blank(),
                axis_title_y=element_blank(),
                legend_key=element_rect(alpha=0, width=0, height=0),
                legend_direction="vertical",
                legend_box="horizontal",
            )
            + scale_size_continuous(range=(0, max_size), aesthetics=["size"])
            + scale_size_continuous(range=(0, max_highlight_size), aesthetics=["stroke"])
        )
        if default_style:
            g = (
                g
                + scale_colour_manual(values=highlight_col, na_translate=False)
                + guides(
                    fill=guide_colourbar(barwidth=4, label=True, ticks=True, draw_ulim=True, draw_llim=True, order=1),
                    size=guide_legend(
                        reverse=True,
                        order=2,
                    ),
                    stroke=guide_legend(
                        reverse=True,
                        order=3,
                    ),
                )
                + scale_fill_continuous(cmap_name=cmap_name)
            )
        else:
            g = (
                g
                + scale_fill_manual(values=highlight_col, na_translate=False)
                + guides(
                    colour=guide_colourbar(barwidth=4, label=True, ticks=True, draw_ulim=True, draw_llim=True, order=1),
                    size=guide_legend(
                        reverse=True,
                        order=2,
                    ),
                    stroke=guide_legend(
                        reverse=True,
                        order=3,
                    ),
                )
            )
            df2 = df.copy()
            for i in df2.index:
                if df2.at[i, "pvals"] < alpha:
                    df2.at[i, colm] = np.nan
            g = (
                g
                + geom_point(aes(x="celltype_group", y="interaction_group", colour=colm, size=colm), df2, inherit_aes=False, na_rm=True)
                + scale_colour_continuous(cmap_name=cmap_name)
            )
        if highlight_size is not None:
            g = g + guides(stroke=None)
        if title != "":
            g = g + ggtitle(title)
        elif gene_family is not None:
            if isinstance(gene_family, list):
                gene_family = ", ".join(gene_family)
            g = g + ggtitle(gene_family)
        return g
