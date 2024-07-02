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
    scale_alpha_continuous,
    scale_colour_continuous,
    scale_colour_manual,
    scale_fill_continuous,
    scale_fill_manual,
    scale_size_continuous,
    theme,
    theme_bw,
)
from typing import List, Literal, Optional, Union, Tuple, Dict

from ktplotspy.utils.settings import (
    DEFAULT_V5_COL_START,
    DEFAULT_COL_START,
    DEFAULT_CLASS_COL,
    DEFAULT_SEP,
    DEFAULT_SPEC_PAT,
    DEFAULT_CELLSIGN_ALPHA,
    DEFAULT_COLUMNS,
)
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
    interaction_scores: Optional[pd.DataFrame] = None,
    cellsign: Optional[pd.DataFrame] = None,
    degs_analysis: bool = False,
    splitby_key: Optional[str] = None,
    alpha: float = 0.05,
    keep_significant_only: bool = True,
    genes: Optional[Union[List[str], str]] = None,
    gene_family: Optional[Union[List[str], Literal["chemokines", "th1", "th2", "th17", "treg", "costimulatory", "coinhibitory"]]] = None,
    interacting_pairs: Optional[Union[List[str], str]] = None,
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
    exclude_interactions: Optional[Union[List[str], str]] = None,
    title: str = "",
    return_table: bool = False,
    figsize: Tuple[Union[int, float], Union[int, float]] = (6.4, 4.8),
    min_interaction_score: int = 0,
    scale_alpha_by_interaction_scores: bool = False,
    scale_alpha_by_cellsign: bool = False,
    filter_by_cellsign: bool = False,
    keep_id_cp_interaction: bool = False,
    result_precision: int = 3,
) -> Union[ggplot, pd.DataFrame]:
    """Plotting CellPhoneDB results as a dot plot.

    Parameters
    ----------
    adata : AnnData
        `AnnData` object with the `.obs` storing the `celltype_key` with or without `splitby_key`.
        The `.obs_names` must match the first column of the input `meta.txt` used for CellPhoneDB.
    cell_type1 : str
        Name of cell type 1. Accepts regex pattern.
    cell_type2 : str
        Name of cell type 1. Accepts regex pattern.
    means : pd.DataFrame
        Data frame corresponding to `means.txt` from CellPhoneDB.
    pvals : pd.DataFrame
        Data frame corresponding to `pvalues.txt` or `relevant_interactions.txt` from CellPhoneDB.
    celltype_key : str
        Column name in `adata.obs` storing the celltype annotations.
        Values in this column should match the second column of the input `meta.txt` used for CellPhoneDB.
    interaction_scores : Optional[pd.DataFrame], optional
        Data frame corresponding to `interaction_scores.txt` from CellPhoneDB version 5 onwards.
    cellsign : Optional[pd.DataFrame], optional
        Data frame corresponding to `CellSign.txt` from CellPhoneDB version 5 onwards.
    degs_analysis : bool, optional
        Whether CellPhoneDB was run in `deg_analysis` mode.
    splitby_key : Optional[str], optional
        If provided, will attempt to split the output plot/table by groups.
        In order for this to work, the second column of the input `meta.txt` used for CellPhoneDB MUST be this format: {splitby}_{celltype}.
    alpha : float, optional
        P value threshold value for significance.
    keep_significant_only : bool, optional
        Whether or not to trim to significant (p<0.05) hits.
    genes : Optional[Union[List[str], str]], optional
        If provided, will attempt to plot only interactions containing the specified gene(s).
    gene_family : Optional[Union[List[str], Literal["chemokines", "th1", "th2", "th17", "treg", "costimulatory", "coinhibitory"]]], optional
        If provided, will attempt to plot a predetermined set of chemokines or genes associated with Th1, Th2, Th17, Treg, costimulatory or coinhibitory molecules.
    interacting_pairs : Optional[Union[List[str], str]], optional
        If provided, will attempt to plot only interactions containing the specified interacting pair(s). Ignores `genes` and `gene_family` if provided.
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
    min_interaction_score: int, optional
        Filtering the interactions shown by including only those above the given interaction score.
    scale_alpha_by_interaction_scores: bool, optional
        Whether or not to filter values by the interaction score.
    scale_alpha_by_cellsign: bool, optional
        Whether or not to filter the transparency of interactions by the cellsign.
    filter_by_cellsign: bool, optional
        Filter out interactions with a 0 value cellsign.
    keep_id_cp_interaction: bool, optional
        Whether to keep the original `id_cp_interaction` value when plotting.
    result_precision: int, optional
        Sets integer value for decimal points of p_value, default to 3
    Returns
    -------
    Union[ggplot, pd.DataFrame]
        Either a plotnine `ggplot` plot or a pandas `Data frame` holding the results.

    Raises
    ------
    KeyError
        If genes and gene_family are both provided, wrong key for gene family provided, or if interaction_score and cellsign are both provided the error will occur.
    """

    if special_character_regex_pattern is None:
        special_character_regex_pattern = DEFAULT_SPEC_PAT
    # prepare data
    metadata = adata.obs.copy()
    means_mat = prep_table(data=means)
    pvals_mat = prep_table(data=pvals)
    col_start = (
        DEFAULT_V5_COL_START if pvals_mat.columns[DEFAULT_CLASS_COL] == "classification" else DEFAULT_COL_START
    )  # in v5, there are 12 columns before the values
    if pvals_mat.shape != means_mat.shape:
        tmp_pvals_mat = pd.DataFrame(index=means_mat.index, columns=means_mat.columns)
        # Copy the values from means_mat to new_df
        tmp_pvals_mat.iloc[:, :col_start] = means_mat.iloc[:, :col_start]
        tmp_pvals_mat.update(pvals_mat)
        if degs_analysis:
            tmp_pvals_mat.fillna(0, inplace=True)
        else:
            tmp_pvals_mat.fillna(1, inplace=True)
        pvals_mat = tmp_pvals_mat.copy()

    if (interaction_scores is not None) & (cellsign is not None):
        raise KeyError("Please specify either interaction scores or cellsign, not both.")

    if interaction_scores is not None:
        interaction_scores_mat = prep_table(data=interaction_scores)
    elif cellsign is not None:
        cellsign_mat = prep_table(data=cellsign)
    if degs_analysis:
        pvals_mat.iloc[:, col_start : pvals_mat.shape[1]] = 1 - pvals_mat.iloc[:, col_start : pvals_mat.shape[1]]
    # front load the dictionary construction here
    if col_start == DEFAULT_V5_COL_START:
        tmp = means_mat.melt(id_vars=means_mat.columns[:col_start])
        direc, classif, is_int = {}, {}, {}
        for _, r in tmp.iterrows():
            key = r.id_cp_interaction + DEFAULT_SEP * 3 + r.interacting_pair.replace("_", "-") + DEFAULT_SEP * 3 + r.variable
            direc[key] = r.directionality
            classif[key] = r.classification
            is_int[key] = r.is_integrin
    # ensure celltypes are ok
    cell_type1 = sub_pattern(cell_type=cell_type1, pattern=special_character_regex_pattern)
    cell_type2 = sub_pattern(cell_type=cell_type2, pattern=special_character_regex_pattern)
    if interacting_pairs is None:
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
    else:
        # ensure that we convert any hyperlinks to underscores
        interacting_pairs = interacting_pairs if isinstance(interacting_pairs, list) else [interacting_pairs]
        interacting_pairs = [re.sub("-", "_", i) for i in interacting_pairs]
        query = interacting_pairs
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
    ct_columns = [ct for ct in means_mat.columns if re.search(cell_type, ct)]
    # filter
    means_matx = filter_interaction_and_celltype(data=means_mat, genes=query, celltype_pairs=ct_columns)
    pvals_matx = filter_interaction_and_celltype(data=pvals_mat, genes=query, celltype_pairs=ct_columns)
    if interaction_scores is not None:
        interaction_scores_matx = filter_interaction_and_celltype(data=interaction_scores_mat, genes=query, celltype_pairs=ct_columns)
    elif cellsign is not None:
        cellsign_matx = filter_interaction_and_celltype(data=cellsign_mat, genes=query, celltype_pairs=ct_columns)
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
    if interaction_scores is not None:
        interaction_scores_matx = interaction_scores_matx[col_order]
    elif cellsign is not None:
        cellsign_matx = cellsign_matx[col_order]
    # whether or not to filter to only significant hits
    if keep_significant_only:
        keep_rows = pvals_matx.apply(lambda r: any(r < alpha), axis=1)
        keep_rows = [r for r, k in keep_rows.items() if k]
        if len(keep_rows) > 0:
            pvals_matx = pvals_matx.loc[keep_rows]
            means_matx = means_matx.loc[keep_rows]
            if interaction_scores is not None:
                interaction_scores_matx = interaction_scores_matx.loc[keep_rows]
            if cellsign is not None:
                # cellsign data is actually a subset so let's do
                keep_rows = [r for r in keep_rows if r in cellsign_matx.index]
                if len(keep_rows) > 0:
                    cellsign_matx = cellsign_matx.loc[keep_rows]
                else:
                    raise ValueError("Your cellsign data may not contain significant hits.")
    # run hierarchical clustering on the rows based on interaction value.
    if cluster_rows:
        if means_matx.shape[0] > 2:
            h_order = hclust(means_matx, axis=0)
            means_matx = means_matx.loc[h_order]
            pvals_matx = pvals_matx.loc[h_order]
            if interaction_scores is not None:
                interaction_scores_matx = interaction_scores_matx.loc[h_order]
            elif cellsign is not None:
                h_order = [h for h in h_order if h in cellsign_matx.index]
                if len(h_order) > 0:
                    cellsign_matx = cellsign_matx.loc[h_order]
                else:
                    raise ValueError("Your cellsign data may not contain significant hits.")
    if standard_scale:
        means_matx = means_matx.apply(lambda r: (r - np.min(r)) / (np.max(r) - np.min(r)), axis=1)
    means_matx.fillna(0, inplace=True)
    # prepare final table
    colm = "scaled_means" if standard_scale else "means"
    df = means_matx.melt(ignore_index=False).reset_index()
    df.index = df["index"] + DEFAULT_SEP * 3 + df["variable"]
    df.columns = DEFAULT_COLUMNS + [colm]
    df_pvals = pvals_matx.melt(ignore_index=False).reset_index()
    df_pvals.index = df_pvals["index"] + DEFAULT_SEP * 3 + df_pvals["variable"]
    df_pvals.columns = DEFAULT_COLUMNS + ["pvals"]
    df.celltype_group = [re.sub(DEFAULT_SEP, "-", c) for c in df.celltype_group]
    df["pvals"] = df_pvals["pvals"]
    if interaction_scores is not None:
        df_interaction_scores = interaction_scores_matx.melt(ignore_index=False).reset_index()
        df_interaction_scores.index = df_interaction_scores["index"] + DEFAULT_SEP * 3 + df_interaction_scores["variable"]
        df_interaction_scores.columns = DEFAULT_COLUMNS + ["interaction_scores"]
        df["interaction_scores"] = df_interaction_scores["interaction_scores"]
    elif cellsign is not None:
        df_cellsign = cellsign_matx.melt(ignore_index=False).reset_index()
        df_cellsign.index = df_cellsign["index"] + DEFAULT_SEP * 3 + df_cellsign["variable"]
        df_cellsign.columns = DEFAULT_COLUMNS + ["cellsign"]  # same as above.
        df["cellsign"] = df_cellsign["cellsign"]

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
                df.at[i, "pvals"] = 10**-result_precision
        if df.at[i, "pvals"] >= alpha:
            if keep_significant_only:
                df.at[i, "y_means"] = np.nan
        if interaction_scores is not None:
            if df.at[i, "interaction_scores"] < 1:
                df.at[i, "x_means"] = np.nan
        elif cellsign is not None:
            if df.at[i, "cellsign"] < 1:
                df.at[i, "cellsign"] = DEFAULT_CELLSIGN_ALPHA

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
    # append the initial data
    if col_start == DEFAULT_V5_COL_START:
        df["is_integrin"] = [is_int[i] for i in df.index]
        df["directionality"] = [direc[i] for i in df.index]
        df["classification"] = [classif[i] for i in df.index]
    if df.shape[0] == 0:
        raise ValueError("No significant results found.")
    if return_table:
        return df
    else:
        # change the labelling of interaction_group
        if keep_id_cp_interaction:
            df.interaction_group = [re.sub(DEFAULT_SEP * 3, "_", c) for c in df.interaction_group]
        else:
            df.interaction_group = [c.split(DEFAULT_SEP * 3)[1] for c in df.interaction_group]
        # set global figure size
        options.figure_size = figsize

        if highlight_size is not None:
            max_highlight_size = highlight_size
            stroke = "x_stroke"
        else:
            stroke = "neglog10p"

        # plotting
        if interaction_scores is not None:
            df = df[df.interaction_scores >= min_interaction_score]
            if scale_alpha_by_interaction_scores:
                if default_style:
                    g = ggplot(
                        df,
                        aes(
                            x="celltype_group",
                            y="interaction_group",
                            colour="significant",
                            fill=colm,
                            size=colm,
                            stroke=stroke,
                            alpha="interaction_scores",
                        ),
                    )
                else:
                    if all(df["significant"] == "no"):
                        g = ggplot(
                            df,
                            aes(
                                x="celltype_group",
                                y="interaction_group",
                                colour="significant",
                                fill=colm,
                                size=colm,
                                stroke=stroke,
                                alpha="interaction_scores",
                            ),
                        )
                        default_style = True
                    else:
                        highlight_col = "#FFFFFF"  # enforce this
                        g = ggplot(
                            df,
                            aes(
                                x="celltype_group",
                                y="interaction_group",
                                colour=colm,
                                fill="significant",
                                size=colm,
                                stroke=stroke,
                                alpha="interaction_scores",
                            ),
                        )
            else:
                g = None
        else:
            if cellsign is not None:
                if filter_by_cellsign:
                    df = df[df.cellsign >= DEFAULT_CELLSIGN_ALPHA]
                if scale_alpha_by_cellsign:
                    if default_style:
                        g = ggplot(
                            df,
                            aes(
                                x="celltype_group",
                                y="interaction_group",
                                colour="significant",
                                fill=colm,
                                size=colm,
                                stroke=stroke,
                                alpha="cellsign",
                            ),
                        )
                    else:
                        if all(df["significant"] == "no"):
                            g = ggplot(
                                df,
                                aes(
                                    x="celltype_group",
                                    y="interaction_group",
                                    colour="significant",
                                    fill=colm,
                                    size=colm,
                                    stroke=stroke,
                                    alpha="cellsign",
                                ),
                            )
                            default_style = True
                        else:
                            highlight_col = "#FFFFFF"  # enforce this
                            g = ggplot(
                                df,
                                aes(
                                    x="celltype_group",
                                    y="interaction_group",
                                    colour=colm,
                                    fill="significant",
                                    size=colm,
                                    stroke=stroke,
                                    alpha="cellsign",
                                ),
                            )
                else:
                    g = None
            else:
                g = None

        if g is None:
            if default_style:
                g = ggplot(
                    df,
                    aes(
                        x="celltype_group",
                        y="interaction_group",
                        colour="significant",
                        fill=colm,
                        size=colm,
                        stroke=stroke,
                    ),
                )
            else:
                if all(df["significant"] == "no"):
                    g = ggplot(
                        df,
                        aes(
                            x="celltype_group",
                            y="interaction_group",
                            colour="significant",
                            fill=colm,
                            size=colm,
                            stroke=stroke,
                        ),
                    )
                    default_style = True
                else:
                    highlight_col = "#FFFFFF"  # enforce this
                    g = ggplot(
                        df,
                        aes(
                            x="celltype_group",
                            y="interaction_group",
                            colour=colm,
                            fill="significant",
                            size=colm,
                            stroke=stroke,
                        ),
                    )
    g = (
        g
        + geom_point(
            na_rm=True,
        )
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
    if (interaction_scores is not None) and scale_alpha_by_interaction_scores:
        g = g + scale_alpha_continuous(breaks=(0, 25, 50, 75, 100))
    if (cellsign is not None) and scale_alpha_by_cellsign:
        g = g + scale_alpha_continuous(breaks=(0, 1))
    if title != "":
        g = g + ggtitle(title)
    elif gene_family is not None:
        if isinstance(gene_family, list):
            gene_family = ", ".join(gene_family)
        g = g + ggtitle(gene_family)
    return g
