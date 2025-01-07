#!/usr/bin/env python
import numpy as np
import pandas as pd
import seaborn as sns

from itertools import product
from matplotlib.colors import ListedColormap
from typing import Optional, Union, Dict, List

from ktplotspy.utils.support import diverging_palette
from ktplotspy.utils.settings import DEFAULT_V5_COL_START, DEFAULT_COL_START, DEFAULT_CLASS_COL, DEFAULT_CPDB_SEP


def plot_cpdb_heatmap(
    pvals: pd.DataFrame,
    cell_types: Optional[List[str]] = None,
    degs_analysis: bool = False,
    log1p_transform: bool = False,
    alpha: float = 0.05,
    linewidths: float = 0.5,
    row_cluster: bool = True,
    col_cluster: bool = True,
    low_col: str = "#104e8b",
    mid_col: str = "#ffdab9",
    high_col: str = "#8b0a50",
    cmap: Optional[Union[str, ListedColormap]] = None,
    title: str = "",
    return_tables: bool = False,
    symmetrical: bool = True,
    default_sep: str = DEFAULT_CPDB_SEP,
    **kwargs
) -> Union[sns.matrix.ClusterGrid, Dict]:
    """Plot cellphonedb results as total counts of interactions.

    Parameters
    ----------
    adata : AnnData
        `AnnData` object with the `.obs` storing the `celltype_key`.
        The `.obs_names` must match the first column of the input `meta.txt` used for `cellphonedb`.
    cell_types : Optional[List[str]], optional
        List of cell types to include in the heatmap. If `None`, all cell types are included.
    pvals : pd.DataFrame
        Dataframe corresponding to `pvalues.txt` or `relevant_interactions.txt` from cellphonedb.
    degs_analysis : bool, optional
        Whether `cellphonedb` was run in `deg_analysis` mode.
    log1p_transform : bool, optional
        Whether to log1p transform the output.
    alpha : float, optional
        P value threshold value for significance.
    linewidths : float, optional
        Width of lines between each cell.
    row_cluster : bool, optional
        Whether to cluster rows.
    col_cluster : bool, optional
        Whether to cluster columns.
    low_col : str, optional
        Low colour in gradient.
    mid_col : str, optional
        Middle colour in gradient.
    high_col : str, optional
        High colour in gradient.
    cmap : Optional[Union[ListedColormap, str]], optional
        Built-in matplotlib colormap names or custom `ListedColormap`
    title : str, optional
        Plot title.
    return_tables : bool, optional
        Whether to return the dataframes storing the interaction network.
    symmetrical : bool, optional
        Whether to return the sum of interactions as symmetrical heatmap.
    default_sep : str, optional
        The default separator used when CellPhoneDB was run.
    **kwargs
        Passed to seaborn.clustermap.


    Returns
    -------
    Union[sns.matrix.ClusterGrid, Dict]
        Either heatmap of cellphonedb interactions or dataframe containing the interaction network.
    """
    all_intr = pvals.copy()
    intr_pairs = all_intr.interacting_pair
    col_start = (
        DEFAULT_V5_COL_START if all_intr.columns[DEFAULT_CLASS_COL] == "classification" else DEFAULT_COL_START
    )  # in v5, there are 12 columns before the values
    all_int = all_intr.iloc[:, col_start : all_intr.shape[1]].T
    all_int.columns = intr_pairs
    if cell_types is None:
        cell_types = sorted(list(set([y for z in [x.split(default_sep) for x in all_intr.columns[col_start:]] for y in z])))
    cell_types_comb = ["|".join(list(x)) for x in list(product(cell_types, cell_types))]
    cell_types_keep = [ct for ct in all_int.index if ct in cell_types_comb]
    empty_celltypes = list(set(cell_types_comb) ^ set(cell_types_keep))
    all_int = all_int.loc[cell_types_keep]
    if len(empty_celltypes) > 0:
        tmp_ = np.zeros((len(empty_celltypes), all_int.shape[1]))
        if not degs_analysis:
            tmp_ += 1
        tmp_ = pd.DataFrame(tmp_, index=empty_celltypes, columns=all_int.columns)
        all_int = pd.concat([all_int, tmp_], axis=0)
    all_count = all_int.melt(ignore_index=False).reset_index()
    if degs_analysis:
        all_count["significant"] = all_count.value == 1
    else:
        all_count["significant"] = all_count.value < alpha
    count1x = all_count[["index", "significant"]].groupby("index").agg({"significant": "sum"})
    tmp = pd.DataFrame([x.split("|") for x in count1x.index])
    count_final = pd.concat([tmp, count1x.reset_index(drop=True)], axis=1)
    count_final.columns = ["SOURCE", "TARGET", "COUNT"]
    if any(count_final.COUNT > 0):
        count_mat = count_final.pivot_table(index="SOURCE", columns="TARGET", values="COUNT")
        count_mat.columns.name, count_mat.index.name = None, None
        count_mat[pd.isnull(count_mat)] = 0
        if symmetrical:
            count_matx = np.triu(count_mat) + np.tril(count_mat.T) + np.tril(count_mat) + np.triu(count_mat.T)
            count_matx[np.diag_indices_from(count_matx)] = np.diag(count_mat)
            count_matx = pd.DataFrame(count_matx)
            count_matx.columns = count_mat.columns
            count_matx.index = count_mat.index
            count_mat = count_matx.copy()
    if log1p_transform:
        count_mat = np.log1p(count_mat)
    if cmap is None:
        colmap = diverging_palette(low=low_col, medium=mid_col, high=high_col)
    else:
        colmap = cmap
    if not return_tables:
        g = sns.clustermap(
            count_mat,
            row_cluster=row_cluster,
            col_cluster=col_cluster,
            linewidths=linewidths,
            tree_kws={"linewidths": 0},
            cmap=colmap,
            **kwargs
        )
        if title != "":
            g.fig.suptitle(title)
        return g
    else:
        if symmetrical:
            all_sum = pd.DataFrame(count_mat.apply(sum, axis=0), columns=["total_interactions"])
        else:
            count_mat = count_mat.T  # so that the table output is the same layout as the plot
            row_sums = pd.DataFrame(count_mat.apply(sum, axis=0), columns=["total_interactions_row"])
            col_sums = pd.DataFrame(count_mat.apply(sum, axis=1), columns=["total_interactions_col"])
            all_sum = pd.concat([row_sums, col_sums], axis=1)
        out = {"count_network": count_mat, "interaction_count": all_sum, "interaction_edges": count_final}
        return out
