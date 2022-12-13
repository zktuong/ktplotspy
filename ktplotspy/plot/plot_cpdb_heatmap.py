#!/usr/bin/env python
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.colors import ListedColormap
from typing import Optional, Union, Dict

from ktplotspy.utils.support import diverging_palette


def plot_cpdb_heatmap(
    adata: "AnnData",
    pvals: pd.DataFrame,
    celltype_key: str,
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
    **kwargs
) -> Union[sns.matrix.ClusterGrid, Dict]:
    """Plot cellphonedb results as total counts of interactions.

    Parameters
    ----------
    adata : AnnData
        `AnnData` object with the `.obs` storing the `celltype_key`.
        The `.obs_names` must match the first column of the input `meta.txt` used for `cellphonedb`.
    pvals : pd.DataFrame
        Dataframe corresponding to `pvalues.txt` or `relevant_interactions.txt` from cellphonedb.
    celltype_key : str
        Column name in `adata.obs` storing the celltype annotations.
        Values in this column should match the second column of the input `meta.txt` used for `cellphonedb`.
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
    **kwargs
        Passed to seaborn.clustermap.


    Returns
    -------
    Union[sns.matrix.ClusterGrid, Dict]
        Either heatmap of cellphonedb interactions or dataframe containing the interaction network.
    """
    metadata = adata.obs.copy()
    all_intr = pvals.copy()
    labels = metadata[celltype_key]
    intr_pairs = all_intr.interacting_pair
    all_int = all_intr.iloc[:, 12 : all_intr.shape[1]].T
    all_int.columns = intr_pairs
    all_count = all_int.melt(ignore_index=False).reset_index()
    if degs_analysis:
        all_count = all_count[all_count.value == 1]
    else:
        all_count = all_count[all_count.value < alpha]
    count1x = all_count[["index", "interacting_pair"]].groupby("index").agg({"interacting_pair": "count"})
    tmp = pd.DataFrame([x.split("|") for x in count1x.index])
    count_final = pd.concat([tmp, count1x.reset_index(drop=True)], axis=1)
    count_final.columns = ["SOURCE", "TARGET", "COUNT"]
    if any(count_final.COUNT > 0):
        count_mat = count_final.pivot_table(index="SOURCE", columns="TARGET", values="COUNT")
        count_mat.columns.name, count_mat.index.name = None, None
        count_mat[pd.isnull(count_mat)] = 0
        all_sum = pd.DataFrame(count_mat.apply(sum, axis=0), columns=["total_interactions"]) + pd.DataFrame(
            count_mat.apply(sum, axis=1), columns=["total_interactions"]
        )
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
        out = {"count_network": count_mat, "interaction_count": all_sum}
        return out
