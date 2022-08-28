#!/usr/bin/env python
import pandas as pd
import pytest

from seaborn.matrix import ClusterGrid
from unittest.mock import patch

from ktplotspy.plot import plot_cpdb_heatmap


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("adata", "pvals")
@pytest.mark.parametrize(
    "deg",
    [True, False],
)
def test_plot_cpdb_heatmap(mock_show, adata, pvals, deg):
    g = plot_cpdb_heatmap(
        adata=adata,
        pvals=pvals,
        celltype_key="celltype",
        degs_analysis=deg,
    )
    g


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("adata", "pvals")
def test_plot_cpdb_heatmap_log(mock_show, adata, pvals):
    g = plot_cpdb_heatmap(
        adata=adata,
        pvals=pvals,
        celltype_key="celltype",
        log1p_transform=True,
    )
    g


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("adata", "pvals")
def test_plot_cpdb_heatmap_cmap(mock_show, adata, pvals):
    g = plot_cpdb_heatmap(
        adata=adata,
        pvals=pvals,
        celltype_key="celltype",
        cmap="viridis",
    )
    g


@pytest.mark.usefixtures("adata", "pvals")
def test_plot_cpdb_heatmap_return(adata, pvals):
    dfs = plot_cpdb_heatmap(
        adata=adata,
        pvals=pvals,
        celltype_key="celltype",
        return_tables=True,
    )
    for d in dfs:
        assert isinstance(dfs[d], pd.DataFrame)
        assert dfs[d].shape[0] > 0
