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
        pvals=pvals,
        degs_analysis=deg,
    )
    g


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("adata", "pvals")
def test_plot_cpdb_heatmap_log(mock_show, adata, pvals):
    g = plot_cpdb_heatmap(
        pvals=pvals,
        log1p_transform=True,
    )
    g


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("adata", "pvals")
def test_plot_cpdb_heatmap_sym(mock_show, adata, pvals):
    g = plot_cpdb_heatmap(
        pvals=pvals,
        symmetrical=True,
    )
    g


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("adata", "pvals")
def test_plot_cpdb_heatmap_title(mock_show, adata, pvals):
    g = plot_cpdb_heatmap(
        pvals=pvals,
        title="hey!",
    )
    g


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("adata", "pvals")
def test_plot_cpdb_heatmap_cmap(mock_show, adata, pvals):
    g = plot_cpdb_heatmap(
        pvals=pvals,
        cmap="viridis",
    )
    g


@pytest.mark.usefixtures("adata", "pvals")
def test_plot_cpdb_heatmap_return(adata, pvals):
    dfs = plot_cpdb_heatmap(
        pvals=pvals,
        return_tables=True,
    )
    for d in dfs:
        assert isinstance(dfs[d], pd.DataFrame)
        assert dfs[d].shape[0] > 0


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("adata", "pvals")
def test_plot_cpdb_heatmap_celltypes(mock_show, adata, pvals):
    g = plot_cpdb_heatmap(
        pvals=pvals,
        cell_types=["CD4 T cell", "CD8 T cell", "B cell"],
    )
    g
