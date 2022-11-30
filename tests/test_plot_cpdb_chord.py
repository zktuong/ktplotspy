#!/usr/bin/env python
import pandas as pd
import pytest

from plotnine import ggplot
from unittest.mock import patch

from ktplotspy.plot import plot_cpdb_chord


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("adata", "means", "pvals", "decon")
def test_plot_cpdb_chord(mock_show, adata, means, pvals, decon):
    g = plot_cpdb_chord(
        adata=adata,
        cell_type1="B cell",
        cell_type2=".",
        means=means,
        pvals=pvals,
        deconvoluted=decon,
        celltype_key="celltype",
        genes=["PTPRC", "TNFSF13", "BMPR2"],
    )
    g


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("adata", "means", "pvals", "decon")
def test_plot_cpdb_complex(mock_show, adata, means, pvals, decon):
    g = plot_cpdb_chord(
        adata=adata,
        cell_type1="B cell",
        cell_type2=".",
        means=means,
        pvals=pvals,
        deconvoluted=decon,
        celltype_key="celltype",
        genes=["PTPRC", "TNFSF13", "BMPR2"],
        keep_significant_only=False,
    )
    g


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("adata", "means", "pvals", "decon")
def test_plot_cpdb_chord_color_dict(mock_show, adata, means, pvals, decon):
    g = plot_cpdb_chord(
        adata=adata,
        cell_type1="B cell",
        cell_type2=".",
        means=means,
        pvals=pvals,
        deconvoluted=decon,
        celltype_key="celltype",
        genes=["PTPRC", "TNFSF13", "BMPR2"],
        face_col_dict={
            "B cell": "red",
            "NK cell": "blue",
            "CD4T cell": "red",
            "pDC": "blue",
            "Neutrophil": "red",
            "Mast cell": "blue",
            "NKT cell": "red",
            "CD8T cell": "blue",
        },
        edge_col_dict={"CD22-PTPRC": "red", "TNFRSF13B-TNFSF13B": "blue"},
    )
    g


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("adata", "means", "pvals", "decon")
def test_plot_cpdb_chord_adata_col(mock_show, adata, means, pvals, decon):
    adata.uns["celltype_colors"] = [
        "#1f77b4",
        "#ff7f0e",
        "#279e68",
        "#d62728",
        "#aa40fc",
        "#8c564b",
        "#e377c2",
        "#b5bd61",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
    ]
    g = plot_cpdb_chord(
        adata=adata,
        cell_type1="B cell",
        cell_type2=".",
        means=means,
        pvals=pvals,
        deconvoluted=decon,
        celltype_key="celltype",
        genes=["PTPRC", "TNFSF13", "BMPR2"],
    )
    g


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("adata", "means", "pvals", "decon")
def test_plot_cpdb_chord_adata_col_nocat(mock_show, adata, means, pvals, decon):
    adata.uns["celltype_colors"] = [
        "#1f77b4",
        "#ff7f0e",
        "#279e68",
        "#d62728",
        "#aa40fc",
        "#8c564b",
        "#e377c2",
        "#b5bd61",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
    ]
    adata.obs["celltype"] = [str(c) for c in adata.obs["celltype"]]
    g = plot_cpdb_chord(
        adata=adata,
        cell_type1="B cell",
        cell_type2=".",
        means=means,
        pvals=pvals,
        deconvoluted=decon,
        celltype_key="celltype",
        genes=["PTPRC", "TNFSF13", "BMPR2"],
    )
    g


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("adata", "means", "pvals", "decon")
def test_plot_cpdb_chord_adata_layer1(mock_show, adata, means, pvals, decon):
    adata.layers["test"] = adata.X.copy()
    g = plot_cpdb_chord(
        adata=adata,
        cell_type1="B cell",
        cell_type2=".",
        means=means,
        pvals=pvals,
        deconvoluted=decon,
        celltype_key="celltype",
        genes=["PTPRC", "TNFSF13", "BMPR2"],
        layer="test",
    )
    g


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("adata", "means", "pvals", "decon")
def test_plot_cpdb_chord_adata_layer2(mock_show, adata, means, pvals, decon):
    adata.layers["test"] = adata.X.toarray().copy()
    g = plot_cpdb_chord(
        adata=adata,
        cell_type1="B cell",
        cell_type2=".",
        means=means,
        pvals=pvals,
        deconvoluted=decon,
        celltype_key="celltype",
        genes=["PTPRC", "TNFSF13", "BMPR2"],
        layer="test",
    )
    g
