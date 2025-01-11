#!/usr/bin/env python
import pandas as pd
import pytest

from plotnine import ggplot
from unittest.mock import patch

from ktplotspy.plot import plot_cpdb_chord


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("adata", "means", "pvals", "decon")
def test_plot_cpdb_chord(mock_show, adata, means, pvals, decon):
    plot_cpdb_chord(
        adata=adata,
        cell_type1="B cell",
        cell_type2=".",
        means=means,
        pvals=pvals,
        deconvoluted=decon,
        celltype_key="celltype",
        interaction=["PTPRC", "TNFSF13", "BMPR2"],
    )


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("adata", "means", "pvals", "decon")
def test_plot_cpdb_chord_self(mock_show, adata, means, pvals, decon):
    plot_cpdb_chord(
        adata=adata,
        cell_type1="B cell",
        cell_type2=".",
        means=means,
        pvals=pvals,
        deconvoluted=decon,
        celltype_key="celltype",
        remove_self=False,
        interaction=["PTPRC", "TNFSF13", "BMPR2"],
    )


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("adata", "means", "pvals", "decon")
def test_plot_cpdb_complex(mock_show, adata, means, pvals, decon):
    plot_cpdb_chord(
        adata=adata,
        cell_type1="B cell",
        cell_type2=".",
        means=means,
        pvals=pvals,
        deconvoluted=decon,
        celltype_key="celltype",
        interaction=["PTPRC", "TNFSF13", "BMPR2"],
        keep_significant_only=False,
    )


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("adata", "means", "pvals", "decon")
def test_plot_cpdb_chord_color_dict(mock_show, adata, means, pvals, decon):
    plot_cpdb_chord(
        adata=adata,
        cell_type1="B cell",
        cell_type2=".",
        means=means,
        pvals=pvals,
        deconvoluted=decon,
        celltype_key="celltype",
        interaction=["PTPRC", "TNFSF13", "BMPR2"],
        sector_colors={
            "B cell": "red",
            "NK cell": "blue",
            "CD4T cell": "red",
            "pDC": "blue",
            "Neutrophil": "red",
            "Mast cell": "blue",
            "NKT cell": "red",
            "CD8T cell": "blue",
        },
        link_colors={"CD22-PTPRC": "red", "TNFRSF13B-TNFSF13B": "blue"},
    )


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
    plot_cpdb_chord(
        adata=adata,
        cell_type1="B cell",
        cell_type2=".",
        means=means,
        pvals=pvals,
        deconvoluted=decon,
        celltype_key="celltype",
        interaction=["PTPRC", "TNFSF13", "BMPR2"],
    )


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
    plot_cpdb_chord(
        adata=adata,
        cell_type1="B cell",
        cell_type2=".",
        means=means,
        pvals=pvals,
        deconvoluted=decon,
        celltype_key="celltype",
        interaction=["PTPRC", "TNFSF13", "BMPR2"],
    )


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("adata", "means", "pvals", "decon")
def test_plot_cpdb_chord_adata_layer1(mock_show, adata, means, pvals, decon):
    adata.layers["test"] = adata.X.copy()
    plot_cpdb_chord(
        adata=adata,
        cell_type1="B cell",
        cell_type2=".",
        means=means,
        pvals=pvals,
        deconvoluted=decon,
        celltype_key="celltype",
        interaction=["PTPRC", "TNFSF13", "BMPR2"],
        layer="test",
    )


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("adata", "means", "pvals", "decon")
def test_plot_cpdb_chord_adata_layer2(mock_show, adata, means, pvals, decon):
    adata.layers["test"] = adata.X.toarray().copy()
    plot_cpdb_chord(
        adata=adata,
        cell_type1="B cell",
        cell_type2=".",
        means=means,
        pvals=pvals,
        deconvoluted=decon,
        celltype_key="celltype",
        interaction=["PTPRC", "TNFSF13", "BMPR2"],
        layer="test",
    )


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("adata", "means", "pvals", "decon")
def test_plot_cpdb_chord_specific(mock_show, adata, means, pvals, decon):
    plot_cpdb_chord(
        adata=adata,
        interaction="CLEC2D-KLRB1",
        keep_celltypes=["NKT cell", "Mast cell", "NK cell"],
        celltype_key="celltype",
        means=means,
        pvals=pvals,
        deconvoluted=decon,
        link_kwargs={"direction": 1, "allow_twist": False, "r1": 95, "r2": 90},
        sector_text_kwargs={"color": "black", "size": 12, "r": 105, "adjust_rotation": True},
        legend_kwargs={"loc": "center", "bbox_to_anchor": (1, 1), "fontsize": 8},
    )


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("adata", "means", "pvals", "decon")
def test_plot_cpdb_chord_equal_sector(mock_show, adata, means, pvals, decon):
    plot_cpdb_chord(
        adata=adata,
        interaction="CLEC2D-KLRB1",
        keep_celltypes=["NKT cell", "Mast cell", "NK cell"],
        celltype_key="celltype",
        means=means,
        pvals=pvals,
        deconvoluted=decon,
        link_kwargs={"direction": 1, "allow_twist": False, "r1": 95, "r2": 90},
        sector_text_kwargs={"color": "black", "size": 12, "r": 105, "adjust_rotation": True},
        legend_kwargs={"loc": "center", "bbox_to_anchor": (1, 1), "fontsize": 8},
        equal_sector_size=True,
    )


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("adata", "means", "pvals", "decon")
def test_plot_cpdb_chord_two_ct(mock_show, adata, means, pvals, decon):
    plot_cpdb_chord(
        adata=adata,
        interaction="CLEC2D-KLRB1",
        keep_celltypes=["Mast cell", "NK cell"],
        celltype_key="celltype",
        means=means,
        pvals=pvals,
        deconvoluted=decon,
        link_kwargs={"direction": 1, "allow_twist": False, "r1": 95, "r2": 90},
        sector_text_kwargs={"color": "black", "size": 12, "r": 105, "adjust_rotation": True},
        legend_kwargs={"loc": "center", "bbox_to_anchor": (1, 1), "fontsize": 8},
        equal_sector_size=True,
    )


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("adata", "means", "pvals", "decon")
def test_plot_cpdb_chord_one_ct(mock_show, adata, means, pvals, decon):
    plot_cpdb_chord(
        adata=adata,
        interaction="CLEC2D-KLRB1",
        keep_celltypes=["NK cell"],
        celltype_key="celltype",
        means=means,
        pvals=pvals,
        deconvoluted=decon,
        link_kwargs={"direction": 1, "allow_twist": False, "r1": 95, "r2": 90},
        sector_text_kwargs={"color": "black", "size": 12, "r": 105, "adjust_rotation": True},
        legend_kwargs={"loc": "center", "bbox_to_anchor": (1, 1), "fontsize": 8},
        equal_sector_size=True,
    )


@patch("matplotlib.pyplot.show")
@pytest.mark.usefixtures("adata", "means", "pvals", "decon")
def test_plot_cpdb_chord_fixed_col(mock_show, adata, means, pvals, decon):
    plot_cpdb_chord(
        adata=adata,
        interaction="CLEC2D-KLRB1",
        keep_celltypes=["NK cell"],
        celltype_key="celltype",
        means=means,
        pvals=pvals,
        deconvoluted=decon,
        link_kwargs={"direction": 1, "allow_twist": False, "r1": 95, "r2": 90},
        sector_text_kwargs={"color": "black", "size": 12, "r": 105, "adjust_rotation": True},
        legend_kwargs={"loc": "center", "bbox_to_anchor": (1, 1), "fontsize": 8},
        link_colors="red",
    )
