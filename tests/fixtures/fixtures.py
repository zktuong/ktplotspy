#!/usr/bin/env python
import anndata as ad
import pandas as pd
import pytest

from pathlib import Path

DATAPATH = Path(__file__).parent.parent.parent / "data"


@pytest.fixture
def adata():
    return ad.read_h5ad(DATAPATH / "kidneyimmune.h5ad")


@pytest.fixture
def means():
    return pd.read_csv(DATAPATH / "out" / "means.txt", sep="\t")


@pytest.fixture
def pvals():
    return pd.read_csv(DATAPATH / "out" / "pvalues.txt", sep="\t")


@pytest.fixture
def decon():
    return pd.read_csv(DATAPATH / "out" / "deconvoluted.txt", sep="\t")


@pytest.fixture
def means_split():
    return pd.read_csv(DATAPATH / "out_split" / "means.txt", sep="\t")


@pytest.fixture
def pvals_split():
    return pd.read_csv(DATAPATH / "out_split" / "pvalues.txt", sep="\t")


@pytest.fixture
def decon_split():
    return pd.read_csv(DATAPATH / "out_split" / "deconvoluted.txt", sep="\t")
