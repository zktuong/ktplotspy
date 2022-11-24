#!/usr/bin/env python
import numpy as np
import pandas as pd
import re
import scipy.cluster.hierarchy as shc

from collections import Counter
from itertools import count, tee
from matplotlib.colors import ListedColormap
from typing import Dict, List, Optional

from ktplotspy.utils.settings import DEFAULT_SEP


def set_x_stroke(df: pd.DataFrame, isnull: bool, stroke: int):
    """Set stroke value in dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe for plotting.
    isnull : bool
        Whether to check for null.
    stroke : int
        Stroke size value to set.
    """
    for i in df.index:
        if isnull:
            nullstatus = pd.isnull(df.at[i, "x_stroke"])
        else:
            nullstatus = pd.notnull(df.at[i, "x_stroke"])
        if nullstatus:
            df.at[i, "x_stroke"] = stroke


def hclust(data: pd.DataFrame, axis: int = 0) -> List:
    """Perform hierarchical clustering on rows or columns.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe to perform clustering on.
    axis : int, optional
        Index = 0 and columns = 1

    Returns
    -------
    List
        Column name order after hierarchical clustering.
    """
    if data.shape[axis] > 1:
        # TODO (KT): perhaps can pass a kwargs so that linkage can be adjusted?
        if axis == 1:
            data = data.T
        labels = list(data.index)
        data_clusters = shc.linkage(data, method="average", metric="euclidean")
        data_dendrogram = shc.dendrogram(Z=data_clusters, no_plot=True, labels=labels)
        data_order = data_dendrogram["ivl"]
    else:
        if axis == 1:
            labels = data.columns
        else:
            labels = data.index
        # just return the column names as is
        data_order = list(labels)
    return data_order


def filter_interaction_and_celltype(data: pd.DataFrame, genes: List, celltype_pairs: List) -> pd.DataFrame:
    """Filter data to interactions and celltypes.

    Parameters
    ----------
    data : pd.DataFrame
        Input table to perform the filtering.
    genes : List
        List of query genes.
    celltype_pairs : List
        Column names of celltype pairs

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """
    filtered_data = data[data.interacting_pair.isin(genes)][celltype_pairs]
    return filtered_data


def ensure_categorical(meta: pd.DataFrame, key: str) -> pd.DataFrame:
    """Enforce categorical columns.

    Parameters
    ----------
    meta : pd.DataFrame
        Metadata table.
    key : str
        Column key to ensure categorical.

    Returns
    -------
    pd.DataFrame
        Table with formatted column.
    """
    if not is_categorical(meta[key]):
        meta[key] = meta[key].astype("category")
    return meta


def prep_celltype_query(
    meta: pd.DataFrame,
    cell_type1: str,
    cell_type2: str,
    pattern: str,
    split_by: Optional[str] = None,
) -> List:
    """Prepare regex query for celltypes.

    Parameters
    ----------
    meta : pd.DataFrame
        Metadata table.
    cell_type1 : str
        Name of celltype 1.
    cell_type2 : str
        Name of celltype 2.
    pattern : str
        Special character string pattern to substitute.
    split_by : Optional[str], optional
        How to split plotting groups.

    Returns
    -------
    List
        List of celltype querys.
    """
    labels = list(meta._labels.cat.categories)
    if split_by is not None:
        groups = list(meta[split_by].cat.categories)
    ct1 = [l for l in labels if re.search(cell_type1, l)]
    ct2 = [l for l in labels if re.search(cell_type2, l)]
    c_type1 = sub_pattern_loop(ct1, pattern)
    c_type2 = sub_pattern_loop(ct2, pattern)
    celltype = []
    for i in range(0, len(c_type1)):
        cq = create_celltype_query(c_type1[i], c_type2, DEFAULT_SEP)
        if split_by is not None:
            for g in groups:
                cqi = keep_interested_groups(g, cq, DEFAULT_SEP)
                if cqi != "":
                    celltype.append(cqi)
        else:
            celltype.append(cq)
    return celltype


def prep_query_group(means: pd.DataFrame, custom_dict: Optional[Dict[str, List[str]]] = None) -> Dict:
    """Return gene family query groups.

    Parameters
    ----------
    means : pd.DataFrame
        Means table.
    custom_dict : Optional[Dict[str, List[str]]], optional
        If provided, will update the query groups with the custom list of genes.

    Returns
    -------
    Dict
        Dictionary of gene families.
    """
    chemokines = [i for i in means.interacting_pair if re.search(r"^CXC|CCL|CCR|CX3|XCL|XCR", i)]
    th1 = [
        i
        for i in means.interacting_pair
        if re.search(
            r"IL2|IL12|IL18|IL27|IFNG|IL10|TNF$|TNF |LTA|LTB|STAT1|CCR5|CXCR3|IL12RB1|IFNGR1|TBX21|STAT4",
            i,
        )
    ]
    th2 = [i for i in means.interacting_pair if re.search(r"IL4|IL5|IL25|IL10|IL13|AREG|STAT6|GATA3|IL4R", i)]
    th17 = [
        i
        for i in means.interacting_pair
        if re.search(
            r"IL21|IL22|IL24|IL26|IL17A|IL17A|IL17F|IL17RA|IL10|RORC|RORA|STAT3|CCR4|CCR6|IL23RA|TGFB",
            i,
        )
    ]
    treg = [i for i in means.interacting_pair if re.search(r"IL35|IL10|FOXP3|IL2RA|TGFB", i)]
    costimulatory = [
        i
        for i in means.interacting_pair
        if re.search(
            r"CD86|CD80|CD48|LILRB2|LILRB4|TNF|CD2|ICAM|SLAM|LT[AB]|NECTIN2|CD40|CD70|CD27|CD28|CD58|TSLP|PVR|CD44|CD55|CD[1-9]",
            i,
        )
    ]
    coinhibitory = [i for i in means.interacting_pair if re.search(r"SIRP|CD47|ICOS|TIGIT|CTLA4|PDCD1|CD274|LAG3|HAVCR|VSIR", i)]

    query_dict = {
        "chemokines": chemokines,
        "th1": th1,
        "th2": th2,
        "th17": th17,
        "treg": treg,
        "costimulatory": costimulatory,
        "coinhibitory": coinhibitory,
    }
    if custom_dict is not None:
        query_dict.update(custom_dict)
    return query_dict


def prep_table(data: pd.DataFrame) -> pd.DataFrame:
    """Generic function to format the means and pvalues tables.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe. Either pandas DataFrame for means or pvalues.

    Returns
    -------
    pd.DataFrame
        Table ready for further analysis.
    """
    dat = data.copy()
    dat.index = make_unique(dat.interacting_pair)
    dat.columns = [re.sub("\\|", DEFAULT_SEP, col) for col in dat.columns]
    dat.index = [re.sub("_", "-", row) for row in dat.index]
    dat.index = [re.sub("[.]", " ", row) for row in dat.index]

    return dat


def make_unique(seq: pd.Series) -> List:
    """Make unique names.

    Parameters
    ----------
    seq : pd.Series
        Series to convert to unique.

    Returns
    -------
    List
        List of unique names.
    """
    seq = list(seq)
    not_unique = [k for k, v in Counter(seq).items() if v > 1]  # so we have: ['name', 'zip']
    # suffix generator dict - e.g., {'name': <my_gen>, 'zip': <my_gen>}
    suff_gens = dict(zip(not_unique, tee(count(1), len(not_unique))))
    for idx, s in enumerate(seq):
        try:
            suffix = "_" + str(next(suff_gens[s]))
        except KeyError:
            # s was unique
            continue
        else:
            seq[idx] += suffix
    return seq


def sub_pattern(cell_type: str, pattern: str) -> str:
    """Substitute special characters in celltype name.

    Parameters
    ----------
    cell_type : str
        Celltype name to find and replace special pattern.
    pattern : str
        Special pattern to find and replace.

    Returns
    -------
    str
        Formatted celltype name.
    """
    cell_type_tmp = [*cell_type]
    cell_typex = "".join(["\\" + str(x) if re.search(pattern, x) else str(x) for x in cell_type_tmp])
    return cell_typex


def sub_pattern_loop(cell_types: list, pattern: str) -> List:
    """For-loop to substitute special characters in celltype names.

    Parameters
    ----------
    cell_types : list
        List of celltypes to find and replace special pattern.
    pattern : str
        Special pattern to find and replace.

    Returns
    -------
    List
        List of formatted celltype names.
    """
    celltypes = []
    for c in cell_types:
        cx = sub_pattern(cell_type=c, pattern=pattern)
        celltypes.append(cx)
    return celltypes


def is_categorical(series: pd.Series) -> bool:
    """Check if pandas Series is categorical

    Parameters
    ----------
    series : pd.Series
        Series to check.

    Returns
    -------
    bool
        Whether it is categorical or not.
    """
    return series.dtype.name == "category"


def create_celltype_query(ctype1: str, ctypes2: List, sep: str) -> List:
    """Create a regex string term with celltypes.

    Parameters
    ----------
    ctype1 : str
        Name of celltype 1.
    ctypes2 : List
        List of celltype 2 names.
    sep : str
        Character separator to store the split between celltype1 and celltype2s.

    Returns
    -------
    List
        List of regex patterns for celltype1-celltype2s.
    """
    ct = []
    for cx2 in ctypes2:
        ct.append("^" + ctype1 + sep + cx2 + "$")
        ct.append("^" + cx2 + sep + ctype1 + "$")
    ct = "|".join(ct)
    return ct


def keep_interested_groups(grp: str, ct: str, sep: str) -> str:
    """Filter function to only keep interested group in a regex string pattern.

    Parameters
    ----------
    grp : str
        Pattern for interested group.
    ct : str
        Input regex term
    sep : str
        Character separator to store the split between celltypes of the same group.

    Returns
    -------
    str
        Final regex string pattern.
    """
    ctx = ct.split("|")
    ctx = [x for x in ctx if re.search(grp + ".*" + sep + grp, x)]
    return "|".join(ctx)


def hex_to_rgb(hex: str) -> List:
    """Convert hex code to RGB values.

    e.g. "#FFFFFF" -> [255,255,255]

    Parameters
    ----------
    hex : str
        Hex colour code.

    Returns
    -------
    List
        RGB colour code.
    """
    # Pass 16 to the integer function for change of base
    return [int(hex[i : i + 2], 16) for i in range(1, 6, 2)]


def rgb_to_hex(rgb: List) -> str:
    """Convert RGB values to hex code.

    e.g. [255,255,255] -> "#FFFFFF"

    Parameters
    ----------
    rgb : List
        RGB colour code.

    Returns
    -------
    str
        Hex colour code.
    """
    # Components need to be integers for hex to make sense
    if len(rgb) == 4:
        rgb = [int(x) for x in rgb[:3]]
    else:
        rgb = [int(x) for x in rgb]
    return "#" + "".join(["0{0:x}".format(v) if v < 16 else "{0:x}".format(v) for v in rgb])


def colour_dict(gradient: List) -> Dict:
    """Generate dictionary of colours.


    Takes in a list of RGB sub-lists and returns dictionary of colours in RGB and hex form.

    Parameters
    ----------
    gradient : List
        List of RGB colours.

    Returns
    -------
    Dict
        Dictionary of colours.
    """
    return {
        "hex": [rgb_to_hex(RGB) for RGB in gradient],
        "r": [RGB[0] for RGB in gradient],
        "g": [RGB[1] for RGB in gradient],
        "b": [RGB[2] for RGB in gradient],
    }


def linear_gradient(start_hex: str, finish_hex: str = "#FFFFFF", n: int = 10) -> Dict:
    """Return a gradient list of n colours between two hex colours.

    `start_hex` and `finish_hex` should be the full six-digit colour string, including the number sign ("#FFFFFF").

    Parameters
    ----------
    start_hex : str
        Starting hex colour code.
    finish_hex : str, optional
        Finishing hex colour code.
    n : int, optional
        Number of colours between start and finish.

    Returns
    -------
    Dict
        Dictionary of colour gradient.
    """
    # Starting and ending colours in RGB form
    s = hex_to_rgb(start_hex)
    f = hex_to_rgb(finish_hex)
    # Initilize a list of the output colours with the starting colour
    RGB_list = [s]
    # Calculate a colour at each evenly spaced value of t from 1 to n
    for t in range(1, n):
        # Interpolate RGB vector for colour at the current value of t
        curr_vector = [int(s[j] + (float(t) / (n - 1)) * (f[j] - s[j])) for j in range(3)]
        # Add it to our list of output colours
        RGB_list.append(curr_vector)

    return colour_dict(RGB_list)


def diverging_palette(low: str, medium: str, high: str, n: int = 4096) -> ListedColormap:
    """ListerColormap with diverging palette.

    Parameters
    ----------
    low : str
        Colour for low.
    medium : str
        Colour for middle.
    high : str
        Colour for high.
    n : int, optional
        Number of colours between low and high.

    Returns
    -------
    ListedColormap
        Diverging colour palette.
    """
    newcmp = ListedColormap(
        linear_gradient(start_hex=low, finish_hex=medium, n=int(n / 2))["hex"]
        + linear_gradient(start_hex=medium, finish_hex=high, n=int(n / 2))["hex"]
    )

    return newcmp
