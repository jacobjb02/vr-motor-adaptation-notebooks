import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt 


def bootstrap_by_groups(
    data,
    group_cols,
    y_col,
    x_col,
    confidence_level=0.95,
    n_resamples=1000,
    random_state=None,
    method="percentile",
):
    # Drop NaNs in y_col or x_col or any grouping columns
    subset_cols = list(group_cols) + [x_col, y_col]
    df = data.dropna(subset=subset_cols)

    results = []

    # Group by group_cols + x_col to get CI per x level within each group
    grouped = df.groupby(list(group_cols) + [x_col], dropna=False)

    for keys, group in grouped:
        y = group[y_col].to_numpy()

        # Skip empty groups (shouldn't happen after dropna)
        if y.size == 0:
            continue

        # Bootstrap CI for mean
        res = stats.bootstrap(
            (y,),
            np.mean,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            random_state=random_state,
            method=method,
        )

        ci_low, ci_high = res.confidence_interval.low, res.confidence_interval.high
        estimate = y.mean()

        # Normalize keys into list for consistent row construction
        if len(group_cols) == 0:
            key_vals = []
        elif len(group_cols) == 1:
            # keys is (group_val, x_val) when group_cols has 1 element
            key_vals = [keys[0]]
        else:
            key_vals = list(keys[:-1])

        x_val = keys[-1]

        row = dict(zip(group_cols, key_vals))
        row.update(
            {
                x_col: x_val,
                "estimate": estimate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": len(y),
            }
        )
        results.append(row)

    return pd.DataFrame(results)







import seaborn as sns
import seaborn.objects as so

# Global static color mapping for targets
_colors = sns.color_palette("bright", 4)
TARGET_PALETTE = {
    "L60": _colors[2],  # Green
    "L30": _colors[0],  # Blue
    "R30": _colors[1],  # Orange
    "R60": _colors[3],  # Red
}

import seaborn.objects as so
def plot_bootstrap_cis(
    ci_df,
    x_col,
    estimate_col="estimate",
    ci_low_col="ci_low",
    ci_high_col="ci_high",
    hue="target_x_label",
    facet_col=None,
    facet_row="speed_label",
    col_wrap=None,
    height=4,
    aspect=1.6,
    width_scale=1.5,
    x_order=None,           
    hue_order=None,         
    title=None,
    palette=TARGET_PALETTE,
    save_path='../figures/bootstrap_ci.png'
):

    style_dict = sns.axes_style("whitegrid")
    style_dict["axes.grid.axis"] = "y"
    
    p = (
        so.Plot(
            ci_df,
            x=x_col,
            y=estimate_col,
            ymin=ci_low_col,
            ymax=ci_high_col,
            color=hue if hue in ci_df.columns else None,
        )
        .add(so.Range(), so.Dodge(gap=0.2))
        .add(so.Dots(), so.Dodge(gap=0.2))
        .scale(color=palette)
    )

    if x_order is not None:
        p = p.scale(x=so.Nominal(order=x_order))
    if hue is not None and hue_order is not None:
        p = p.scale(color=so.Nominal(order=hue_order))

    if facet_col or facet_row:
        p = p.facet(col=facet_col, row=facet_row, wrap=col_wrap)

    p = p.layout(size=(height * aspect * width_scale, height))
    
    if title:
        p = p.label(title=title)

    p = p.theme(style_dict)

    if save_path:
        p.save(save_path, dpi=600, bbox_inches='tight')

    return p