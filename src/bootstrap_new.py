import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt 


# Global static color mapping for targets
_colors = sns.color_palette(["#FF0000", "#0000FF", "#FF4500", "#05472A"])
TARGET_PALETTE = {
    "L60": _colors[3], # Green
    "L30": _colors[2], # Blue
    "R30":   _colors[1], # Orange
    "R60":   _colors[0]  # Red
}

def run_bootstrap_new(
    data,
    y_col,
    ppid_col,
    target_col,
    water_col,
    trial_col,
    baseline_str = 'baseline',
    washout_str = 'washout_1',
    n_resamples=9999
):

        # subset baseline to final 8 trials
        baseline = data[data['phase'] == baseline_str]
        max_trials = baseline['trial_num'].max()
        baseline_subset = baseline[baseline['trial_num'] >= max_trials - 7]
        # get baseline means
        baseline_means = (baseline_subset.groupby([ppid_col, target_col], observed=True)[y_col].mean().reset_index(name='avg_baseline_dev'))    

        # get washout subset
        washout = data[(data['phase'] == washout_str) & (data[water_col] == 0)]
        
        merged_df = pd.merge(baseline_means, washout[[ppid_col, target_col, trial_col, y_col]], on=[ppid_col, target_col])
        merged_df['diff'] = merged_df[y_col] - merged_df['avg_baseline_dev']

        # List to store results from t loop
        results = []

        # loop through each target
        group_cols = [target_col, trial_col]

        for (target, trial), group in merged_df.groupby(group_cols, observed=True):
            diffs = group['diff'].to_numpy()

            # SciPy bootstrap
            res = stats.bootstrap(
                            (diffs,), 
                            np.mean, 
                            confidence_level=0.99375, 
                            n_resamples=n_resamples, 
                            method='BCa',
                            random_state=1
                )
    
            # return dictionary
            results.append({
                    'target': target,
                    'trial': trial,
                    'mean_change': np.mean(diffs),
                    'ci_low': res.confidence_interval.low,
                    'ci_high': res.confidence_interval.high,
                    'n_ppid': len(diffs),
                    'is_aftereffect': res.confidence_interval.low > 0 or res.confidence_interval.high < 0
            })

        return pd.DataFrame(results).sort_values(['target','trial'])





# ////////



    
    
        # results_df = pd.DataFrame(results)
    
        # # Convert group_name into separate columns
        # if group_cols is not None:
        #     if len(group_cols) == 1:
        #         # Grouping by 1 column returns scalars
        #         results_df[group_cols[0]] = results_df['group_name']
        #     else:
        #         # Grouping by multiple columns returns tuples. Expand them into a DataFrame.
        #         expanded_cols = pd.DataFrame(results_df['group_name'].tolist(), index=results_df.index)
        #         results_df[group_cols] = expanded_cols
            
        #     # Clean up the intermediate column
        #     results_df = results_df.drop(columns=['group_name'])
    
        # return results_df




def run_bootstrap_old(
    data,
    y_col,
    group_cols=None,
    n_resamples=9999
):

    results = []

        # n ppid
    print(data.groupby('target_x_label', observed=True)[y_col].count())
    
    # sd of ppid means
    print(data.groupby('target_x_label', observed=True)[y_col].std())



    if group_cols != None:

        for name, group in data.groupby(group_cols, observed=True):
            
            # print(len(name))
    
            # Convert the y-values to a 1D array for SciPy 
            y_data = group[y_col].to_numpy()
                    
            # SciPy bootstrap
            res = stats.bootstrap(
                        (y_data,), 
                        np.mean, 
                        confidence_level=0.99, 
                        n_resamples=n_resamples, 
                        method='percentile',
                        random_state=1
                    )

            # Store the results
            results.append({
                        'group_name':name,
                        'mean': np.mean(y_data),
                        'ci_low': res.confidence_interval.low,
                        'ci_high': res.confidence_interval.high
                    })

    else:
            # Convert the y-values to a 1D array for SciPy 
            y_data = data[y_col].to_numpy()
                    
                    
            # SciPy bootstrap
            res = stats.bootstrap(
                        (y_data,), 
                        np.mean, 
                        confidence_level=0.99, 
                        n_resamples=n_resamples, 
                        method='percentile',
                        random_state=1
                    )
                
            # Store the results
            results.append({
                    'mean': np.mean(y_data),
                    'ci_low': res.confidence_interval.low,
                    'ci_high': res.confidence_interval.high
                })
    
    results_df = pd.DataFrame(results)

    # Convert group_name into separate columns
    if group_cols is not None:
        if len(group_cols) == 1:
            # Grouping by 1 column returns scalars
            results_df[group_cols[0]] = results_df['group_name']
        else:
            # Grouping by multiple columns returns tuples. Expand them into a DataFrame.
            expanded_cols = pd.DataFrame(results_df['group_name'].tolist(), index=results_df.index)
            results_df[group_cols] = expanded_cols
        
        # Clean up the intermediate column
        results_df = results_df.drop(columns=['group_name'])

    return results_df
    
    


def calculate_distribution_bounds(data, y_col, group_cols, ppid_col):

    agg_cols = group_cols + ppid_col
    ppid_means = data.groupby(agg_cols, observed=True)[y_col].mean().reset_index()
    
    
    results = []

    for name, group in ppid_means.groupby(group_cols, observed=True):
        
        y_data = group[y_col].to_numpy()
            
        # Calculate the 95% of distribution bounds
        mean_val = np.mean(y_data)
        std_val = np.std(y_data, ddof=1)
        sigma_low = mean_val - (1.96 * std_val)
        sigma_high = mean_val + (1.96 * std_val)
            
        results.append({
                'group_name': name,
                'mean': mean_val,
                'sigma_low': sigma_low,
                'sigma_high': sigma_high
            })
            
    results_df = pd.DataFrame(results)

    # Convert group_name into separate columns
    if group_cols is not None:
        if len(group_cols) == 1:
            # Grouping by 1 column returns scalars
            results_df[group_cols[0]] = results_df['group_name']
        else:
            # Grouping by multiple columns returns tuples. Expand them into a DataFrame.
            expanded_cols = pd.DataFrame(results_df['group_name'].tolist(), index=results_df.index)
            results_df[group_cols] = expanded_cols
        
        # Clean up the intermediate column
        results_df = results_df.drop(columns=['group_name'])

    return results_df

























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
    grouped = df.groupby(list(group_cols) + [x_col], dropna=False,  observed=True)

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
    height=6,
    aspect=1.0,
    width_scale=1.0,
    x_order=None,            
    hue_order=None,          
    title=None,
    palette=TARGET_PALETTE,
    save_path='../figures/bootstrap_ci.svg'
):
    # 1. Enforce horizontal grid lines and suppress all vertical structural lines
    style_dict = sns.axes_style("white")
    style_dict.update({
        "axes.grid": True,
        "axes.grid.axis": "y",        # Restricts internal grid to horizontal lines only
        "grid.color": "#e0e0e0",
        "grid.linestyle": "-",
        "axes.edgecolor": "#7f7f7f",  # Medium gray for remaining spines
        
        # Spine Control (Bounding Box)
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,    # Removes the vertical y-axis anchoring line
        "axes.spines.bottom": True,   # Retains the horizontal x-axis baseline
        
        # Tick Mark Control
        "xtick.bottom": False,        # Removes small vertical tick marks on the x-axis
        "ytick.left": False           # Removes small horizontal tick marks on the y-axis
    })
    
    p = (
        so.Plot(
            ci_df,
            x=x_col,
            y=estimate_col,
            ymin=ci_low_col,
            ymax=ci_high_col,
            color=hue if hue in ci_df.columns else None,
        )
        # 2. Thicker lines with partial transparency to match previous trace alphas
        .add(so.Range(linewidth=3.5, alpha=0.6), so.Dodge(empty="drop", gap=0.3))
        # 3. Larger markers with white edges for crisp separation in clusters
        .add(so.Dots(pointsize=9, alpha=0.9), so.Dodge(empty="drop", gap=0.3))
    )

    # 4. Consolidate scale mappings
    if x_order is not None:
        p = p.scale(x=so.Nominal(order=x_order))
        
    if hue is not None:
        if hue_order is not None:
            p = p.scale(color=so.Nominal(order=hue_order, values=palette))
        else:
            p = p.scale(color=so.Nominal(values=palette))

    if facet_col or facet_row:
        p = p.facet(col=facet_col, row=facet_row, wrap=col_wrap)

    p = p.layout(size=(height * aspect * width_scale, height))
    
    if title:
        p = p.label(title=title)

    p = p.theme(style_dict)

    if save_path:
        p.save(save_path, dpi=600, bbox_inches='tight')

    return p