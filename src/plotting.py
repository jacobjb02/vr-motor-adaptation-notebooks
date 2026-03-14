"""
Plotting functions.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm, Normalize


# trial schedule plot
def plot_trial_schedule(
    data, 
    y_col,
    context='notebook',
    font_scale=2,
    save_path='../figures/trial_schedule.pdf',
    dpi=300
):

    # edit font size:
    sns.set_context(context, font_scale=5)  
    sns.set_theme()
    sns.set_style("white")

        
    # build grid
    g = sns.relplot(
        data=data, 
        kind='line', linewidth = 3,
        x='trial_num', y=y_col,
        height=5, 
        aspect=3
    )

    # set axis labels
    g.set_axis_labels('Trial Number', 'Water State')
    
    # save figure
    if save_path:
        g.fig.savefig(save_path, dpi=dpi)  
    
    # display
    plt.show()
    return g


def plot_baseline(
    data,
    ppid_col,
    cond_col,
    x_col='trial_num_target',
    y_col='ball_dist_to_center_cm',
    x_lim=(6,13),
    y_lim=(None,50.0),
    show_zero_line=True,
    context='notebook',
    font_scale=3,
    save_path='../figures/baseline_trials_by_target.pdf',
    dpi=300
):

    # filter for baseline
    #baseline_df = data[data['phase'] == 'baseline']

    sns.set_context(context, font_scale) 
    sns.set_theme()
    sns.set_style("white")

    # set grid and make facets by target
    g = sns.FacetGrid(data, 
                      col='target_x_label',
                      sharex=False, sharey=True)

    # clean facet titles 
    
    # x and y lims
    g.set(ylim=y_lim)
    g.set(xlim=x_lim)
    
    # individual data
    g.map_dataframe(sns.lineplot,
                    data=data, units=ppid_col, estimator=None,
                    x=x_col, y=y_col,
                    linewidth = 2, hue='target_x_label', alpha=0.10
                   )
    
    # mean data
    g.map_dataframe(sns.lineplot,
                    data=data,
                    x=x_col, y=y_col,
                    estimator='mean', errorbar='se', err_kws={'alpha':0.35, 'linewidth':0},
                    linewidth=3, hue='target_x_label', style=cond_col
                   )
                    
    # set axis labels
    #g.set_axis_labels('Trial Number (per target)', 'Min Distance (cm)')

        
    if show_zero_line == True:
        for ax in g.axes.flat:
            ax.axhline(y=0.0, color = 'black', linestyle='--', alpha = 0.3)
            ax.set_xticks(range(int(x_lim[0] + 1), int(x_lim[1] + 1), 4))

    # make trial numbers integers
    for ax in g.axes.flat:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    g.fig.set_size_inches(14, 7)   # width, height in inches
    
    # save figure
    if save_path:
        g.fig.savefig(save_path, dpi=dpi) 
        
    # display
    plt.show()

    return g

def plot_all_trials(
    data,
    cond_col,
    ppid_col,
    row_col,
    col_col,
    target_col,
    transition_col=None,
    show_sd_line=False,
    show_zero_line=False,
    y_col='baseline_corrected_dist',
    y_lim=(None, 110.0),
    x_col='trial_num_target',
    estimator='mean',
    context='notebook',
    marker_size=4,
    font_scale=3,
    save_path='../figures/exposure_trials_by_target_x_set.pdf',
    dpi=300
):
    data = data.copy()
    data[x_col] = data[x_col].astype(float)

    # --- Global Statistics for SD Lines ---
    if show_sd_line:
        global_mean = data[y_col].mean()
        global_sd = data[y_col].std()

    # --- Identify Transition Trials and Active Spans ---
    transition_trials = []
    inactive_spans = []
    
    if transition_col and transition_col in data.columns:
        schedule_df = data[[x_col, transition_col]].drop_duplicates().sort_values(x_col).reset_index(drop=True)
        
        # 1. Get explicit transition lines
        shifted_state = schedule_df[transition_col].shift(1)
        is_transition = (schedule_df[transition_col] != shifted_state) & shifted_state.notna()
        transition_trials = schedule_df.loc[is_transition, x_col].unique()

        # 2. Get spans where transition_col == 1 (Perturbation ON)
        in_inactive_block = False
        start_x = None
        
        for idx, row in schedule_df.iterrows():
            val = row[transition_col]
            x_val = row[x_col]
            
            if val == 0 and not in_inactive_block:
                start_x = x_val
                in_inactive_block = True
            elif val == 1 and in_inactive_block:
                inactive_spans.append((start_x, x_val))
                in_inactive_block = False
                
        # Close any trailing active block that reaches the end of the dataframe
        if in_inactive_block:
            inactive_spans.append((start_x, schedule_df[x_col].max()))

    # --- PAD MISSING X-VALUES WITH NANS TO BREAK LINES ---
    grouping_cols = [c for c in [ppid_col, cond_col, target_col, row_col, col_col] if c and c in data.columns]
    
    unique_groups = data[grouping_cols].drop_duplicates().assign(_key=1)
    unique_x = pd.DataFrame({x_col: data[x_col].dropna().unique(), '_key': 1})
    full_grid = pd.merge(unique_groups, unique_x, on='_key').drop('_key', axis=1)

    # Merge to insert rows with NaN in y_col for missing x-values
    data = pd.merge(full_grid, data, on=grouping_cols + [x_col], how='left')

    # --- CREATE PHASE/SET ORDER IDENTIFIER ---
    if transition_col and transition_col in data.columns and cond_col in data.columns:
        data['_phase_id'] = data[transition_col].astype(str) + '_cond_' + data[cond_col].astype(str)
    elif transition_col and transition_col in data.columns:
        data['_phase_id'] = data[transition_col].astype(str)
    else:
        data['_phase_id'] = '0'

    # --- Categorical Assignment and Plotting Setup ---
    labels = sorted(data[target_col].dropna().unique(), key=str)
    data[target_col] = pd.Categorical(data[target_col], categories=labels, ordered=True)
    palette_map = dict(zip(labels, sns.color_palette("bright", len(labels))))

    sns.set_context(context, font_scale=font_scale)
    sns.set_theme(style="darkgrid")

    g = sns.FacetGrid(
        data,
        row=row_col,
        col=col_col,
        sharex=True,
        sharey=True,
        margin_titles=True
    )
    g.set(ylim=y_lim)

    # 1. Plot individual participant traces
    g.map_dataframe(
        sns.lineplot,
        x=x_col, y=y_col,
        units=ppid_col, estimator=None,
        hue=target_col,
        palette=palette_map,
        alpha=0.05, legend=False
    )

    # 2. Plot Mean + SE separately per phase to prevent cross-phase connections
    # Handle both faceted and non-faceted cases
    axes_to_plot = []
    
    if row_col or col_col:
        # Faceted case: iterate through axes_dict
        for facet_key, ax in g.axes_dict.items():
            if not ax.get_visible():
                continue
            
            # Handle both (row,col) and single row cases
            if isinstance(facet_key, tuple):
                row_val, col_val = facet_key
            else:
                row_val = facet_key
                col_val = None
            
            # Get data for this facet
            facet_data = data.copy()
            if row_col and row_col in data.columns:
                facet_data = facet_data[facet_data[row_col] == row_val]
            if col_col and col_col in data.columns:
                facet_data = facet_data[facet_data[col_col] == col_val]
            
            axes_to_plot.append((ax, facet_data))
    else:
        # Non-faceted case: single axis
        axes_to_plot.append((g.ax, data))
    
    # Plot means on each axis
    for ax, facet_data in axes_to_plot:
        # Calculate means per phase/target for this facet
        grouped = facet_data.dropna(subset=[y_col]).groupby(
            [x_col, target_col, '_phase_id']
        ).agg({y_col: ['mean', 'sem', 'count']}).reset_index()
        
        grouped.columns = [x_col, target_col, '_phase_id', 'mean', 'sem', 'count']
        grouped['sem'] = grouped['sem'].fillna(0)
        
        # Plot each target color separately
        for target_label in labels:
            target_data = grouped[grouped[target_col] == target_label].sort_values([x_col])
            color = palette_map[target_label]
            
            if len(target_data) > 0:
                # Plot line per phase (this naturally breaks at phase boundaries)
                for phase_id in target_data['_phase_id'].unique():
                    phase_subset = target_data[target_data['_phase_id'] == phase_id].sort_values(x_col)
                    
                    if len(phase_subset) > 0:
                        ax.plot(
                            phase_subset[x_col],
                            phase_subset['mean'],
                            marker='o',
                            markersize=marker_size,
                            linewidth=3.0,
                            color=color,
                            alpha=0.80
                        )
                        
                        # Add error bars
                        ax.fill_between(
                            phase_subset[x_col],
                            phase_subset['mean'] - phase_subset['sem'],
                            phase_subset['mean'] + phase_subset['sem'],
                            alpha=0.25,
                            color=color,
                            linewidth=0
                        )

    g.fig.set_size_inches(24, 16)

    # --- AXIS TRACKING FOR LABEL RESTORATION ---
    visible_bottom_axes = {}
    visible_left_axes = {}
    nrows, ncols = g.axes.shape

    # --- TICKER, REFERENCE LINE, SHADING & EMPTY FACETS ---
    for i in range(nrows):
        for j in range(ncols):
            ax = g.axes[i, j]

            if not ax.lines and not ax.collections:
                ax.set_visible(False)
                continue

            visible_bottom_axes[j] = ax
            if i not in visible_left_axes:
                visible_left_axes[i] = ax

            # Reference Lines Logic
            if show_zero_line:
                ax.axhline(y=0.0, color='black', linestyle='--', alpha=0.3)
                
            if show_sd_line:
                # Plot +/- 1 SD lines (alpha=0.4)
                ax.axhline(y=global_mean + global_sd, color='red', linestyle=':', alpha=0.4, lw = 3, zorder=1)
                ax.axhline(y=global_mean - global_sd, color='red', linestyle=':', alpha=0.4, lw = 3, zorder=1)
                
                # Plot +/- 2 SD lines (alpha=0.2 for visual hierarchy)
                ax.axhline(y=global_mean + (2 * global_sd), color='red', linestyle=':', alpha=0.2, lw = 1.5, zorder=1)
                ax.axhline(y=global_mean - (2 * global_sd), color='red', linestyle=':', alpha=0.2, lw = 1.5, zorder=1)
                
            for span_start, span_end in inactive_spans:
                ax.axvspan(span_start, span_end, color='gray', alpha=0.15, zorder=0, lw=0)
                
            for t_x in transition_trials:
                ax.axvline(x=t_x, color='gray', linestyle='--', alpha=0.7, zorder=0)
            
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
            ax.xaxis.get_major_formatter().set_scientific(False)

    # --- RESTORE LABELS ON NEW BOUNDARY AXES ---
    for ax in visible_bottom_axes.values():
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_xlabel(x_col)
        ax.xaxis.label.set_visible(True) 
        
    for ax in visible_left_axes.values():
        ax.yaxis.set_tick_params(labelleft=True)
        ax.set_ylabel(y_col)
        ax.yaxis.label.set_visible(True) 

    # --- ROBUST LEGEND EXTRACTION ---
    handles, legend_labels = [], []
    for ax in g.axes.flat:
        if ax.get_visible():
            h, l = ax.get_legend_handles_labels()
            if h:
                handles, legend_labels = h, l
                break
    
    if handles:
        g.fig.legend(handles, legend_labels,
                     title=cond_col.replace("_"," ").title(),
                     loc="center left", 
                     bbox_to_anchor=(0.88, 0.5), 
                     frameon=True)

    g.fig.subplots_adjust(right=0.82, bottom=0.2, left=0.1, wspace=0.1)

    if save_path:
        g.fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()
    return g

# early late exposure
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_early_late_exposure(
    data,
    cond_col, # colour
    ppid_col,
    y_col,
    x_col,
    line_col,
    facet_row,
    facet_col,
    ylim=None,              # NEW: Argument for y-axis limits (e.g., [-30, 30])
    show_zero_line=False,
    context='notebook',
    font_scale=1.2,          
    facet_height=4,          
    facet_aspect=1.2,        
    save_path='../figures/early_late_exposure_by_target_x_set.pdf',
    dpi=300
):
    sns.set_theme(context=context, font_scale=font_scale, style="white")
    palette = sns.color_palette('bright')
    
    data = data.copy()
    data['unit_id'] = data[ppid_col].astype(str) + '_' + data[line_col].astype(str)
    dynamic_x_order = list(data[x_col].dropna().unique())
    data[x_col] = pd.Categorical(data[x_col], categories=dynamic_x_order, ordered=True)

    g = sns.FacetGrid(
        data,
        col=facet_col,
        row=facet_row,
        height=facet_height, 
        aspect=facet_aspect,
        sharey=True,
        sharex=True,
        margin_titles=True
    )

    # Individual data points
    g.map_dataframe(
        sns.stripplot,
        x=x_col, y=y_col,
        hue=cond_col,
        order=dynamic_x_order,
        jitter=0.05, alpha=0.40, size=4,
        palette=palette,
        legend=False
    )

    # Individual participant lines
    g.map_dataframe(
        sns.lineplot,
        x=x_col, y=y_col,
        units='unit_id',
        estimator=None,
        color='0.5', alpha=0.15, linewidth=0.8,
        legend=False
    )
        
    # 1. Extract the explicit global order of hue levels
    global_hue_order = list(data[cond_col].unique())
    n_hues = len(global_hue_order)
    
    # 2. Define markers and linestyles as lists scaled to the number of hues
    marker_list = ["o", "s", "D", "^", "v", "<", ">"]
    style_list = ["-", "--", "-.", ":", "-", "--", "-."]
    
    dynamic_markers = marker_list[:n_hues]
    dynamic_linestyles = style_list[:n_hues]
    
    # 3. Update the pointplot call
    g.map_dataframe(
        sns.pointplot,
        x=x_col, y=y_col,
        hue=cond_col,                    
        order=dynamic_x_order, 
        hue_order=global_hue_order,    # CRITICAL: Synchronizes mapping across all facets
        palette=palette,
        linestyles=dynamic_linestyles, # Passed as lists
        markers=dynamic_markers,       # Passed as lists
        scale=0.8,
        estimator=np.mean,
        errorbar='se',
        capsize=.1
    )

    # Apply Y-limits and Zero Line
    if ylim is not None:
        g.set(ylim=ylim)

    if show_zero_line:
        for ax in g.axes.flat:
            ax.axhline(0.0, color='black', linestyle='--', alpha=0.3)
    
    g.add_legend(title=cond_col)

    if save_path:
        g.fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()
    return g
    

def plot_continuous_exposure(
    data,
    cond_col,
    ppid_col,
    row_col,
    x_col,
    target_col,
    phase_col='phase',          # NEW: Used for vertical demarcations instead of columns
    show_zero_line=False,
    y_col='baseline_corrected_dist',
    y_lim=(None, 110.0),
    estimator='mean',
    context='notebook',
    marker_size=4,
    font_scale=3,
    save_path=None,
    dpi=300
):
    data = data.copy()
    data[x_col] = data[x_col].astype(float)

    labels = sorted(data[target_col].unique(), key=str)
    data[target_col] = pd.Categorical(data[target_col], categories=labels, ordered=True)
    palette_map = dict(zip(labels, sns.color_palette("bright", len(labels))))

    sns.set_context(context, font_scale=font_scale)
    sns.set_theme(style="white")

    # Removed col_col entirely to allow maximum width per row
    g = sns.FacetGrid(
        data,
        row=row_col,
        sharex=True,
        sharey=True,
        margin_titles=True,
        aspect=2.5,  # Force a wide aspect ratio for the single column
        height=5     # Adjust height per row
    )
    g.set(ylim=y_lim)

    # 1. Individual participant traces
    g.map_dataframe(
        sns.lineplot,
        x=x_col, y=y_col,
        units=ppid_col, estimator=None,
        hue=target_col, palette=palette_map,
        alpha=0.03, legend=False
    )

    # 2. Mean + SE
    g.map_dataframe(
        sns.lineplot,
        x=x_col, y=y_col,
        estimator=estimator, linewidth=1.5,
        errorbar='se', err_kws={"alpha":0.25, "linewidth":0},
        hue=target_col, style=cond_col,
        markers=True, markersize=marker_size,
        palette=palette_map, alpha=1, dashes=True
    )

    # --- PHASE DEMARCATION (The Alternative to Faceting) ---
    # Calculate the trial transitions to draw vertical lines
    phase_transitions = data.groupby(phase_col)[x_col].min().sort_values()
    
    for ax in g.axes.flat:
        if show_zero_line:
            ax.axhline(y=0.0, color='black', linestyle='--', alpha=0.3)
            
        # Draw vertical lines for phase changes
        for transition_trial in phase_transitions:
            if transition_trial > data[x_col].min(): # Skip line at the very beginning
                ax.axvline(x=transition_trial - 0.5, color='gray', linestyle=':', alpha=0.8)

        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, integer=True))
        ax.xaxis.get_major_formatter().set_scientific(False)

    # --- LEGEND & SPACING FIX ---
    handles, legend_labels = g.axes.flat[0].get_legend_handles_labels()
    
    g.fig.legend(handles, legend_labels,
                 title=cond_col.replace("_"," ").title(),
                 loc="center left", 
                 bbox_to_anchor=(0.90, 0.5), 
                 frameon=True)

    g.fig.subplots_adjust(right=0.85, wspace=0.1) 

    if save_path:
        g.fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()
    return g



    

def plot_exposure_trials_2m(
    data,
    y_col='baseline_corrected_dist',
    y_lim=(None,110.0),
    estimator='mean',
    context='notebook',
    font_scale=3,
    save_path='../figures/exposure_trials_by_target_x_set_2m.pdf',
    dpi=300
):

    sns.set_context(context, font_scale) 
    sns.set_theme()
    sns.set_style("white")

    # set facets by target
    g = sns.FacetGrid(data, 
                      col='target_x_label',
                      col_order=["neg0.6", "neg0.3", "p0.3", "p0.6"],
                      row='set_order', 
                      sharex=True, 
                      sharey=True)

    g.set_titles("")
    
    # set Y lim to 110.0 cm
    g.set(ylim=y_lim)
    
    # individual data
    g.map_dataframe(sns.lineplot,
                    x='phase_trial_target', y=y_col,
                    estimator=None, units='ppid',
                    hue = 'target_x_label', palette='bright',
                    alpha=0.25)
    
    # mean line and se bands
    g.map_dataframe(sns.lineplot,
                    x='phase_trial_target', y=y_col,
                    estimator=estimator, errorbar='se', err_kws={'alpha':0.25, 'linewidth':0},
                    hue = 'target_x_label', palette='bright', alpha=1, dashes=True)

    g.fig.set_size_inches(14, 10.5)   # width, height in inches
    
    
    # Save
    if save_path:
        g.fig.savefig(save_path, dpi=dpi) 

    # display
    plt.show()

    return g





# generalization plot
def plot_generalization(
    data,
    cond_col,
    y_col='mean_dist',
    show_zero_line=True,
    context='notebook',
    font_scale=3,
    save_path='../figures/generalization_by_set.pdf',
    dpi=300
    
    
):

    sns.set_context(context, font_scale) 
    sns.set_theme()
    sns.set_style("white")


        
    # set facets by target
    g = sns.FacetGrid(data, 
                      row=cond_col,
                      col='target_x_label', 
                      col_order=["neg0.6", "neg0.3", "p0.3", "p0.6"],
                      sharex=True, 
                      sharey=True)    
    # no titles
    #g.set_titles("")

    
    # individual data
    g.map_dataframe(sns.stripplot,
                    x='phase', y=y_col,
                    order=['training_1','training_2'],
                    hue='set_order',
                    jitter=0.15, alpha=0.5, size=5, linewidth=0,
                    legend=False
    )
    
    
    # mean line and se bands
    g.map_dataframe(sns.pointplot,
                    x='phase', y=y_col,
                    order=['training_1','training_2'],
                    hue=cond_col, palette='bright', alpha = 0.7,
                    estimator=np.mean, errorbar='se', capsize=.15,
                    legend=False
    )


    g.fig.set_size_inches(14, 7)   # width, height in inches

    for ax in g.axes.flat:
        ax.axhline(y=0.0, color='black', linestyle='--', alpha=0.3)

        
    # Save
    if save_path:
        g.fig.savefig(save_path, dpi=dpi) 

    # display
    plt.show()

    return g




def plot_boxplot_target_error(
    data,
    measure,
    facet_by_target = True,
    context='notebook',
    font_scale=2,
    save_path='../figures/boxplot_target_error.pdf',
    dpi=300
    
    
):

    sns.set_context(context, font_scale) 
    sns.set_theme()
    sns.set_style("white")

    if facet_by_target == True:
        # set facets by target
        g = sns.FacetGrid(data, col='target_x_label', sharex=True, sharey=True)  
    else:
        g = sns.FacetGrid(data, sharex=True, sharey=True)  
    # individual data
    g.map_dataframe(sns.boxplot,
                    x=measure
    )
        
    # Save
    if save_path:
        g.fig.savefig(save_path, dpi=dpi) 

    # display
    plt.show()

    return g




# solution space plot
def plot_solution_space(
    data,
    x_col,
    y_col,
    error,
    context='notebook',
    font_scale=2,
    save_path='../figures/solution_space_by_target.pdf',
    dpi=300
    
    
):

    sns.set_context(context, font_scale) 
    sns.set_theme()
    sns.set_style("white")

        
    # set facets by target
    g = sns.FacetGrid(data, col='target_x_label', sharex=True, sharey=True)    
    
    # individual data
    g.map_dataframe(sns.heatmap,
                    x=x_col, y=y_col,
                    hue=error
    )

        
    # Save
    if save_path:
        g.fig.savefig(save_path, dpi=dpi) 

    # display
    plt.show()

    return g




# generalization plot
def plot_baseline_washout(
    data,
    ppid_col,
    speed_col,
    x_col='trial_num', 
    y_col='launch_deviation',
    y_lim=(-10,50),
    start_trial=6,
    block_len=4,
    divide_phases=False,
    show_hits=False,
    show_speeds=False,
    marker_size=4,
    context='notebook',
    font_scale=3,
    save_path='../figures/baseline_washout.pdf',
    dpi=300
    
    
):


    # set facets by target
    g = sns.FacetGrid(data, 
                      col='target_x_label', 
                      col_order=["neg0.6", "neg0.3", "p0.3", "p0.6"], 
                      row='set_order', 
                      sharex=True, 
                      sharey=True)
    # set Y lim
    g.set(ylim=y_lim)

    # individual data
    g.map_dataframe(sns.lineplot,
                    x=x_col, y=y_col,
                    estimator=None, units=ppid_col,
                    hue = 'target_x_label', palette='bright',
                    alpha=0.1)

    
    if show_hits == True:
                  
        hit_palette = {'False': 'red', 'True': 'green'}
        hit_markers = {'False': 'o',   'True': 's'}
        hit_dashes  = {'False': '',    'True': (2, 2)}

        
        g.map_dataframe(
                        sns.lineplot,
                        x=x_col, y=y_col,
                        estimator='mean', errorbar='se',
                        hue='target_hit',
                        style='target_hit',
                        palette=hit_palette,
                        markers=hit_markers,
                        dashes=hit_dashes,
                        markersize=marker_size,
                        alpha=1
                        )
        
        g.add_legend(title="legend")

    elif show_speeds == True:

        g.map_dataframe(
                        sns.lineplot,
                        x=x_col, y=y_col,
                        estimator='mean', errorbar='se',
                        hue='target_x_label',
                        style=speed_col,
                        palette='bright',
                        markers=True,
                        markersize=marker_size,
                        alpha=1
                        )
        
        g.add_legend(title="legend")


    else:   
        # mean line and se bands
        g.map_dataframe(sns.lineplot,
                        x=x_col, y=y_col, marker='o', markersize=marker_size,
                        estimator='mean', errorbar='se', err_kws={'alpha':0.25, 'linewidth':0},
                        hue = 'target_x_label', palette='bright', alpha=1, dashes=True)


    # final trial per target 
    end_trial = data[x_col].max()      # 36 total trials per target
    print(end_trial)
    
    block_bounds = (data.groupby(['phase','block'])[x_col].agg(['min','max']).reset_index())

    washout_bounds = block_bounds[block_bounds['phase'].str.contains("washout")]

    washout_bounds = washout_bounds.drop_duplicates(subset=['block'])
        
    for ax in g.axes.flat:
        for _, row in washout_bounds.iterrows():
    
            if row['block'] % 2 == 0:  # odd blocks => current OFF
                ax.axvspan(
                    row['min'] - 0.5,
                    row['max'] + 0.5,
                    color='black',
                    alpha=0.12
                )
    


        
    # add horizontal line at error of 0
    for ax in g.axes.flat:
        ax.axhline(y=0.0, color = 'black', linestyle='--', alpha = 0.3)
        ax.set_xticks(range(1, int(data[x_col].max()) + 1, 4))
    

    if divide_phases == True:
        # add vertical line at x = 6 (seperates baseline from washout: washout starts on x = 7)
        for ax in g.axes.flat:
            ax.axvline(x=start_trial - 0.5, color='red', linestyle='--', linewidth=1, alpha=0.8)
        
    g.fig.set_size_inches(14, 7)   # width, height in inches

    # Save
    if save_path:
        g.fig.savefig(save_path, dpi=dpi) 

    # display
    plt.show()

    return g






def plot_density_targets(
    data,
    x_col,
    x_lim,
    cond_col,
    r_col='set_order',
    c_col='target_x_label',
    context='notebook',
    font_scale=3,
    save_path='../figures/baseline_trials_by_target.pdf',
    dpi=300
):
    sns.set_context(context, font_scale) 
    sns.set_theme()
    sns.set_style("white")

    # create facets by row and column
    g = sns.FacetGrid(
        data,
        row=r_col,
        col=c_col,
        hue=cond_col,
        sharex=True,
        sharey=True,
        legend_out=True
    )

    # Map KDE to each facet
    g.map_dataframe(
        sns.kdeplot,
        x=x_col,
        fill=True,  
        alpha=0.3 
    )

    g.set(xlim = x_lim)

    g.add_legend(title=cond_col)

    # adjust figure size
    g.fig.set_size_inches(14, 7)

    if save_path:
        g.fig.savefig(save_path, dpi=dpi)

    plt.show()
    return g




def plot_min_x_z(data,
                 c_col,
                 r_col,
                 show_slopes = False,
                 slope_array = None,
                 target_x_array = None,
                 hue_col = 'water_speed_binary',
                 context='notebook',
                 font_scale=3,
                 save_path='../figures/baseline_trials_by_target.pdf',
                 dpi=300
                ):

    slope_array = np.array(slope_array, dtype=np.float64)
    target_array = np.array(target_x_array, dtype=np.float64)

    if data[c_col].dtype.name == 'category':
        data = data.copy()
        data[c_col] = data[c_col].cat.remove_unused_categories()

    # set grid and make facets by target
    g = sns.FacetGrid(data, 
                      col=c_col,
                      row=r_col,
                      hue = hue_col,
                      sharex=True, sharey=True)



    g.map_dataframe(sns.scatterplot,
                    data=data,
                    x='min_pos_from_target_x', y='min_pos_from_target_z',
                    alpha=0.02
                   )

    if show_slopes == True:

        assert g.axes.shape == slope_array.shape, "Check the slopes_array shape!"

        rows, cols = g.axes.shape
        
        for col_idx in range(cols):

            for row_idx in range(rows):

                ax = g.axes[row_idx, col_idx]
    
                slope_deg = slope_array[row_idx, col_idx]
                target_x = target_array[col_idx]
    
                slope_val = np.tan(np.deg2rad(slope_deg))
    
                ax.axline(xy1=(target_x, 1.4),
                          slope=slope_val,
                          color='black',
                          linestyle='--',
                          linewidth=2,
                          alpha=0.6)
                ax.set_aspect('equal')


    # Add legend
    g.add_legend()

    g.fig.set_size_inches(14, 7)   # width, height in inches
    
    # save figure
    if save_path:
        g.fig.savefig(save_path, dpi=dpi) 
        
    # display
    plt.show()

    return g



from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

def plot_heatmap(data, x_col, y_col, colour_col, facet_col=None, facet_row=None, 
                 style_col=None, mode='correlation', dark=True, font_scale=1.2, 
                 show_legend=True, show_mean_line=False, mean_line_color=None): 
    
    data = data.copy().reset_index()
    
    # Ensure numerical consistency for the heat map
    data[colour_col] = pd.to_numeric(data[colour_col], errors='coerce')
    
    palette = 'icefire' if dark else 'RdBu_r' 
    bg_color, text_color = ("black", "white") if dark else ("white", "black")
    
    if mean_line_color is None:
        mean_line_color = "white" if dark else "black"

    v_min, v_max = data[colour_col].min(), data[colour_col].max()
    norm = TwoSlopeNorm(vcenter=0.0, vmin=v_min, vmax=v_max) if v_min < 0 and v_max > 0 else plt.Normalize(vmin=v_min, vmax=v_max)

    sns.set_context("notebook", font_scale=font_scale)

    with sns.axes_style("darkgrid" if dark else "whitegrid", rc={
        "axes.facecolor": bg_color, "figure.facecolor": bg_color,
        "grid.color": "#333333" if dark else "#DDDDDD", "text.color": text_color,
        "axes.labelcolor": text_color, "xtick.color": text_color, "ytick.color": text_color
    }):
        # Primary Scatter Plot
        g = sns.relplot(
            data=data, x=x_col, y=y_col, hue=colour_col, 
            style=style_col, col=facet_col, row=facet_row,
            palette=palette, hue_norm=norm, alpha=0.4, kind='scatter', 
            height=5, aspect=1.0, facet_kws={'margin_titles': True}
        )
        g.fig.set_facecolor(bg_color)
        
        # Mean Overlay Logic
        if show_mean_line:
            if mean_line_color in data.columns:
                g.map_dataframe(
                    sns.lineplot, x=x_col, y=y_col, 
                    style=style_col, 
                    hue=mean_line_color, 
                    palette='viridis', 
                    linewidth=3, errorbar=None, zorder=10
                )
            else:
                g.map_dataframe(
                    sns.lineplot, x=x_col, y=y_col, 
                    style=style_col, color=mean_line_color, 
                    linewidth=3, errorbar=None, zorder=10
                )
        
        # Colorbar Handling
        sm = ScalarMappable(cmap=palette, norm=norm)
        sm.set_array([])
        # Placed at 0.85 to leave room for the legend on the right
        cbar_ax = g.fig.add_axes([0.85, 0.2, 0.02, 0.6]) 
        cbar = g.fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(colour_col, color=text_color)
        
        if g._legend: g._legend.remove()
        
        # Robust Legend Handling across all facets
        if show_legend:
            unique_labels = {}
            for ax in g.axes.flat:
                handles, labels = ax.get_legend_handles_labels()
                for h, l in zip(handles, labels):
                    if l not in unique_labels and not l.replace('.','',1).replace('-','',1).isdigit():
                        unique_labels[l] = h
            
            if unique_labels:
                # Placed at 0.92, to the right of the colorbar
                g.fig.legend(unique_labels.values(), unique_labels.keys(), 
                             loc='center left', bbox_to_anchor=(0.92, 0.5), 
                             fontsize='small', title="Factors")
        
        for ax in g.axes.flat:
            ax.set_facecolor(bg_color)
            if dark: ax.tick_params(colors=text_color)
        
        # Compress subplots to 82% figure width to prevent overlap
        plt.subplots_adjust(right=0.82, top=0.9) 
        plt.show()
        return g
