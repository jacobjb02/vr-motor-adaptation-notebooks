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
import matplotlib.lines as mlines

# Global static color mapping for targets
_colors = sns.color_palette("bright", 4)
TARGET_PALETTE = {
    "L60": _colors[2], # Green
    "L30": _colors[0], # Blue
    "R30":   _colors[1], # Orange
    "R60":   _colors[3]  # Red
}


# trial schedule plot
def plot_trial_schedule(
    data, 
    y_col,
    context='notebook',
    font_scale=2,
    save_path='../figures/trial_schedule.svg',
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
    font_scale=1.2,
    save_path='../figures/baseline_trials_by_target.pdf',
    dpi=300
):

    # filter for baseline
    #baseline_df = data[data['phase'] == 'baseline']

    sns.set_context(context, font_scale=font_scale)
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
    marker_size=2,
    font_scale=3,
    save_path='../figures/exposure_trials_by_target_x_set.svg',
    dpi=300
):
    data = data.copy()
    data[x_col] = data[x_col].astype(float)

    # --- Setup Condition Line Styles ---
    available_linestyles = ['-', '--', ':', '-.']
    unique_conditions = data[cond_col].dropna().unique() if cond_col in data.columns else []
    cond_style_dict = {
        cond: available_linestyles[i % len(available_linestyles)] 
        for i, cond in enumerate(unique_conditions)
    }

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
    
     # --- CREATE PHASE/SET ORDER IDENTIFIER FIRST ---
    if transition_col and transition_col in data.columns and cond_col in data.columns:
        data['_phase_id'] = data[transition_col].astype(str) + '_cond_' + data[cond_col].astype(str)
    elif transition_col and transition_col in data.columns:
        data['_phase_id'] = data[transition_col].astype(str)
    else:
        data['_phase_id'] = '0'
    
    # --- PAD MISSING X-VALUES WITH NANS TO BREAK LINES ---
    grouping_cols = [c for c in [ppid_col, cond_col, target_col, row_col, col_col, '_phase_id'] if c and c in data.columns]
    
    unique_groups = data[grouping_cols].drop_duplicates().assign(_key=1)
    unique_x = pd.DataFrame({x_col: data[x_col].dropna().unique(), '_key': 1})
    full_grid = pd.merge(unique_groups, unique_x, on='_key').drop('_key', axis=1)
    
    # Merge to insert rows with NaN in y_col for missing x-values
    data = pd.merge(full_grid, data, on=grouping_cols + [x_col], how='left')
    
    # CREATE COMPOSITE UNITS COLUMN AFTER THE MERGE
    data['_units_composite'] = data[ppid_col].astype(str) + '_' + data['_phase_id'].astype(str)

    # FILL NaN values in _units_composite for padded rows
    data['_units_composite'] = data.groupby(grouping_cols)['_units_composite'].transform(lambda x: x.ffill().bfill())
    
    # --- Categorical Assignment and Plotting Setup ---
    labels = sorted(data[target_col].dropna().unique(), key=str)
    data[target_col] = pd.Categorical(data[target_col], categories=labels, ordered=True)

    sns.set_context(context, font_scale=font_scale)
    sns.set_theme(style="whitegrid")

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
    for ax in g.axes.flat:
        for unit in data['_units_composite'].unique():
            if pd.isna(unit):
                continue
            unit_data = data[data['_units_composite'] == unit].sort_values(x_col)
            
            # Split by NaN gaps
            for target in unit_data[target_col].unique():
                target_unit = unit_data[unit_data[target_col] == target]
                color = TARGET_PALETTE.get(target, 'gray')
                
                # Plot only non-NaN segments
                mask = target_unit[y_col].notna()
                if mask.any():
                    ax.plot(target_unit[mask][x_col], target_unit[mask][y_col],
                           color=color, alpha=0.05, linewidth=0.5)
                    
    # 2. Plot Mean + SE separately per phase to prevent cross-phase connections
    axes_to_plot = []
    
    if row_col or col_col:
        for facet_key, ax in g.axes_dict.items():
            if not ax.get_visible():
                continue
            
            if isinstance(facet_key, tuple):
                row_val, col_val = facet_key
            else:
                row_val = facet_key
                col_val = None
            
            facet_data = data.copy()
            if row_col and row_col in data.columns:
                facet_data = facet_data[facet_data[row_col] == row_val]
            if col_col and col_col in data.columns:
                facet_data = facet_data[facet_data[col_col] == col_val]
            
            axes_to_plot.append((ax, facet_data))
    else:
        axes_to_plot.append((g.ax, data))
    
    # Plot means on each axis
    for ax, facet_data in axes_to_plot:
        
        # --- UPDATE: Add cond_col to GroupBy ---
        groupby_cols = [x_col, target_col, cond_col, '_phase_id']
        grouped = facet_data.dropna(subset=[y_col]).groupby(
            groupby_cols
        ).agg({y_col: ['mean', 'sem', 'count']}).reset_index()
        
        grouped.columns = groupby_cols + ['mean', 'sem', 'count']
        grouped['sem'] = grouped['sem'].fillna(0)
        
        for target_label in labels:
            color = TARGET_PALETTE[target_label]
            
            # --- UPDATE: Iterate through condition categories ---
            for cond_val in grouped[cond_col].unique():
                current_linestyle = cond_style_dict.get(cond_val, '-')
                
                # Filter by both target and condition
                target_data = grouped[
                    (grouped[target_col] == target_label) & 
                    (grouped[cond_col] == cond_val)
                ].sort_values([x_col])
                
                if len(target_data) > 0:
                    for phase_id in target_data['_phase_id'].unique():
                        phase_subset = target_data[target_data['_phase_id'] == phase_id].sort_values(x_col)
                        
                        if len(phase_subset) > 0:
                            ax.plot(
                                phase_subset[x_col],
                                phase_subset['mean'],
                                marker='o',
                                markersize=marker_size,
                                linewidth=1.75,
                                color=color,
                                alpha=0.80,
                                linestyle=current_linestyle  # <-- Applied here
                            )
                            
                            ax.fill_between(
                                phase_subset[x_col],
                                phase_subset['mean'] - phase_subset['sem'],
                                phase_subset['mean'] + phase_subset['sem'],
                                alpha=0.25,
                                color=color,
                                linewidth=0
                            )

    #g.fig.set_size_inches(24, 16)
    #g.fig.set_size_inches(36 / 2.54, 12 / 2.54)
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

            if show_zero_line:
                ax.axhline(y=0.0, color='black', linestyle='--', alpha=0.3)
                
            if show_sd_line:
                ax.axhline(y=global_mean + global_sd, color='red', linestyle=':', alpha=0.4, lw=3, zorder=1)
                ax.axhline(y=global_mean - global_sd, color='red', linestyle=':', alpha=0.4, lw=3, zorder=1)
                
                ax.axhline(y=global_mean + (2 * global_sd), color='red', linestyle=':', alpha=0.2, lw=1.5, zorder=1)
                ax.axhline(y=global_mean - (2 * global_sd), color='red', linestyle=':', alpha=0.2, lw=1.5, zorder=1)
                
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

    # --- CUSTOM LEGEND CREATION ---
    handles = []
    
    # 1. Target Colors
    for target_label, color in TARGET_PALETTE.items():
        handle = mlines.Line2D(
            [], [], 
            color=color, 
            marker='o', 
            markersize=marker_size, 
            linewidth=3.0, 
            linestyle='-', # Keep solid in legend for pure color reference
            label=f"Target: {target_label}"
        )
        handles.append(handle)

    # 2. Condition Linestyles
    for cond_val, l_style in cond_style_dict.items():
        handle = mlines.Line2D(
            [], [],
            color='gray', # Neutral color to show style independently
            marker='None',
            linewidth=3.0,
            linestyle=l_style,
            label=f"Cond: {cond_val}"
        )
        handles.append(handle)

    # Apply the legend
    if handles:
        g.fig.legend(
            handles=handles,
            title="Targets & Conditions",
            loc="center left", 
            bbox_to_anchor=(0.88, 0.5), 
            frameon=True
        )

    g.fig.subplots_adjust(right=0.82, bottom=0.2, left=0.1, wspace=0.1)

    if save_path:
        g.fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()
    return g







# early late exposure
def plot_early_late_exposure(
    data,
    cond_col, # colour
    ppid_col,
    y_col,
    x_col,
    line_col,
    facet_row,
    facet_col,
    target_col='target_x_label',  
    ylim=None,              
    show_zero_line=False,
    context='notebook',
    font_scale=1.2,          
    facet_height=4,          
    facet_aspect=1.2,        
    save_path='../figures/early_late_exposure_by_target_x_set.png',
    dpi=300
):
    sns.set_theme(context=context, font_scale=font_scale, style="white")
    
    data = data.copy()

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

    # Individual participant lines - colored by target
    g.map_dataframe(
        sns.lineplot,
        x=x_col, y=y_col,
        units=ppid_col,
        hue=target_col,
        estimator=None,
        palette=TARGET_PALETTE,
        alpha=0.10, 
        linewidth=0.8,
        legend=False
    )
        
    # Extract the explicit global order of hue levels for condition
    global_hue_order = list(data[cond_col].unique())
    
    # Mean + SE with pointplot - by condition
    g.map_dataframe(
        sns.pointplot,
        x=x_col, y=y_col,
        hue=cond_col,                    
        hue_order=global_hue_order,
        palette=TARGET_PALETTE,
        scale=0.8,
        estimator=np.mean,
        errorbar='se',
        capsize=.1
    )

    if ylim is not None:
        g.set(ylim=ylim)

    # --- HIDE EMPTY FACETS & RESTORE LABELS ---
    visible_bottom_axes = {}
    visible_left_axes = {}
    nrows, ncols = g.axes.shape

    for i in range(nrows):
        for j in range(ncols):
            ax = g.axes[i, j]

            # If no data elements were mapped to this axis, hide it
            if not ax.lines and not ax.collections:
                ax.set_visible(False)
                continue

            # Track the outermost visible axes for label restoration
            visible_bottom_axes[j] = ax
            if i not in visible_left_axes:
                visible_left_axes[i] = ax

            # Apply Zero Line only to populated axes
            if show_zero_line:
                ax.axhline(0.0, color='black', linestyle='--', alpha=0.3)

    # Restore X and Y labels/ticks on the new boundary axes
    for ax in visible_bottom_axes.values():
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.xaxis.label.set_visible(True) 
        
    for ax in visible_left_axes.values():
        ax.yaxis.set_tick_params(labelleft=True)
        ax.yaxis.label.set_visible(True) 
    
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
        hue=target_col, palette=TARGET_PALETTE,
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
                    hue = 'target_x_label', palette=TARGET_PALETTE,
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
                    hue=cond_col, palette=TARGET_PALETTE, alpha = 0.7,
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
                    hue = 'target_x_label', palette=TARGET_PALETTE,
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
                        palette=TARGET_PALETTE,
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
                 x_col_title,
                 y_col_title,
                 c_col,
                 r_col,
                 show_slopes=False,
                 slope_array=None,
                 target_x_array=None,
                 hue_col='water_speed_binary',
                 context='poster',      # Context set to poster
                 font_scale=0.4,        # Drastically reduced to offset 'poster' base multiplier
                 facet_height=4.5,      # Height in inches per subplot
                 facet_aspect=0.8,      # Matches spatial ratio: 200cm(X) / 250cm(Z) = 0.8
                 save_path='../figures/PCA_slopes.svg',
                 dpi=300
                ):

    slope_array = np.array(slope_array, dtype=np.float64)
    target_array = np.array(target_x_array, dtype=np.float64)
    
    data = data.copy()

    if context == 'poster':
        data = data[data[c_col].isin(['L60', 'R60'])]

    if data[c_col].dtype.name == 'category':
        data[c_col] = data[c_col].cat.remove_unused_categories()

    with sns.plotting_context(context=context, font_scale=font_scale):

        g = sns.FacetGrid(data, 
                          col=c_col,
                          row=r_col,
                          hue=hue_col,
                          height=facet_height, 
                          aspect=facet_aspect, 
                          sharex=True, sharey=True)
        
        g.map_dataframe(sns.scatterplot,
                        data=data,
                        x='min_pos_from_target_x_cm', y='min_pos_from_target_z_cm',
                        alpha=0.2       # Increased from 0.02 so smaller markers remain visible on posters
                       )
    
        if show_slopes == True:
            assert g.axes.shape == slope_array.shape, f"Check shape! Axes are {g.axes.shape} but slope_array is {slope_array.shape}"
            rows, cols = g.axes.shape
            
            for col_idx in range(cols):
                for row_idx in range(rows):
                    ax = g.axes[row_idx, col_idx]
                    slope_deg = slope_array[row_idx, col_idx]
                    target_x = target_array[col_idx]
                    slope_val = np.tan(np.deg2rad(slope_deg))
        
                    ax.axline(xy1=(target_x, 140),
                              slope=slope_val,
                              color='black',
                              linestyle='--',
                              linewidth=2,
                              alpha=0.6)
                    
                    ax.set_aspect('equal')
    
        # Move legend outside the plot area
        #g.add_legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
        g.set_axis_labels(x_col_title, y_col_title)
        
        # Strip redundant variables from facet titles (removes "c_col = L60")
        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        
        # Mechanically force whitespace to prevent text collisions
        # hspace: vertical gap between rows | wspace: horizontal gap between columns
        g.fig.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.85, hspace=0.5, wspace=0.3)
                
        if save_path:
            g.fig.savefig(save_path, dpi=dpi, bbox_inches='tight') 

    plt.show()

    return g



    

from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

def plot_heatmap(data, x_col, y_col, x_col_title, y_col_title, colour_col, facet_col=None, facet_row=None, 
                 style_col=None, mode='correlation', dark=True, font_scale=1.2, 
                 show_legend=True, show_mean_line=False, mean_line_color=None, 
                 colour_type='auto', palette='plasma'): 

    data = data.copy().reset_index(drop=True)
    
    # Drop rows with NaN in critical columns
    data = data.dropna(subset=[x_col, y_col, colour_col])
    if facet_col is not None:
        data = data.dropna(subset=[facet_col])
    if facet_row is not None:
        data = data.dropna(subset=[facet_row])
    
    # Detect colour_type if auto
    if colour_type == 'auto':
        try:
            pd.to_numeric(data[colour_col], errors='raise')
            colour_type = 'continuous'
        except (ValueError, TypeError):
            colour_type = 'categorical'
    
    bg_color, text_color = ("black", "white") if dark else ("white", "black")
    
    if mean_line_color is None:
        mean_line_color = "white" if dark else "black"

    sns.set_context("notebook", font_scale=font_scale)

    with sns.axes_style("darkgrid" if dark else "whitegrid", rc={
        "axes.facecolor": bg_color, "figure.facecolor": bg_color,
        "grid.color": "#333333" if dark else "#DDDDDD", "text.color": text_color,
        "axes.labelcolor": text_color, "xtick.color": text_color, "ytick.color": text_color
    }):
        
        if colour_type == 'continuous':
            # ============= CONTINUOUS COLOR MAPPING =============
            data[colour_col] = pd.to_numeric(data[colour_col], errors='coerce')
            
            v_min, v_max = data[colour_col].min(), data[colour_col].max()
            norm = TwoSlopeNorm(vcenter=0.0, vmin=v_min, vmax=v_max) if v_min < 0 and v_max > 0 else plt.Normalize(vmin=v_min, vmax=v_max)
            
            # Create plot without hue to avoid _hue error
            g = sns.relplot(
                data=data, x=x_col, y=y_col,
                style=style_col, col=facet_col, row=facet_row,
                alpha=0.6, kind='scatter', 
                height=5, aspect=1.0, facet_kws={'margin_titles': True}
            )
            g.fig.set_facecolor(bg_color)
            
            # Manually apply continuous colormap
            cmap = plt.cm.get_cmap(palette)
            for ax in g.axes.flat:
                collections = [c for c in ax.collections if hasattr(c, 'get_offsets')]
                
                if len(collections) > 0:
                    collection = collections[0]
                    offsets = collection.get_offsets()
                    
                    if len(offsets) > 0:
                        color_values = data[colour_col].values[:len(offsets)]
                        if len(color_values) > 0:
                            normalized_colors = norm(color_values)
                            colors = cmap(normalized_colors)
                            collection.set_color(colors)
            
            # Colorbar for continuous data
            sm = ScalarMappable(cmap=palette, norm=norm)
            sm.set_array([])
            cbar_ax = g.fig.add_axes([0.85, 0.2, 0.02, 0.6])
            cbar = g.fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label(colour_col, color=text_color)
            cbar.ax.tick_params(colors=text_color)
        
        else:
            # ============= CATEGORICAL COLOR MAPPING =============
            # Use hue for categorical data (works well with seaborn)
            g = sns.relplot(
                data=data, x=x_col, y=y_col, hue=colour_col,
                style=style_col, col=facet_col, row=facet_row,
                palette=palette, alpha=0.3, kind='scatter',
                height=5, aspect=1.0, facet_kws={'margin_titles': True}
            )
            g.fig.set_facecolor(bg_color)
        
        # Mean Overlay Logic
        if show_mean_line:
            if isinstance(mean_line_color, str) and mean_line_color in data.columns:
                g.map_dataframe(
                    sns.lineplot, x=x_col, y=y_col, 
                    style=style_col, 
                    hue=mean_line_color, 
                    palette='Set2', 
                    linewidth=3, errorbar=None, zorder=10
                )
            else:
                g.map_dataframe(
                    sns.lineplot, x=x_col, y=y_col, 
                    style=style_col, color=mean_line_color, 
                    linewidth=3, errorbar=None, zorder=10
                )
        
        # Legend Handling
        if g._legend:
            g._legend.remove()
        
        if show_legend and colour_type == 'categorical':
            unique_labels = {}
            for ax in g.axes.flat:
                handles, labels = ax.get_legend_handles_labels()
                for h, l in zip(handles, labels):
                    if l not in unique_labels and not l.replace('.','',1).replace('-','',1).isdigit():
                        unique_labels[l] = h
            
            if unique_labels:
                g.fig.legend(unique_labels.values(), unique_labels.keys(), 
                             loc='center left', bbox_to_anchor=(0.92, 0.5), 
                             fontsize='small', title=colour_col)
        
        # Styling
        for ax in g.axes.flat:
            ax.set_facecolor(bg_color)
            if dark:
                ax.tick_params(colors=text_color)
        
        g.set_axis_labels(x_col_title, y_col_title)
        plt.subplots_adjust(right=0.82, top=0.9)
        plt.show()
        
        return g




def plot_early_late_exposure_with_slopes(
    data,
    cond_col,      # colour
    ppid_col,
    y_col,
    x_col,
    line_col,
    facet_row,
    facet_col,
    target_col='target_x_label',
    ylim=None,
    show_zero_line=False,
    context='notebook',
    font_scale=1.2,
    facet_height=4,
    facet_aspect=1.2,
    jitter_amount=0.035,    # control jitter width
    slope_offset=0.035,     # gap between points and slope lines
    save_path='../figures/early_late_exposure_by_target_x_set_slopes.png',
    dpi=300
):
    
    sns.set_theme(context=context, font_scale=font_scale, style="white")
    
    data = data.copy()
    
    # Create numeric positions for categorical x_col
    unique_x_cats = sorted(data[x_col].dropna().unique())
    x_cat_to_numeric = {cat: i for i, cat in enumerate(unique_x_cats)}
    data['_x_numeric'] = data[x_col].map(x_cat_to_numeric)

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

    # --- JITTERED INDIVIDUAL POINTS BY TARGET (colored by target) ---
    for ax in g.axes.flat:
        for target_val in data[target_col].dropna().unique():
            target_data = data[data[target_col] == target_val].dropna(subset=[x_col, y_col, '_x_numeric'])
            
            if len(target_data) == 0:
                continue
            
            color = TARGET_PALETTE.get(target_val, 'gray')
            
            # Jitter the numeric x positions while keeping real y values
            x_jittered = target_data['_x_numeric'].values + np.random.normal(0, jitter_amount, len(target_data))
            
            ax.scatter(
                x_jittered,
                target_data[y_col].values,
                color=color,
                alpha=0.15,
                s=20,
                zorder=1
            )

    # --- REGRESSION SLOPES WITH CI (by condition, PER FACET) ---
    from scipy import stats
    
    global_hue_order = list(data[cond_col].dropna().unique())
    
    # Get facet coordinates to match axes
    if facet_row or facet_col:
        for facet_key, ax in g.axes_dict.items():
            if not ax.get_visible():
                continue
            
            # Extract facet values
            if isinstance(facet_key, tuple):
                row_val, col_val = facet_key
            else:
                row_val = facet_key
                col_val = None
            
            # Filter data for this facet
            facet_data = data.copy()
            if facet_row and facet_row in data.columns:
                facet_data = facet_data[facet_data[facet_row] == row_val]
            if facet_col and facet_col in data.columns:
                facet_data = facet_data[facet_data[facet_col] == col_val]
            
            # Check if facet has any valid data
            facet_data_valid = facet_data.dropna(subset=[y_col, '_x_numeric'])
            if len(facet_data_valid) == 0:
                ax.set_visible(False)
                continue
            
            # Now compute slopes for each condition using FACET-SPECIFIC data
            for cond_val in global_hue_order:
                cond_data = facet_data[facet_data[cond_col] == cond_val].dropna(subset=[y_col, '_x_numeric'])
                
                if len(cond_data) < 2:
                    continue
                
                color = 'gray'
                
                # Perform linear regression on numeric x positions
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    cond_data['_x_numeric'].values,
                    cond_data[y_col].values
                )
                
                # Generate x range for plotting (with gap offset)
                x_min = cond_data['_x_numeric'].min() + slope_offset
                x_max = cond_data['_x_numeric'].max() + slope_offset
                x_line = np.array([x_min, x_max])
                y_line = slope * x_line + intercept
                
                # Calculate prediction interval (95% CI)
                y_pred = slope * cond_data['_x_numeric'].values + intercept
                residuals = cond_data[y_col].values - y_pred
                residual_std_err = np.sqrt(np.sum(residuals**2) / (len(cond_data) - 2))
                
                # Standard error for prediction
                n = len(cond_data)
                x_mean = cond_data['_x_numeric'].mean()
                sxx = np.sum((cond_data['_x_numeric'].values - x_mean)**2)
                se_pred = residual_std_err * np.sqrt(1/n + (x_line - x_mean)**2 / sxx)
                
                # 95% CI
                ci_factor = 1.96
                y_upper = slope * x_line + intercept + ci_factor * se_pred
                y_lower = slope * x_line + intercept - ci_factor * se_pred
                
                # Plot slope line
                ax.plot(x_line, y_line, color=color, linewidth=2.5, alpha=0.8, zorder=10)
                
                # Plot CI band
                ax.fill_between(x_line, y_lower, y_upper, color=color, alpha=0.2, zorder=9)
    else:
        # Single plot (no faceting)
        for cond_val in global_hue_order:
            cond_data = data[data[cond_col] == cond_val].dropna(subset=[y_col, '_x_numeric'])
            
            if len(cond_data) < 2:
                continue
            
            color = 'gray'
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                cond_data['_x_numeric'].values,
                cond_data[y_col].values
            )
            
            x_min = cond_data['_x_numeric'].min() + slope_offset
            x_max = cond_data['_x_numeric'].max() + slope_offset
            x_line = np.array([x_min, x_max])
            y_line = slope * x_line + intercept
            
            y_pred = slope * cond_data['_x_numeric'].values + intercept
            residuals = cond_data[y_col].values - y_pred
            residual_std_err = np.sqrt(np.sum(residuals**2) / (len(cond_data) - 2))
            
            n = len(cond_data)
            x_mean = cond_data['_x_numeric'].mean()
            sxx = np.sum((cond_data['_x_numeric'].values - x_mean)**2)
            se_pred = residual_std_err * np.sqrt(1/n + (x_line - x_mean)**2 / sxx)
            
            ci_factor = 1.96
            y_upper = slope * x_line + intercept + ci_factor * se_pred
            y_lower = slope * x_line + intercept - ci_factor * se_pred
            
            g.ax.plot(x_line, y_line, color=color, linewidth=2.5, alpha=0.8, zorder=10)
            g.ax.fill_between(x_line, y_lower, y_upper, color=color, alpha=0.2, zorder=9)

    if ylim is not None:
        g.set(ylim=ylim)

    # --- HIDE EMPTY FACETS & RESTORE LABELS ---
    visible_bottom_axes = {}
    visible_left_axes = {}
    nrows, ncols = g.axes.shape

    for i in range(nrows):
        for j in range(ncols):
            ax = g.axes[i, j]

            # Hide if no data and no lines/collections
            if not ax.collections and not ax.lines:
                ax.set_visible(False)
                continue

            visible_bottom_axes[j] = ax
            if i not in visible_left_axes:
                visible_left_axes[i] = ax

            if show_zero_line:
                ax.axhline(0.0, color='black', linestyle='--', alpha=0.3)

    # Restore X and Y labels/ticks on the new boundary axes
    for ax in visible_bottom_axes.values():
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.xaxis.label.set_visible(True)

    for ax in visible_left_axes.values():
        ax.yaxis.set_tick_params(labelleft=True)
        ax.yaxis.label.set_visible(True)

    # --- SET X-AXIS TICKS TO CATEGORY LABELS ---
    for ax in visible_bottom_axes.values():
        ax.set_xticks(range(len(unique_x_cats)))
        ax.set_xticklabels(unique_x_cats)

    # --- CUSTOM LEGEND ---
    handles = []
    
    # Target colors
    for target_label, color in TARGET_PALETTE.items():
        handle = mlines.Line2D(
            [], [],
            color=color,
            marker='o',
            markersize=6,
            linewidth=0,
            label=f"Target: {target_label}"
        )
        handles.append(handle)
    
    # Condition line (gray for slopes)
    handle = mlines.Line2D(
        [], [],
        color='gray',
        linewidth=2.5,
        label="Slope (by condition)"
    )
    handles.append(handle)

    if handles:
        g.fig.legend(
            handles=handles,
            title=cond_col,
            loc="center left",
            bbox_to_anchor=(0.92, 0.5),
            frameon=True
        )

    if save_path:
        g.fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()
    return g


def plot_violin_with_slopes(
    data,
    cond_col,      # colour for violins
    ppid_col,
    y_col,
    x_col,
    facet_row,
    facet_col,
    target_col='target_x_label',
    ylim=None,
    show_zero_line=False,
    context='notebook',
    font_scale=1.2,
    facet_height=4,
    facet_aspect=1.2,
    slope_x_buffer=0.5,    # NEW: extends slope line beyond data range
    facet_row_order=None,
    facet_col_order=None,
    save_path='../figures/violin_with_slopes.png',
    dpi=300
):
    
    sns.set_theme(context=context, font_scale=font_scale, style="white")
    
    data = data.copy()
    
    # Create numeric positions for categorical x_col
    unique_x_cats = sorted(data[x_col].dropna().unique())
    x_cat_to_numeric = {cat: i for i, cat in enumerate(unique_x_cats)}
    data['_x_numeric'] = data[x_col].map(x_cat_to_numeric).astype(float)

    # Set default order if not provided
    if facet_row_order is None and facet_row:
        facet_row_order = sorted(data[facet_row].dropna().unique())
    if facet_col_order is None and facet_col:
        facet_col_order = sorted(data[facet_col].dropna().unique())

    g = sns.FacetGrid(
        data,
        col=facet_col,
        row=facet_row,
        col_order=facet_col_order,
        row_order=facet_row_order,
        height=facet_height,
        aspect=facet_aspect,
        sharey=True,
        sharex=True,
        margin_titles=True
    )

    # --- VIOLIN PLOTS (colored by condition) ---
    g.map_dataframe(
        sns.violinplot,
        x=x_col,
        y=y_col,
        hue=cond_col,
        palette=TARGET_PALETTE,
        inner='quartile',
        alpha=0.15,
        cut=0,
        linewidth=1.5
    )

    # Individual data points
    g.map_dataframe(
        sns.stripplot,
        x=x_col,
        y=y_col,
        hue=cond_col,
        palette=TARGET_PALETTE,
        size=3,
        alpha=0.3,
        dodge=True
    )

    # --- REGRESSION SLOPES WITH CI (by condition, PER FACET) ---
    from scipy import stats
    
    global_hue_order = sorted(list(data[cond_col].dropna().unique()), key=str)
    
    # Get facet coordinates to match axes
    if facet_row or facet_col:
        for facet_key, ax in g.axes_dict.items():
            if not ax.get_visible():
                continue
            
            # Extract facet values
            if isinstance(facet_key, tuple):
                row_val, col_val = facet_key
            else:
                row_val = facet_key
                col_val = None
            
            # Filter data for this facet
            facet_data = data.copy()
            if facet_row and facet_row in data.columns:
                facet_data = facet_data[facet_data[facet_row] == row_val]
            if facet_col and facet_col in data.columns:
                facet_data = facet_data[facet_data[facet_col] == col_val]
            
            # Check if facet has any valid data
            facet_data_valid = facet_data.dropna(subset=[y_col, '_x_numeric'])
            if len(facet_data_valid) == 0:
                ax.set_visible(False)
                continue
            
            # Check if all x_col categories are present in this facet
            facet_x_cats = sorted(facet_data[x_col].dropna().unique())
            if facet_x_cats != unique_x_cats:
                # Missing x categories in this facet
                ax.set_visible(False)
                continue
            
            # Compute slopes for each condition using FACET-SPECIFIC data
            for cond_val in global_hue_order:
                cond_data = facet_data[facet_data[cond_col] == cond_val].dropna(subset=[y_col, '_x_numeric'])
                
                if len(cond_data) < 2:
                    continue
                
                # Get color for this condition from TARGET_PALETTE
                color = TARGET_PALETTE.get(cond_val, 'gray')
                
                # Perform linear regression on numeric x positions
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    cond_data['_x_numeric'].values,
                    cond_data[y_col].values
                )
                
                # Extend slope line BEYOND the data range with buffer
                x_min = cond_data['_x_numeric'].min() - slope_x_buffer
                x_max = cond_data['_x_numeric'].max() + slope_x_buffer
                x_line = np.array([x_min, x_max])
                y_line = slope * x_line + intercept
                
                # Calculate prediction interval (95% CI)
                y_pred = slope * cond_data['_x_numeric'].values + intercept
                residuals = cond_data[y_col].values - y_pred
                
                if len(cond_data) > 2:
                    residual_std_err = np.sqrt(np.sum(residuals**2) / (len(cond_data) - 2))
                else:
                    residual_std_err = np.std(residuals)
                
                # Standard error for prediction
                n = len(cond_data)
                x_mean = cond_data['_x_numeric'].mean()
                sxx = np.sum((cond_data['_x_numeric'].values - x_mean)**2)
                
                if sxx > 0:
                    se_pred = residual_std_err * np.sqrt(1/n + (x_line - x_mean)**2 / sxx)
                else:
                    se_pred = np.array([residual_std_err, residual_std_err])
                
                # 95% CI
                ci_factor = 1.96
                y_upper = slope * x_line + intercept + ci_factor * se_pred
                y_lower = slope * x_line + intercept - ci_factor * se_pred
                
                # Plot slope line with condition color - thicker and more opaque
                ax.plot(x_line, y_line, color=color, linewidth=3, alpha=1.0, zorder=10, solid_capstyle='round')
                
                # Plot CI band with condition color - lighter shade
                ax.fill_between(x_line, y_lower, y_upper, color=color, alpha=0.25, zorder=9)
    else:
        # Single plot (no faceting)
        for cond_val in global_hue_order:
            cond_data = data[data[cond_col] == cond_val].dropna(subset=[y_col, '_x_numeric'])
            
            if len(cond_data) < 2:
                continue
            
            color = TARGET_PALETTE.get(cond_val, 'gray')
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                cond_data['_x_numeric'].values,
                cond_data[y_col].values
            )
            
            x_min = cond_data['_x_numeric'].min() - slope_x_buffer
            x_max = cond_data['_x_numeric'].max() + slope_x_buffer
            x_line = np.array([x_min, x_max])
            y_line = slope * x_line + intercept
            
            y_pred = slope * cond_data['_x_numeric'].values + intercept
            residuals = cond_data[y_col].values - y_pred
            residual_std_err = np.sqrt(np.sum(residuals**2) / (len(cond_data) - 2))
            
            n = len(cond_data)
            x_mean = cond_data['_x_numeric'].mean()
            sxx = np.sum((cond_data['_x_numeric'].values - x_mean)**2)
            se_pred = residual_std_err * np.sqrt(1/n + (x_line - x_mean)**2 / sxx)
            
            ci_factor = 1.96
            y_upper = slope * x_line + intercept + ci_factor * se_pred
            y_lower = slope * x_line + intercept - ci_factor * se_pred
            
            g.ax.plot(x_line, y_line, color=color, linewidth=3, alpha=1.0, zorder=10, solid_capstyle='round')
            g.ax.fill_between(x_line, y_lower, y_upper, color=color, alpha=0.25, zorder=9)

    if ylim is not None:
        g.set(ylim=ylim)

    # --- HIDE EMPTY FACETS & RESTORE LABELS ---
    visible_bottom_axes = {}
    visible_left_axes = {}
    nrows, ncols = g.axes.shape

    for i in range(nrows):
        for j in range(ncols):
            ax = g.axes[i, j]

            # Hide if no data and no lines/collections
            if not ax.collections and not ax.lines:
                ax.set_visible(False)
                continue

            visible_bottom_axes[j] = ax
            if i not in visible_left_axes:
                visible_left_axes[i] = ax

            if show_zero_line:
                ax.axhline(0.0, color='black', linestyle='--', alpha=0.3, linewidth=1.5)

    # Restore X and Y labels/ticks on the new boundary axes
    for ax in visible_bottom_axes.values():
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.xaxis.label.set_visible(True)

    for ax in visible_left_axes.values():
        ax.yaxis.set_tick_params(labelleft=True)
        ax.yaxis.label.set_visible(True)

    # --- CUSTOM LEGEND ---
    handles = []
    
    # Condition colors
    for cond_label in global_hue_order:
        color = TARGET_PALETTE.get(cond_label, 'gray')
        handle = mlines.Line2D(
            [], [],
            color=color,
            linewidth=3,
            label=cond_label
        )
        handles.append(handle)

    if handles:
        g.fig.legend(
            handles=handles,
            title=cond_col,
            loc="center left",
            bbox_to_anchor=(0.92, 0.5),
            frameon=True,
            fontsize='medium'
        )

    g.fig.subplots_adjust(right=0.85)

    if save_path:
        g.fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()
    return g