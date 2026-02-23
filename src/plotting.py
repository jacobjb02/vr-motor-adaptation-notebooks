"""
Plotting functions.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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






# early late exposure
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_early_late_exposure(
    data,
    cond_col,
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

    n_hues = data[cond_col].nunique()
    dynamic_markers = ["o", "s", "D", "v"][:n_hues]
    dynamic_linestyles = ["-", "--", "-.", ":"][:n_hues]

    # Mean line and SE
    g.map_dataframe(
        sns.pointplot,
        x=x_col, y=y_col,
        hue=cond_col,            
        order=dynamic_x_order, 
        palette=palette,
        linestyles=dynamic_linestyles, 
        markers=dynamic_markers,      
        scale=0.8,
        estimator=np.mean,
        errorbar='se',
        capsize=.1,
        legend=False
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
    
# all exposure
import matplotlib.ticker as ticker

def plot_exposure_trials(
    data,
    cond_col,
    ppid_col,
    row_col,
    col_col,
    target_col,
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

    labels = sorted(data[target_col].unique(), key=str)
    data[target_col] = pd.Categorical(data[target_col], categories=labels, ordered=True)
    palette_map = dict(zip(labels, sns.color_palette("bright", len(labels))))

    sns.set_context(context, font_scale=font_scale)
    sns.set_theme(style="white")

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
        alpha=0.03, legend=False
    )

    # 2. Plot Mean + SE
    g.map_dataframe(
        sns.lineplot,
        x=x_col, y=y_col,
        estimator=estimator,
        linewidth=1.5,
        errorbar='se', err_kws={"alpha":0.25,"linewidth":0},
        hue=target_col, style=cond_col,
        markers=True,
        markersize=marker_size,
        palette=palette_map,
        alpha=1, dashes=True
    )

    g.fig.set_size_inches(18, 11) # Widened further for high font scale labels

    # --- AGGRESSIVE TICKER FIX ---
    for ax in g.axes.flat:
        if show_zero_line:
            ax.axhline(y=0.0, color='black', linestyle='--', alpha=0.3)
        
        # Limit the number of ticks to 4-5 to prevent overlap at large font sizes
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
        
        # Remove scientific notation if it appears
        ax.xaxis.get_major_formatter().set_scientific(False)

    # --- LEGEND & SPACING FIX ---
    handles, legend_labels = g.axes.flat[0].get_legend_handles_labels()
    
    # Legend centered on the right gutter
    g.fig.legend(handles, legend_labels,
                 title=cond_col.replace("_"," ").title(),
                 loc="center left", 
                 bbox_to_anchor=(0.88, 0.5), 
                 frameon=True)

    # Increased right/bottom margins to accommodate rotated text and external legend
    g.fig.subplots_adjust(right=0.82, bottom=0.2, wspace=0.1) 

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



def plot_heatmap(data, x_col, y_col, colour_col, facet_col=None, facet_row=None, 
                 style_col=None, mode='correlation', dark=True, font_scale=1.2, 
                 show_legend=True, show_mean_line=False, mean_line_color=None): 
    
    # SAFETY: Ensure we are working with columns, not indices
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
            # Check if mean_line_color refers to a data column
            if mean_line_color in data.columns:
                g.map_dataframe(
                    sns.lineplot, x=x_col, y=y_col, 
                    style=style_col, 
                    hue=mean_line_color, 
                    palette='viridis', 
                    linewidth=3, errorbar=None, zorder=10
                )
            else:
                # Treat as a literal Matplotlib color string
                g.map_dataframe(
                    sns.lineplot, x=x_col, y=y_col, 
                    style=style_col, color=mean_line_color, 
                    linewidth=3, errorbar=None, zorder=10
                )
        
        # Colorbar and Legend Handling
        sm = ScalarMappable(cmap=palette, norm=norm)
        sm.set_array([])
        cbar_ax = g.fig.add_axes([0.92, 0.2, 0.02, 0.6]) 
        cbar = g.fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(colour_col, color=text_color)
        
        if g._legend: g._legend.remove()
        if show_legend:
            handles, labels = g.axes.flat[0].get_legend_handles_labels()
            # Filter out the colorbar's numeric labels to keep factor labels only
            unique_labels = {}
            for h, l in zip(handles, labels):
                if l not in unique_labels and not l.replace('.','',1).replace('-','',1).isdigit():
                    unique_labels[l] = h
            
            if unique_labels:
                g.fig.legend(unique_labels.values(), unique_labels.keys(), 
                             loc='center right', bbox_to_anchor=(0.91, 0.5), 
                             fontsize='small', title="Factors")
        
        for ax in g.axes.flat:
            ax.set_facecolor(bg_color)
            if dark: ax.tick_params(colors=text_color)
        
        plt.subplots_adjust(right=0.85, top=0.9) 
        plt.show()
        return g