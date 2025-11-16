"""
Plotting functions.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd




# trial schedule plot
def plot_trial_schedule(
    data, 
    context='notebook',
    font_scale=2,
    save_path='../figures/trial_schedule.pdf',
    dpi=300
):

    # check reqiured cols
    assert 'trial_num' in data.columns and 'water_speed_binary' in data.columns, \
            "Data must contain 'trial_num' and 'water_speed_binary'"

    # edit font size:
    sns.set_context(context, font_scale=5)  
    sns.set_theme()
    sns.set_style("white")

        
    # build grid
    g = sns.relplot(
        data=data, 
        kind='line', linewidth = 3,
        x='trial_num', y='water_speed_binary',
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






    

# baseline plot
def plot_baseline(
    data,
    ppid_col,
    cond_col,
    y_col='ball_dist_to_center_cm',
    y_lim=(None,50.0),
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
                      col_order=["neg0.6", "neg0.3", "p0.3", "p0.6"],
                      sharex=False, sharey=True)

    # clean facet titles 
    g.set_titles("")
    
    # x and y lims
    g.set(ylim=y_lim)
    g.set(xlim=(7, 13))
    
    # individual data
    g.map_dataframe(sns.lineplot,
                    data=data, units=ppid_col, estimator=None,
                    x='trial_num_target', y=y_col,
                    linewidth = 2, hue='target_x_label', alpha=0.10
                   )
    
    # mean data
    g.map_dataframe(sns.lineplot,
                    data=data,
                    x='trial_num_target', y=y_col,
                    estimator='mean', errorbar='se', err_kws={'alpha':0.35, 'linewidth':0},
                    linewidth=3, hue='target_x_label', style=cond_col
                   )
                    
    # set axis labels
    #g.set_axis_labels('Trial Number (per target)', 'Min Distance (cm)')

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
def plot_early_late_exposure(
    data,
    cond_col,
    ppid_col,
    y_col='mean_dist',
    context='notebook',
    font_scale=3,
    save_path='../figures/early_late_exposure_by_target_x_set.pdf',
    dpi=300
):

    # Ensure categories
    data[cond_col] = data[cond_col].astype('category')
    data['target_x_label'] = data['target_x_label'].astype('category')

    print(f"\nCondition levels ({cond_col}): {list(data[cond_col].cat.categories)}")
    print(f"Target levels: {list(data['target_x_label'].cat.categories)}\n")

    sns.set_context(context, font_scale)
    sns.set_theme()
    sns.set_style("white")

    g = sns.FacetGrid(
        data,
        col='set_order',
        row=cond_col,
        sharey=True,
        sharex=True,
        margin_titles=True
    )

    g.set_titles("")

    # phase-based palette for raw points
    phase_palette = {
        'baseline':   '#a9c358',
        'training_1': '#58b5e1',
        'washout_1':  '#56ebd3',
        'training_2': '#691b9e',
        'washout_2':  '#115d52',
    }

    # INDIVIDUAL DOTS
    g.map_dataframe(
        sns.stripplot,
        x='section', y=y_col,
        hue='phase',
        order=['early', 'late'],
        palette=phase_palette,
        jitter=0.05, alpha=0.7, size=5
    )

    # add legend ONLY for phases
    handles, labels = g.axes[0,0].get_legend_handles_labels()
    g.axes[0,0].legend(handles, labels, title="Phase")

    # INDIVIDUAL TRAJECTORY LINES (gray)
    g.map_dataframe(
        sns.lineplot,
        x='section', y=y_col,
        units=ppid_col, estimator=None,
        color='0.4', alpha=0.35, linewidth=1
    )

    # SUMMARY POINTS (target hue)
    g.map_dataframe(
        sns.pointplot,
        x='section', y=y_col,
        order=['early','late'],
        hue='target_x_label',
        palette='bright',
        dodge=0.5,
        estimator=np.mean,
        errorbar='se',
        capsize=0.15
    )

    # target legend manually added to figure (avoids double legends)
    target_levels = list(data['target_x_label'].cat.categories)
    palette = sns.color_palette('bright', len(target_levels))
    target_handles = [plt.Line2D([0], [0], marker='o', color=palette[i],
                                 linestyle='-', markersize=10)
                      for i in range(len(target_levels))]

    g.fig.legend(
        target_handles,
        target_levels,
        title="Target",
        loc="upper right",
        bbox_to_anchor=(1.15, 0.95)
    )

    # formatting
    g.fig.set_size_inches(14, 10.5)

    if save_path:
        g.fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()
    return g





# all exposure
def plot_exposure_trials(
    data,
    cond_col,
    ppid_col,
    y_col='baseline_corrected_dist',
    y_lim=(None,110.0),
    x_col='trial_num_target',
    estimator='mean',
    context='notebook',
    font_scale=3,
    save_path='../figures/exposure_trials_by_target_x_set.pdf',
    dpi=300
):
    data = data.copy()
    data[x_col] = data[x_col].astype(float)

    # sort levels to preserve interpretable order
    labels = sorted(data[cond_col].unique(), key=str)

    data[cond_col] = pd.Categorical(data[cond_col],
                                    categories=labels,
                                    ordered=True)
    data["set_order"] = data["set_order"].astype("category")

    palette_map = dict(zip(labels, sns.color_palette("bright", len(labels))))

    sns.set_context(context, font_scale)
    sns.set_theme()
    sns.set_style("white")

    g = sns.FacetGrid(
        data,
        row='set_order',
        col='target_x_label',
        sharex=True,
        sharey=True,
    )
    g.set(ylim=y_lim)

    # individual participant traces
    g.map_dataframe(
        sns.lineplot,
        x=x_col, y=y_col,
        units=ppid_col, estimator=None,
        hue=cond_col, style=cond_col,
        palette=palette_map,
        alpha=0.03
    )

    # mean + SE
    g.map_dataframe(
        sns.lineplot,
        x=x_col, y=y_col,
        estimator=estimator,
        errorbar='se', err_kws={"alpha":0.25,"linewidth":0},
        hue=cond_col, style=cond_col,
        palette=palette_map,
        alpha=1, dashes=True
    )

    g.fig.set_size_inches(14, 10.5)

    legend_title = cond_col.replace("_"," ").title()

    handles = [
        plt.Line2D([0],[0], color=palette_map[label], lw=3)
        for label in labels
    ]

    g.fig.legend(handles, labels,
                 title=legend_title,
                 loc="upper right", bbox_to_anchor=(1.12,0.85))

    if save_path:
        g.fig.savefig(save_path, dpi=dpi)

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
                    alpha=0.025)
    
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
                    legend=True
    )


    g.fig.set_size_inches(14, 7)   # width, height in inches

    # Add legend
    g.add_legend(title="Speed Label")

        
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
    y_col='launch_deviation',
    y_lim=(-10,50),
    start_trial=7,
    block_len=4,
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
                    x='phase_trial_target', y=y_col,
                    estimator=None, units=ppid_col,
                    hue = 'target_x_label', palette='bright',
                    alpha=0.1)

    
    if show_hits == True:
                  
        hit_palette = {'False': 'red', 'True': 'green'}
        hit_markers = {'False': 'o',   'True': 's'}
        hit_dashes  = {'False': '',    'True': (2, 2)}

        
        g.map_dataframe(
                        sns.lineplot,
                        x='phase_trial_target', y=y_col,
                        estimator='mean', errorbar='se',
                        hue='target_hit',
                        style='target_hit',
                        palette=hit_palette,
                        markers=hit_markers,
                        dashes=hit_dashes,
                        markersize=marker_size,
                        alpha=1
                        )
        
        g.add_legend(title="Target Hit")

    elif show_speeds == True:

        g.map_dataframe(
                        sns.lineplot,
                        x='phase_trial_target', y=y_col,
                        estimator='mean', errorbar='se',
                        hue='target_x_label',
                        style=speed_col,
                        palette='bright',
                        markers=True,
                        markersize=marker_size,
                        alpha=1
                        )
        
        g.add_legend(title="Target Hit")


    else:   
        # mean line and se bands
        g.map_dataframe(sns.lineplot,
                        x='phase_trial_target', y=y_col, marker='o', markersize=marker_size,
                        estimator='mean', errorbar='se', err_kws={'alpha':0.25, 'linewidth':0},
                        hue = 'target_x_label', palette='bright', alpha=1, dashes=True)


    # final trial per target 
    end_trial = data['phase_trial_target'].max()      # 36 total trials per target
    
    # total number of blocks
    blocks = np.arange(start_trial, end_trial + block_len, block_len)
    
    for ax in g.axes.flat:
        for i in range(len(blocks) - 1):
            # skip blocks that end before starting trial
            if blocks[i + 1] <= start_trial:
                continue
            if i % 2 == 1:  # shade only odd-numbered blocks (i.e., in this case when water current is active)
                ax.axvspan(blocks[i] - 0.5, 
                           blocks[i + 1] - 0.5,
                           color='black', alpha=0.15)
    
    # add horizontal line at error of 0
    for ax in g.axes.flat:
        ax.axhline(y=0.0, color = 'black', linestyle='--', alpha = 0.3)
        ax.set_xticks(range(1, int(data['phase_trial_target'].max()) + 1, 4))
    
    
    # add vertical line at x = 7 (seperates baseline from washout: washout starts on x = 7)
    for ax in g.axes.flat:
        ax.axvline(x=start_trial - 0.5, color='red', linestyle='--', linewidth=1, alpha=0.8)
    
    g.fig.set_size_inches(14, 7)   # width, height in inches

    # Save
    if save_path:
        g.fig.savefig(save_path, dpi=dpi) 

    # display
    plt.show()

    return g
    





