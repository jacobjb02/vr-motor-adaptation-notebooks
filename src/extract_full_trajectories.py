import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def trial_trajectory(
    data,
    x_array,
    z_array,
    colour_col,
    ppid_col='ppid_full',   
    max_trials_per_participant=496, #496
    random_state=42,
    water_col='water_speed_m_s', 
    target_col='target_x_label',
    trial_col='trial_num',
    save_path='../figures/traject.svg'
):

    
    df = data.copy()

    # Sample up to N trials per participant 
    if ppid_col in df.columns:
        df = (
            df.groupby(ppid_col, group_keys=False, observed=False)
              .apply(lambda g: g.sample(n=min(len(g), max_trials_per_participant),
                                        random_state=random_state), include_groups=False)
              .reset_index(drop=True)
        )
    else:
        print(f"[trial_trajectory] Warning: '{ppid_col}' not found; sampling globally.")
        df = df.sample(n=min(len(df), max_trials_per_participant), random_state=random_state).reset_index(drop=True)

    # Ensure a unique trial grouping column exists
    if trial_col is None or trial_col not in df.columns:
        trial_col = 'trial_id'
        df[trial_col] = df.index.astype(str)

    # Always create unique plotting id to prevent any accidental cross-trial linking
    df['_row_id'] = df.index.astype(str)
    df[trial_col] = df[trial_col].astype(str)
    df['_trial_uid'] = df[trial_col] + '__row' + df['_row_id']
    plot_trial_col = '_trial_uid'

    # Parse strings -> float lists
    def parse_coord_string(s):
        if pd.isna(s):
            return []
        parts = str(s).replace("'", "").split('_')
        return [float(v) for v in parts if v != ""]

    df['_x_list'] = df[x_array].apply(parse_coord_string)
    df['_z_list'] = df[z_array].apply(parse_coord_string)

    # Build long dataframe safely (truncate to min length per row)
    rows = []
    dropped_mismatch = 0

    for _, r in df.iterrows():
        xs = r['_x_list']
        zs = r['_z_list']
        n = min(len(xs), len(zs))
        dropped_mismatch += abs(len(xs) - len(zs))

        for t in range(n):
            rows.append({
                x_array: xs[t],
                z_array: zs[t],
                'point_idx': t,
                water_col: r[water_col],
                target_col: r[target_col],
                ppid_col: r[ppid_col] if ppid_col in df.columns else None,
                trial_col: r[trial_col],
                plot_trial_col: r[plot_trial_col],
                colour_col: r[colour_col]
            })


    df_long = pd.DataFrame(rows).sort_values([plot_trial_col, 'point_idx', colour_col])

    df_long[colour_col] = pd.to_numeric(df_long[colour_col], errors='coerce')

    if dropped_mismatch > 0:
        print(f"[trial_trajectory] Warning: dropped {dropped_mismatch} unmatched x/z points due to unequal lengths.")

    sns.set_style("whitegrid")


    target_order = sorted(df_long[target_col].unique(), reverse=False) # plot facet order
        
    g = sns.relplot(
        data=df_long,
        x=z_array,
        y=x_array,
        col=target_col,
        col_order=target_order, 
        row=water_col,
        hue=colour_col,
        units=plot_trial_col,
        kind="line",
        palette="viridis", 
        estimator=None,
        height=6.0,
        aspect=1.0,
        sort=False,
        alpha=0.8,
        linewidth=1.0,
        legend="brief",   
        facet_kws={'sharex': True, 'sharey': True}
    )

    # Change gridline thickness
    for ax in g.axes.flat:
        ax.grid(True, linewidth=3.0)  
        ax.set_xlim(-1.0, 1.5)  # Sets the horizontal bounds for each facet
        
    # 3. Layout and Spacing
    g.set_axis_labels(f"X Coordinate ({x_array})", f"Z Coordinate ({z_array})")
    g.fig.suptitle("Participant Trial Trajectories", y=1.05, fontsize=14)
    g.tight_layout()

    if save_path:
        # Added default dpi value to prevent NameError
        g.fig.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    return