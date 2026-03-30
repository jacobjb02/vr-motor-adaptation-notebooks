import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def trial_trajectory(
    data,
    x_array,
    z_array,
    ppid_col='ppid_full',   
    max_trials_per_participant=100,
    random_state=42,
    water_col='water_speed_m_s',
    target_col='target_x_label',
    trial_col='trial_num'
):
    """
    Parse underscore-separated coordinate strings and plot trial trajectories
    without cross-trial line connections.

    Speed optimization: random sample up to `max_trials_per_participant`
    trials per participant before expansion.
    """
    df = data.copy()

    # 0) Sample up to N trials per participant (FASTEST place to reduce size)
    if ppid_col in df.columns:
        df = (
            df.groupby(ppid_col, group_keys=False)
              .apply(lambda g: g.sample(n=min(len(g), max_trials_per_participant),
                                        random_state=random_state))
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
            })

    df_long = pd.DataFrame(rows).sort_values([plot_trial_col, 'point_idx'])

    if dropped_mismatch > 0:
        print(f"[trial_trajectory] Warning: dropped {dropped_mismatch} unmatched x/z points due to unequal lengths.")

    g = sns.relplot(
        data=df_long,
        x=z_array,
        y=x_array,
        col=target_col,
        row=water_col,
        hue=plot_trial_col,
        units=plot_trial_col,
        kind="line",
        estimator=None,
        sort=False,
        alpha=0.2,
        linewidth=1.0,
        legend=False
    )

    g.set_axis_labels(f"X Coordinate ({x_array})", f"Z Coordinate ({z_array})")
    g.fig.suptitle("Participant Trial Trajectories (sampled)", y=1.02)
    plt.show()

    return df_long