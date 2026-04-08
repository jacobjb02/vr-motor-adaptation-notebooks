from concave_hull import concave_hull_indexes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.interpolate import splprep, splev
from scipy.spatial import ConvexHull
from matplotlib.ticker import MultipleLocator

def scale_to_panel(x, y, xlim, ylim):
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    
    xs = (np.asarray(x) - xlim[0]) / (x_range if x_range != 0 else 1.0)
    ys = (np.asarray(y) - ylim[0]) / (y_range if y_range != 0 else 1.0)
    return np.column_stack([xs, ys])

def unscale_from_panel(points_scaled, xlim, ylim):
    x = points_scaled[:, 0] * (xlim[1] - xlim[0]) + xlim[0]
    y = points_scaled[:, 1] * (ylim[1] - ylim[0]) + ylim[0]
    return np.column_stack([x, y])

def smooth_closed_polygon(points, n_interp=300, smooth=0.0015):
    pts = np.asarray(points)

    if len(pts) < 3:
        return pts

    # 1. Strip ALL adjacent duplicates to prevent singular matrices in splprep
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep_idx = np.insert(diffs > 1e-8, 0, True) 
    pts = pts[keep_idx]

    if len(pts) < 4:
        return pts

    # 2. Ensure closed loop explicitly for periodic parameterization
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    # 3. Dynamically set spline degree (Max k=3 for cubic). 
    k_degree = min(3, len(pts) - 1)

    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=smooth, per=True, k=k_degree)
        u_new = np.linspace(0, 1, n_interp, endpoint=False)
        x_new, y_new = splev(u_new, tck)
        
        return np.column_stack([x_new, y_new])
        
    except ValueError:
        return points

def draw_concave_hull(
    data, x_col, y_col, concavity, threshold, target_col, speed_col, 
    xlim=None, ylim=None, scale_to_axes=True
):
    grouped = data.groupby([target_col, speed_col])
    hulls = {}

    if xlim is None:
        xlim = (data[x_col].min(), data[x_col].max())
    if ylim is None:
        ylim = (data[y_col].min(), data[y_col].max())

    for (target, speed), group_data in grouped:
        x = group_data[x_col].to_numpy()
        y = group_data[y_col].to_numpy()

        keep = np.isfinite(x) & np.isfinite(y)
        x = x[keep]
        y = y[keep]

        if len(x) < 10:
            continue

        if scale_to_axes:
            pts = scale_to_panel(x, y, xlim, ylim)
        else:
            pts = np.column_stack([x, y])

        convex = ConvexHull(pts)
        convex_idx = convex.vertices.astype(np.int32)

        idx = concave_hull_indexes(
            pts,
            convex_hull_indexes=convex_idx,
            concavity=concavity,
            length_threshold=threshold
        )

        hull_pts = pts[idx]

        if scale_to_axes:
            hull_pts = unscale_from_panel(hull_pts, xlim, ylim)

        hulls[(target, speed)] = hull_pts

    return hulls

def plot_faceted_hulls(
    data, hulls, x_col, y_col, target_col, speed_col,
    smooth=True, smooth_points=300, smooth_factor=0.0015,
    xlim=None, ylim=None, facet_height=6.5, facet_aspect=0.82,
    save_path='../figures/hulls.svg', dpi=300
):
    targets = sorted(data[target_col].unique())
    speeds = sorted(data[speed_col].unique())

    n_rows = len(speeds)
    n_cols = len(targets)

    # Replicate seaborn sizing logic
    facet_width = facet_height * facet_aspect
    total_width = facet_width * n_cols
    total_height = facet_height * n_rows

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(total_width, total_height),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    if xlim is None:
        xlim = (data[x_col].min(), data[x_col].max())
    if ylim is None:
        ylim = (data[y_col].min(), data[y_col].max())

    for i, speed in enumerate(speeds):
        for j, target in enumerate(targets):
            ax = axes[i, j]

            # 1. Add gridlines behind the data (zorder=0)
            ax.grid(True, color='#E0E0E0', linestyle='-', linewidth=0.8, zorder=0)

            subset = data[(data[target_col] == target) & (data[speed_col] == speed)]
            ax.scatter(subset[x_col], subset[y_col], alpha=0.35, s=18, c='gray', zorder=3)

            if (target, speed) in hulls:
                hull_points_raw = hulls[(target, speed)]
                
                hull_points = hull_points_raw
                if smooth and len(hull_points_raw) >= 4:
                    hp_scaled = scale_to_panel(hull_points_raw[:, 0], hull_points_raw[:, 1], xlim, ylim)
                    
                    hp_smooth = smooth_closed_polygon(
                        hp_scaled,
                        n_interp=min(smooth_points, max(120, len(hull_points_raw) * 8)),
                        smooth=smooth_factor
                    )
                    
                    hp_smooth_unscaled = unscale_from_panel(hp_smooth, xlim, ylim)
                    
                    if len(hp_smooth_unscaled) > 10:
                        hull_points = hp_smooth_unscaled
            
                if len(hull_points) >= 3:
                    poly = Polygon(
                        hull_points,
                        closed=True,
                        fill=True,
                        alpha=0.45,
                        color='blue',
                        edgecolor='darkblue',
                        linewidth=1.0,
                        zorder=1
                    )
                    ax.add_patch(poly)
                    ax.plot(hull_points[:, 0], hull_points[:, 1], color='blue', linewidth=1.0, zorder=2)

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

            # Force X-axis ticks to increments of 25
            ax.xaxis.set_major_locator(MultipleLocator(25))
            
            # 2. Force tick values to render on all subplots despite sharex/sharey
            ax.tick_params(labelbottom=True, labelleft=True)

            # 3. Apply axis labels universally (removed boundary conditionals)
            ax.set_xlabel(f"Target: {target}")
            ax.set_ylabel(f"Speed: {speed}")


    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight') 

    return fig, axes