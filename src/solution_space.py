import numpy as np

target_labels = np.unique(exposure_trials['target_x_label'])


speeds = np.array(exposure_trials['launch_Speed']) #.reshape(-1,1)
angles = np.array(exposure_trials['launch_deviation'])#.reshape(-1,1)
error = np.array(exposure_trials['baseline_corrected_dist'])#.reshape(-1,1)
targets = np.array(exposure_trials['target_x_label'])#.reshape(-1,1)


for label in target_labels:

    mask = (targets == label)

    errors_masked = error[mask].ravel()

    # launch speeds per target mask
    speeds_masked = speeds[mask].ravel()
         
    # launch deviations per target mask
    angles_masked = angles[mask].ravel()

    plt.figure()

    sns.scatterplot(x=angles_masked, y=speeds_masked, hue=errors_masked)

    plt.xlim(-100,30)

    plt.title(label)
    plt.show()
    