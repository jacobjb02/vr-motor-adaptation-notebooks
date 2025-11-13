"""
Sandbox for experimental / unused functions and code.
If any function here becomes actively used, move it
into the appropriate module (cleaning, features, plotting, etc.).
"""


### add_training_phase

               #training_1_min, training_1_max,
               #washout_1_min, washout_1_max,
               
               #training_2_min, training_2_max,
               #washout_2_min, washout_2_max

def add_phases(df,
               
               baseline_min, baselime_max,

              ):

    print(df['trial_num'][baseline_min -1])
    print(df['trial_num'][baseline_max -1])


# function to extract water current transition points
# may be used to identify phases and associated trial numbers


### CHECKING WHO IMPROVED / GOT WORSE ACROSS EXPOSURE

combined_10_means_diff = (
    combined_10_means.pivot_table(
        index=["ppid", "target_x_label", "set_order", "block", "training_status"],
        columns="section",
        values=["mean_dist", "var_dist"]
    )
)

# compute difference: last10 - first10
combined_10_means_diff["mean_dist_diff"] = combined_10_means_diff["mean_dist"]["last10"] - combined_10_means_diff["mean_dist"]["first10"]
combined_10_means_diff["var_dist_diff"]  = combined_10_means_diff["var_dist"]["last10"] - combined_10_means_diff["var_dist"]["first10"]

combined_10_means_diff = combined_10_means_diff.reset_index()

display(combined_10_means_diff)

# filter df so we only have ppid who had positive differences
combined_10_means_diff_worse = combined_10_means_diff[combined_10_means_diff['mean_dist_diff'] > 0]



# First and last 10 trials of exposure mean differences
g = sns.FacetGrid(combined_10_means_diff_worse, col='target_x_label', sharex=True)

# Individual participant data points
g.map_dataframe(sns.barplot, 
                x='mean_dist_diff', 
                y='ppid',
                hue='set_order')

# Add shading (x < 0) to indicate improvement zone
for ax in g.axes.flat:
    ax.axvspan(ax.get_xlim()[0], 0, color="gray", alpha=0.2)

# legend
plt.legend(title="Set Order")

# Save and display
g.fig.set_size_inches(12, 12) 
plt.savefig("ppid_diff.pdf", dpi=300)
plt.show()



# ANOVA libraries
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# fit ANOVA
aovrm = AnovaRM(early_late_10_means, 'mean_dist', 'ppid', within=['section','target_x_label'])
results = aovrm.fit()
print(results)



# Predicted means from the fitted model (FOLLOWING MIXED MODEL EXECUTION)
pred = fit.predict(combined_10_means)

combined_10_means['pred'] = pred

means = (combined_10_means
         .groupby(['section','target_x_label'])['pred']
         .mean()
         .reset_index())

print(means)


# view generalization data
max_tbl = (gen_exposure
           .groupby(['target_x_label','training_status','set_order'], dropna=False)['mean_dist']
           .max()
           .reset_index(name='max_mean_dist'))
max_tbl






    
               
               
