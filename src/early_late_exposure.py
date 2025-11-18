
import numpy as np
import pandas as pd



def early_late_means(
    df, 
    error_col,
    ppid,
    phases = ['training_1', 'training_2'],
    window_size=8,
    include_status=False 
):
    # Function to extract the first and last 10 trials within a set range.
    # In the current experiment there are two 'blocks' of exposure to the
    # active water current.
    
    # min_b1: refers to the first trial of the first exposure block
    # max_b2: refers to the final trial of the second exposure block
    
    
    # keep only exposure trials from the two windows
    df_phase = df[df['phase'].isin(phases)]

    # assuming df has columns: ppid, target_x_label, block
    # (block already assigned via your trial_num → 1/2 code)
    
    ppid_block_target = (
        df_phase
        .groupby([ppid,'phase','speed_label','target_x_label'])
        .size()
        .reset_index(name='n_trials')
        .sort_values([ppid,'phase','target_x_label'])
    )
    

    # order within each (ppid, target, block) by trial order
    ordered = (df_phase
               .sort_values([ppid,'target_x_label','phase','trial_num_target'])
               .groupby([ppid,'target_x_label','phase','speed_label'], group_keys=False))

    counts = (df_phase.groupby([ppid,'target_x_label','phase','speed_label'], observed=True)
                .size())

    # take early/late trials per (ppid, target, block)
    early = ordered.head(window_size).assign(section='early')
    
    late_all  = ordered.tail(window_size).assign(section='late')
    late = late_all[~late_all.index.isin(early.index)]

    early_to_late = pd.concat([early, late], ignore_index=True)

    # store trial_num_target lists for each grouping
    trial_lists = (
        early_to_late.groupby([ppid,'target_x_label','phase','section'])
                   .agg(trial_list=('trial_num_target', list))
                   .reset_index()
    )


    # collapse to ONE row per subject x cell for ANOVA
    groupby_cols = [ppid,'section','target_x_label','set_order','phase','speed_label']
    if include_status:
        groupby_cols.append('training_status')  
        
    summary_early_to_late = (early_to_late.groupby(groupby_cols, as_index=False, observed=True)
              .agg(mean_x_delta_cm=(error_col, 'mean'),
                   var_x_delta_cm=(error_col, 'var'),   
                   n=('trial_num_target', 'size')))

    # ordered factor for section
    summary_early_to_late['section'] = pd.Categorical(
        summary_early_to_late['section'], ['early','late'], ordered=True)
    
    return summary_early_to_late







def early_to_late_means_old(
    df, 
    error_col,
    min_b1, max_b1, 
    min_b2, max_b2, 
    window_size=10,
    include_status=False 
):

        
    # Function to extract the first and last 10 trials within a set range.
    # In the current experiment there are two 'blocks' of exposure to the
    # active water current.
    
    # min_b1: refers to the first trial of the first exposure block
    # max_b2: refers to the final trial of the second exposure block

    
    # check reqiured cols
    assert 'baseline_corrected_dist' in df.columns or 'baseline_corrected_x' in df.columns, \
            "Data must contain 'baseline_corrected_dist' or 'baseline_corrected_x'. Make sure you run 'summarize_baseline'!"


    
    # keep only exposure trials from the two windows
    df_exp = df[df['trial_num'].between(min_b1, max_b1) |
                df['trial_num'].between(min_b2, max_b2)].copy()

    # label exposure block 1 vs 2
    df_exp['block'] = np.where(df_exp['trial_num'] <= max_b1, 1, 2)



    # assuming df has columns: ppid, target_x_label, block
    # (block already assigned via your trial_num → 1/2 code)
    
    ppid_block_target = (
        df_exp
        .groupby(['ppid','block','target_x_label'])
        .size()
        .reset_index(name='n_trials')
        .sort_values(['ppid','block','target_x_label'])
    )
    

    # order within each (ppid, target, block) by trial order
    ordered = (df_exp
               .sort_values(['ppid','target_x_label','block','trial_num_target'])
               .groupby(['ppid','target_x_label','block'], group_keys=False))

    counts = (df_exp.groupby(['ppid','target_x_label','block'], observed=True)
                .size())
    #print(counts)


    # take early/late trials per (ppid, target, block)
    early = ordered.head(window_size).assign(section='early')
    
    late_all  = ordered.tail(window_size).assign(section='late')
    late = late_all[~late_all.index.isin(early.index)]

    early_to_late = pd.concat([early, late], ignore_index=True)

    # store trial_num_target lists for each grouping
    trial_lists = (
        early_to_late.groupby(['ppid','target_x_label','block','section'])
                   .agg(trial_list=('trial_num_target', list))
                   .reset_index()
    )


    # collapse to ONE row per subject x cell for ANOVA
    groupby_cols = ['ppid','section','target_x_label','set_order','block']
    if include_status:
        groupby_cols.append('training_status')  
        
    summary_early_to_late = (early_to_late.groupby(groupby_cols, as_index=False, observed=True)
              .agg(mean_dist=(error_col, 'mean'),
                   var_dist=(error_col, 'var'),   
                   n=('trial_num_target', 'size')))

    # ordered factor for section
    summary_early_to_late['section'] = pd.Categorical(
        summary_early_to_late['section'], ['early','late'], ordered=True)
    
    return summary_early_to_late







