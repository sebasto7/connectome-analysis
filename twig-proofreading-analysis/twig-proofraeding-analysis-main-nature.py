# -*- coding: utf-8 -*-
"""
Twig-profreafing analysis of presynaptic inputs
Clean code for publication

@author: Sebastian Molina-Obando
"""


#Importing packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np; np.random.seed(0)
import matplotlib.pyplot as plt
import scipy.stats

#Plots settings

font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 12}
axes = {'labelsize': 16, 'titlesize': 16}
ticks = {'labelsize': 14}
legend = {'fontsize': 14}
plt.rc('font', **font)
plt.rc('axes', **axes)
plt.rc('xtick', **ticks)
plt.rc('ytick', **ticks)
cm = 1/2.54  # centimeters in inches

save_figures = True
PC_disc = 'D'
save_path = f'{PC_disc}:\Connectomics-Data\FlyWire\Pdf-plots'#P ath to the PROCESSED_DATA folder


#%%Loading data sets
#Choose path and file
PC_disc = 'D'
dataPath =  f'{PC_disc}:\Connectomics-Data\FlyWire\Excels\drive-data-sets\submission_nature'# Path to the RAW_DATA folder
fileDate = '20230718'
fileName = f'All_Tm9_neurons_input_count_ME_L_impact_twig_proofreading_{fileDate}.xlsx' # Choose data set of interest
neuron_name = 'Tm9'
filePath = os.path.join(dataPath,fileName)
df = pd.read_excel(filePath)

#Filtering out separator row (e.g.,"NEXT INPUTS") and data were backbone-proofread was also done
df = df[(df['presynaptic_ID'] != 'NEXT INPUTS') & (df['comments'] != 'backbone-proofread')].copy()

#Filtering out all counts bigger than X number
apply_threshold = False
synapse_range = 'all synapses'

if apply_threshold:
    min_threshold = 3
    max_threshold = 150
    df = df[(df['counts'] >= min_threshold) & (df['counts'] <= max_threshold)].copy()
    synapse_range = f'synapses range: {min_threshold} - {max_threshold}'  

#%% Analysis

# Some quentifications
before_stats = df[df['twig_proofreading'] == 'before'].describe()
after_stats = df[df['twig_proofreading'] == 'after'].describe()
n = len(df['column_ID'].unique())
total_syn_number_before = df[df['twig_proofreading'] == 'before']['counts'].sum() 
total_syn_number_after = df[df['twig_proofreading'] == 'after']['counts'].sum() 
absolut_syn_number_change = total_syn_number_after - total_syn_number_before
relative_syn_number_change = (total_syn_number_after - total_syn_number_before)/total_syn_number_before

# Some data in print
print(f"Number of neurons/columns being analyzed: {n}")
print(f"Number of rows: {len(df)}")
print(f"Synapse counts being considered: {synapse_range}")
print(f"Mean across columns, the absolute change in synapse number: {absolut_syn_number_change/n}")
print(f"Mean across columns, the percentage change in synapse number: {round((relative_syn_number_change *100)/n,2)} %")
print(f"Mean across columns, new segments (potential partners): {(after_stats.loc['count'][0]-before_stats.loc['count'][0])/n}")
print('---------------------------------------------')
print(f"Before twig-proofreading: \n\n {before_stats}")
print('---------------------------------------------')
print(f"After twig-proofreading: \n\n {after_stats}")


# For distributino differences
#KDE calculations
a = df[df['twig_proofreading'] == 'after']['counts']
b = df[df['twig_proofreading'] == 'before']['counts']
kdea = scipy.stats.gaussian_kde(a)
kdeb = scipy.stats.gaussian_kde(b)
grid_a = np.linspace(min(a),max(a), 10000)
grid_b = np.linspace(min(b),max(b), 10000)
grid_ab = np.linspace(min(min(a),min(b)),max(max(a),max(b)), 10000)

# Quartiles calculations
counts = df['counts'].values
counts_a = df.loc[df.twig_proofreading=='after', 'counts'].values
counts_b = df.loc[df.twig_proofreading=='before', 'counts'].values

df_pct = pd.DataFrame()
df_pct['q_after'] = np.percentile(counts_a, range(100))
df_pct['q_before'] = np.percentile(counts_b, range(100))

#%% Plotting distributions

# Potting distributions

fig, axs = plt.subplots(2, 2,figsize=(15, 15))
sns.histplot(df, x="counts", hue="twig_proofreading", binwidth=1, element="step", common_norm=False, ax=axs[0,0]) 
axs[0,0].set_title(f'{neuron_name}, step distribution, {synapse_range}')
axs[0,0].set_xlabel('number of inputs')
# another axis

sns.histplot(data=df, x="counts", hue="twig_proofreading", binwidth=1, stat='density', common_norm=False, ax=axs[0,1]);
axs[0,1].set_title(f'{neuron_name}, Density Histogram, {synapse_range}')
axs[0,1].set_xlabel('number of inputs')


# another axis
sns.kdeplot(data = df, x="counts", hue="twig_proofreading", common_norm = False, ax=axs[1,0])
axs[1,0].set_title(f'{neuron_name}, Kernel Density Function, {synapse_range}')
axs[1,0].set_xlabel('number of inputs')
# another axis
sns.histplot(
    data=df, x="counts", hue="twig_proofreading",
    hue_order=["after", "before"],
    log_scale=False, element="step", fill=False,
    cumulative=True, stat="density", common_norm=False,
    ax=axs[1,1]) 
axs[1,1].set_title(f'{neuron_name}, cumulative distribution, {synapse_range}')
axs[1,1].set_xlabel('number of inputs')

#Plot saving
if save_figures:
    figure_title = f'\{neuron_name}-twig-analysis-distributions-{synapse_range}.pdf'
    fig.savefig(save_path+figure_title)
    print('Distributions plotted ands saved')
plt.close(fig)


# Plotting difference of distributions
fig, axs = plt.subplots(2, 2,figsize=(15, 15))
sns.boxplot(data=df, x="twig_proofreading", y="counts", ax = axs[0,0]);
axs[0,0].set_title(f'{neuron_name}, Boxplot, {synapse_range}');

# another axis
axs[0,1].plot(grid_a, kdea(grid_a), label="after")
axs[0,1].plot(grid_b, kdeb(grid_b), label="before")
axs[0,1].set_title(f'{neuron_name}, KDE matplotlib, {synapse_range}')
axs[0,1].set_xlabel('number of inputs')
axs[0,1].set_ylabel('Density')
axs[0,1].legend()

# another axis
axs[1,1].plot(grid_ab, kdea(grid_ab)-kdeb(grid_ab), color ="k", label="difference")
axs[1,1].set_title(f'{neuron_name}, KDE matplotlib, {synapse_range}')
axs[1,1].set_xlabel('number of inputs')
axs[1,1].set_ylabel('Density')
axs[1,1].legend()

#another axis

axs[1,0].scatter(x='q_before', y='q_after', data=df_pct, label='percentile');
sns.lineplot(x='q_before', y='q_before', data=df_pct, color='r', label='Line of perfect fit',  ax = axs[1,0]);
axs[1,0].set_xlabel('Counts, before twig-proofreading')
axs[1,0].set_ylabel('Counts, after twig-proofreading')
axs[1,0].legend()
axs[1,0].set_title(f'{neuron_name}, Q-Q plot, {synapse_range}');

#Plot saving
if save_figures:
    figure_title = f'\{neuron_name}-twig-analysis-KDE-QQ-and-differences-{synapse_range}.pdf'
    fig.savefig(save_path+figure_title)
    print('Difference of distributions potted and saved')
plt.close(fig)

# Plotting data per column

# perform groupby
df_grouped = df.copy()
df_grouped= df_grouped.groupby(['column_ID', 'twig_proofreading']).agg(total_counts=("counts", 'sum'))
df_grouped= df_grouped.reset_index()

# perform difference
after_before_diff_df = df_grouped[df_grouped.twig_proofreading=='after'].merge(df_grouped[df_grouped.twig_proofreading=='before'],
                                on=['column_ID'],
                                suffixes = ('_after', '_before'))
count_diff = after_before_diff_df['total_counts_after']-after_before_diff_df['total_counts_before']
after_before_diff_df['count_difference'] = count_diff 
#after_before_diff_df.drop(['twig_proofreading_after', 'twig_proofreading_before'], axis=1, inplace=True)
  
# plot barplot
fig, axs = plt.subplots(2, 2,figsize=(10, 10),gridspec_kw={'width_ratios': [3, 1]}) # n: number of columns, defined at the beginning of the analysis
fig.tight_layout(pad=3.0) # Adding some space between subplots
sns.barplot(x="column_ID",
           y="total_counts",
           hue="twig_proofreading",
           data=df_grouped, ax = axs[0,0])
axs[0,0].set_title(f'{neuron_name}, Barplot, {synapse_range}')

#another axes:
sns.boxplot(data=df_grouped,y="total_counts",x ="twig_proofreading",ax = axs[0,1])

#another axes:
sns.barplot(x="column_ID", y="count_difference",data=after_before_diff_df, ax = axs[1,0], color= 'k')
axs[1,0].set_title(f'{neuron_name}, Barplot, {synapse_range}')

#another axes:
sns.boxplot(y="count_difference",data=after_before_diff_df,ax = axs[1,1], color = 'k', medianprops=dict(color="w", alpha=1))
axs[1,1].set_xlabel('all columns')

#Plot saving
if save_figures:
    figure_title = f'\{neuron_name}-twig-analysis-differences-per-column-{synapse_range}.pdf'
    fig.savefig(save_path+figure_title)
    print('Analysis per column potted and saved')
plt.close(fig)

