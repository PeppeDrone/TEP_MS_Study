# Define the directories
import os
import mne
import pdb
import pickle
from pathlib import Path
import numpy as np
from scipy import signal
import pandas as pd
# from statannot import add_stat_annotation # type: ignore
import seaborn as sns
import matplotlib.pyplot as plt
# from statannotations.Annotator import Annotator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import permutation_test
from scipy.stats import spearmanr, mannwhitneyu
from scipy.stats import linregress


#%%
# Define the intersection of channel names
intersection_channels = {
    'EEG AF3', 'EEG AF4', 'EEG C3', 'EEG C4', 'EEG Cz', 'EEG F3', 'EEG F4', 
    'EEG F7', 'EEG F8', 'EEG Fz', 'EEG Oz', 'EEG P3', 'EEG P4', 'EEG PO3', 
    'EEG PO4', 'EEG Pz', 'EEG T3', 'EEG T4', 'EEG T5', 'EEG T6'
}

# Function to load MNE files from a directory
def elaborate_files(directory, opt_ex):
    mne_files = []
    if opt_ex['OrganizeData'] == 1:
        for file_name in os.listdir(directory):
            if file_name.endswith('.EDF'):  
                # 1. Load data 
                file_path = os.path.join(directory, file_name)
                raw = mne.io.read_raw_edf(file_path, preload=True)
                print(f"Original Channels: {raw.info.ch_names}")
                # 2. Set channels
                try:
                    channels_to_keep = [ch for ch in raw.info.ch_names if ch in intersection_channels]
                    
                    if channels_to_keep:
                        raw.pick_channels(channels_to_keep)  # Keep only these channels
                    
                    # Rename channels by removing 'EEG ' prefix
                    channel_rename_dict = {ch: ch.split(' ')[1] for ch in channels_to_keep}
                    raw.rename_channels(channel_rename_dict)

                    print(f"Retained Channels: {raw.info.ch_names}")

                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")
                    continue  # Skip this file if an error occurs
                
                # Define the output directory
                output_dir = 'C:/Users/PLIUZZIADM/Desktop/AI-EEG-Embedding/TMSEEG'
                os.makedirs(output_dir, exist_ok=True)


                # Create a folder for each patient based on the session name
                session_name = 'SESSION ' + file_name.split('.')[0]
                patient_output_dir = os.path.join(output_dir, session_name)
                os.makedirs(patient_output_dir, exist_ok=True)


                # # Save the EEG data to the patient's session folder
                output_file_path = os.path.join(patient_output_dir, file_name.replace('.EDF', '-processed.edf'))
                raw.export(output_file_path, fmt = 'edf', overwrite=True)
    return 'Success'


def find_peak_in_window(times, gfp, t_min, t_max, polarity='positive', prominence=0.1):
    """Find peak in a specific time window."""
    # Get data in time window
    mask = (times >= t_min) & (times <= t_max)
    window_gfp = gfp[mask]
    window_times = times[mask]
    
    # Invert signal for negative peaks
    if polarity == 'negative':
        window_gfp = -window_gfp
    
    # Find peaks
    peaks, properties = signal.find_peaks(window_gfp, prominence=prominence)
    
    if len(peaks) > 0:
        # Get highest prominence peak
        best_peak = peaks[np.argmax(properties['prominences'])]
        peak_time = window_times[best_peak]
        peak_amplitude = window_gfp[best_peak] * (-1 if polarity == 'negative' else 1)
        return peak_time, peak_amplitude
    return None, None

def analyze_tep_components(opt_ex, evoked: mne.Evoked, prominence: float = 0.1, flag_who = 1):
    """Analyze TEP components using GFP and find peaks."""
    # Get data type
    ch_types = list(set(evoked.get_channel_types()))
    data_type = 'csd' if 'csd' in ch_types else 'eeg'
    
    # Calculate GFP
    times = evoked.times * 1000  # Convert to ms
    data = evoked.get_data()
    gfp = np.std(data, axis=0)
    
    if flag_who == 1:
        # Find components
        components = {}
        for name, criteria in opt_ex['TEP_COMPONENTS'].items():
            t_min, t_max = criteria['time']
            peak_time, peak_amplitude = find_peak_in_window(
                times, gfp, t_min, t_max, 
                criteria['polarity'], 
                prominence=prominence * np.max(gfp)
            )        
            if peak_time is not None:
                components[name] = {
                    'time': peak_time,
                    'amplitude': peak_amplitude,
                    'data_type': data_type
                }
        
        return components, gfp, times
    
    if flag_who == 'F3':
        sig = data[3,:] #F3
        # Find components
        components = {}
        for name, criteria in opt_ex['TEP_COMPONENTS'].items():
            t_min, t_max = criteria['time']
            peak_time, peak_amplitude = find_peak_in_window(
                times, sig, t_min, t_max, 
                criteria['polarity'], 
                prominence=prominence * np.max(sig)
            )        
            if peak_time is not None:
                components[name] = {
                    'time': peak_time,
                    'amplitude': peak_amplitude,
                    'data_type': data_type
                }
        
        return components, gfp, times
    
    if flag_who == 'C3':
        sig = data[8,:] #F3
        # Find components
        components = {}
        for name, criteria in opt_ex['TEP_COMPONENTS'].items():
            t_min, t_max = criteria['time']
            peak_time, peak_amplitude = find_peak_in_window(
                times, sig, t_min, t_max, 
                criteria['polarity'], 
                prominence=prominence * np.max(sig)
            )        
            if peak_time is not None:
                components[name] = {
                    'time': peak_time,
                    'amplitude': peak_amplitude,
                    'data_type': data_type
                }
        
        return components, gfp, times


def compute_metrics(dir, dir2, opt_ex):
    if opt_ex['ComputeMetrics']:
        _gfp, _evoked = [], []
        metrics = {'SubId':[],
                   'GFP peak':[],
                   'GFP latency':[],
                   'PCI':[],
                   'nPCIst':[],
                   'N15':[],'P30':[],'N45':[],'P60':[],'N100':[],'P180':[],'N280':[],
                   'F3_N15':[],'F3_P30':[],'F3_N45':[],'F3_P60':[],'F3_N100':[],'F3_P180':[],'F3_N280':[],
                   'C3_N15':[],'C3_P30':[],'C3_N45':[],'C3_P60':[],'C3_N100':[],'C3_P180':[],'C3_N280':[],
                   }
        # pdb.set_trace()
        for file_name in os.listdir(dir):
            if file_name.endswith('.fif'):
                source_name = file_name.split('-')[0]
                metrics['SubId'].append(source_name)
                file_path = os.path.join(dir, file_name)
                epochs = mne.read_epochs(file_path, preload=True)          
                epochs.crop(tmin=epochs.times[0], tmax=epochs.times[579])
                evoked = epochs.average()
                _evoked.append(evoked)
                # print(file_name.split('-')[0])
                # pdb.set_trace()
       
                components, gfp, times = analyze_tep_components(opt_ex, evoked, 0.01, flag_who = 1)
                for component_name in opt_ex['TEP_COMPONENTS'].keys():
                    if component_name in components:
                        metrics[component_name].append(1E3*components[component_name]['amplitude'])
                    else:
                        metrics[component_name].append(np.nan)
                        
                components, gfp, times = analyze_tep_components(opt_ex, evoked, 0.01, flag_who = 'F3')
                for component_name in opt_ex['TEP_COMPONENTS'].keys():
                    if component_name in components:
                        metrics['F3_'+component_name].append(1E3*components[component_name]['amplitude'])
                    else:
                        metrics['F3_'+component_name].append(np.nan)
                        
                components, gfp, times = analyze_tep_components(opt_ex, evoked, 0.01, flag_who = 'C3')
                for component_name in opt_ex['TEP_COMPONENTS'].keys():
                    if component_name in components:
                        metrics['C3_'+component_name].append(1E3*components[component_name]['amplitude'])
                    else:
                        metrics['C3_'+component_name].append(np.nan)
                        
                        
                metrics['GFP peak'].append(np.max(gfp))
                metrics['GFP latency'].append(evoked.times[np.argmax(gfp)])
                _gfp.append(gfp)
                par = {'baseline_window':(-400,-50), 'response_window':(0,300), 'k':1.2, 'min_snr':1.1,
                       'max_var':99, 'embed':False,'n_steps':50}
                pci, npci = computePCI(epochs, par)
        
                metrics['PCI'].append(pci)
                metrics['nPCIst'].append(npci)


        pdb.set_trace()
        pd.DataFrame.from_dict(metrics).to_excel('MetricsSubjects.xlsx')
        
        gfp_output_path = os.path.join(dir2, "gfp.npy")
        np.save(gfp_output_path, gfp)

        evoked_output_path = os.path.join(dir2, "evoked.pkl")
        with open(evoked_output_path, 'wb') as f:
            pickle.dump(_evoked, f)

        return pd.DataFrame.from_dict(metrics), _gfp, _evoked
    else:
        return 0, 0, 0 
    
def plot_results(metrics, gfp, evoked, opt_ex):
    # Group patients by the ones which have an 's' in the SubId at the beginning or not
    metrics['Group'] = metrics['SubId'].apply(lambda x: 'Controls' if x.startswith('S') else 'Patients')

    # Evoked
    healthy_evoked = [evoked for evoked, sub_id in zip(evoked, metrics['SubId']) if sub_id.startswith('S')]
    ms_evoked = [evoked for evoked, sub_id in zip(evoked, metrics['SubId']) if not sub_id.startswith('S')]
    avg_healthy_evoked = mne.grand_average(healthy_evoked)
    avg_ms_evoked = mne.grand_average(ms_evoked)
    avg_healthy_evoked.plot_joint(times = [0.015, 0.030, 0.100, 0.180, 0.280],
                                  ts_args = {'spatial_colors':False, 'xlim':[-0.1,0.4], 'ylim':dict(csd=[-1.8,1])},
                                  topomap_args = {'vlim':[-1,1]})
    ax = plt.gca()
    ax.set_ylabel('CSD [mV/$m^2$]')
    ax.set_xlabel('Time [s]')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    if hasattr(ax, 'lines'):  # Look for time series subplot
        for line, ch_name in zip(ax.lines, avg_healthy_evoked.ch_names):
            if ch_name == 'C3':  
                line.set_color('red')  # Set F3 trace to green
                line.set_linewidth(1.5)  # Make it bold
            if ch_name == 'F3':  
                line.set_color('green')  # Set F3 trace to green
                line.set_linewidth(1.5)  # Make it bold
    plt.gcf().savefig('Evoked_healthy.svg')
    plt.show()
    
    
    
    avg_ms_evoked.plot_joint(times = [0.015, 0.030, 0.100, 0.180, 0.280],
                                  ts_args = {'spatial_colors':False, 'xlim':[-0.1,0.4], 'ylim':dict(csd=[-1.8,1])},
                                  topomap_args = {'vlim':[-1,1]})
    ax = plt.gca()
    ax.set_ylabel('CSD [mV/$m^2$]')
    ax.set_xlabel('Time [s]')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    if hasattr(ax, 'lines'):  # Look for time series subplot
        for line, ch_name in zip(ax.lines, avg_healthy_evoked.ch_names):
            if ch_name == 'C3':  
                line.set_color('red')  # Set F3 trace to green
                line.set_linewidth(1.5)  # Make it bold
            if ch_name == 'F3':  
                line.set_color('green')  # Set F3 trace to green
                line.set_linewidth(1.5)  # Make it bold
    plt.gcf().savefig('Evoked_ms.svg')
    plt.show()

    # GFP
    healthy_gfp = np.array([gfp[i] for i, sub_id in enumerate(metrics['SubId']) if sub_id.startswith('S')])
    ms_gfp = np.array([gfp[i] for i, sub_id in enumerate(metrics['SubId']) if not sub_id.startswith('S')])
        
    # Define test statistic: Difference in mean GFP
    def test_statistic(x, y):
        return np.mean(x) - np.mean(y)
    
    # Perform permutation test
    res = permutation_test((healthy_gfp, ms_gfp),
                           statistic=test_statistic,
                           permutation_type='independent',
                           n_resamples=1000,
                           alternative='two-sided',
                           random_state=42)

    print('p= ' + str(np.mean(res.pvalue)))
    print('ST= ' + str(np.mean(res.statistic)))




    mean_healthy_gfp = np.mean(healthy_gfp, axis=0)
    std_healthy_gfp = np.std(healthy_gfp, axis=0)
    mean_ms_gfp = np.mean(ms_gfp, axis=0)
    std_ms_gfp = np.std(ms_gfp, axis=0)
    times = evoked[0].times 

    # Plot GFP for both groups on the same figure
    fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.plot(times, mean_healthy_gfp, label='Controls', color='green')
    ax.fill_between(times, mean_healthy_gfp - std_healthy_gfp, mean_healthy_gfp + std_healthy_gfp, 
                    color='green', alpha=0.4)
    
    ax.plot(times, mean_ms_gfp, label='Patients', color='red', linewidth = 1.5)
    ax.fill_between(times, mean_ms_gfp - std_ms_gfp, mean_ms_gfp + std_ms_gfp, 
                    color='red', alpha=0.2)

    # Styling the axes
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlim(-0.1,0.4)
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('GFP [mV/$m^2$]', fontsize=12)
    ax.legend(fontsize = 14)
    legend = ax.get_legend()  # Access the first legend
    legend.get_texts()[0].set_text("Healthy")  # Change "Controls" to "Healthy"
    legend.get_texts()[1].set_text("MS")  
    plt.tight_layout()
    plt.savefig('GFP_comparison.svg', bbox_inches='tight')
    plt.show()
    

    # GFP peaks, latency, and area under the curve (integral)
    fig, axes = plt.subplots(1, 3, figsize=(9,4))  # Adjusted for three metrics

    # Compute the area under the GFP curve using the trapezoidal rule
    metrics['GFP integral'] = [np.trapz(gfp[i], evoked[i].times) for i in range(len(gfp))]

    # GFP Peak
    sns.boxplot(x='Group', y='GFP peak', data=metrics, ax=axes[0], showfliers=False, palette = {'Controls':'Green','Patients':'Red'})
    axes[0].set_ylabel('GFP Peak [mV/$m^2$]')
    # annotator = Annotator(axes[0], [("Controls", "Patients")], data=metrics, x='Group', y='GFP peak', test='Mann-Whitney')
    # annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
    # annotator.apply_and_annotate()
    # print(annotator)
    # GFP Latency
    sns.boxplot(x='Group', y='GFP latency', data=metrics, ax=axes[1], showfliers=False, palette = {'Controls':'Green','Patients':'Red'})
    axes[1].set_ylabel('GFP Latency [s]')
    # annotator = Annotator(axes[1], [("Controls", "Patients")], data=metrics, x='Group', y='GFP latency', test='Mann-Whitney')
    # annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
    # annotator.apply_and_annotate()
    # print(annotator)

    # GFP Integral (Area under the curve)
    sns.boxplot(x='Group', y='GFP integral', data=metrics, ax=axes[2], showfliers=False, palette = {'Controls':'Green','Patients':'Red'})
    axes[2].set_ylabel(r"GFP$_{\int}$ [mV$\cdot$ s/$m^2$]")
    # annotator = Annotator(axes[2], [("Controls", "Patients")], data=metrics, x='Group', y='GFP integral', test='Mann-Whitney')
    # annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
    # annotator.apply_and_annotate()
    # print(annotator)

    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.set_xticklabels(['MS','Healthy'])
        ax.set_xlabel('')
        ax.legend(fontsize = 12)
    
    plt.tight_layout()
    fig.savefig('GFP_peaks_latencies_integral.svg', bbox_inches = 'tight')
    plt.show()
    
    
    # Generate boxplots for N15, P30, N100, and P180 components
    fig, axes = plt.subplots(1,5, figsize=(11,4), sharey = True)
    axes = axes.flatten()
    components_to_plot = ['N15', 'P30', 'N100', 'P180','N280']
    palette = {'Controls': 'Green', 'Patients': 'Red'}
    
    for ax, component in zip(axes, components_to_plot):
        sns.boxplot(x='Group', y=component, data=metrics, ax=ax, showfliers=False, palette=palette)
        ax.set_xlabel('')
        ax.set_title(f'{component}', fontsize = 12)
        
        # annotator = Annotator(ax, [("Controls", "Patients")], data=metrics, x='Group', y=component, test='Mann-Whitney')
        # annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
        # annotator.apply_and_annotate()
        # print(annotator)
        # Styling
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.set_xticklabels(['MS','Healthy'], fontsize = 12)
        ax.set_xlabel('')
        if component == 'N15':
            ax.set_ylabel(r"Absolute Amplitude [mV/$m^2$]", fontsize = 12)
        else:
            ax.set_ylabel("")
    plt.tight_layout()
    plt.show()
    fig.savefig('TEP_latencies.svg', bbox_inches = 'tight')
    # Count NaNs per component, split by group
    components_to_check = ['N15', 'P30', 'N100', 'P180', 'N280']
    nan_counts = {}
    for component in components_to_check:
        nan_counts[component] = metrics.groupby('Group')[component].apply(lambda x: x.isna().sum())
    nan_counts_df = pd.DataFrame(nan_counts)
    print(nan_counts_df)

    # methods = ['GFP', 'F3', 'C3']  # Assuming these are the three peak computation methods
    # components_to_plot = ['N15', 'P30', 'N100', 'P180', 'N280']
    # palette = {'Controls': 'Green', 'Patients': 'Red'}
    
    # fig, axes = plt.subplots(3, 5, figsize=(15, 12))  # 3 rows for methods, 5 columns for components
    
    # for row, method in enumerate(methods):
    #     for col, component in enumerate(components_to_plot):
    #         ax = axes[row, col]
    #         col_name = f'{method}_{component}'
            
    #         if col_name in metrics.columns:
    #             sns.boxplot(x='Group', y=col_name, data=metrics, ax=ax, showfliers=False, palette=palette)
    #             ax.set_xlabel('Group')
    #             ax.set_title(f'{method} {component} Comparison')
                
    #             annotator = Annotator(ax, [("Controls", "Patients")], data=metrics, x='Group', y=col_name, test='Mann-Whitney')
    #             annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
    #             annotator.apply_and_annotate()
                
    #             ax.spines['right'].set_visible(False)
    #             ax.spines['top'].set_visible(False)
    #             ax.spines['left'].set_linewidth(1.5)
    #             ax.spines['bottom'].set_linewidth(1.5)
                
    #             if component == 'PCI':
    #                 ax.set_ylabel(f'{component} [#]')
    #             else:
    #                 ax.set_ylabel(f'{component} Amplitude [V]')
    #         else:
    #             ax.set_visible(False)  # Hide the plot if the column is missing
    
    # plt.tight_layout()
    # plt.show()
    # fig.savefig('TEP_latencies_PCI.svg', bbox_inches='tight')
    
    # # Generate boxplots for N15, P30, N100, and P180 components
    # fig, axes = plt.subplots(1,6, figsize=(11,4))
    # axes = axes.flatten()
    # components_to_plot = ['N15', 'P30', 'N100', 'P180','N280']
    # palette = {'Controls': 'Green', 'Patients': 'Red'}
    
    # for ax, component in zip(axes, components_to_plot):
    #     sns.boxplot(x='Group', y=component, data=metrics, ax=ax, showfliers=False, palette=palette)
    #     ax.set_xlabel('Group')
    #     ax.set_title(f'{component} Comparison')
        
    #     annotator = Annotator(ax, [("Controls", "Patients")], data=metrics, x='Group', y=component, test='Mann-Whitney')
    #     annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
    #     annotator.apply_and_annotate()
    #     # Styling
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['left'].set_linewidth(1.5)
    #     ax.spines['bottom'].set_linewidth(1.5)
    #     if component == 'PCI':
    #         ax.set_ylabel(f'{component} [#]')
    #     else:
    #         ax.set_ylabel(f'{component} Amplitude [V]')
    # plt.tight_layout()
    # plt.show()
    
    # fig.savefig('TEP_latencies_PCI.svg', bbox_inches = 'tight')
    
        
    # # Load or preprocess your evoked data
    # evoked = avg_healthy_evoked  # Assuming this is your evoked dataset
    
    # # Get channel positions
    # pos = np.array([evoked.info['chs'][evoked.ch_names.index(ch)]['loc'][:2] for ch in evoked.ch_names])
    
    # # Normalize positions for plotting
    # x, y = pos[:, 0], pos[:, 1]
    # x = (x - x.min()) / (x.max() - x.min())  # Normalize between 0 and 1
    # y = (y - y.min()) / (y.max() - y.min())  # Normalize between 0 and 1
    # y_mean = y.mean()
    # y = np.where(y > y_mean, y - 0.12, y + 0.12) 
    # x_mean = x.mean()
    # x = np.where(x > x_mean, x - 0.1, x + 0.1)

    # # Create figure
    # fig = plt.figure(figsize=(18, 15))  # Adjusted figure size
    
    # # Plot each channel at its scalp location
    # for i, ch_name in enumerate(evoked.ch_names):

    #     # Extract channel data for permutation test
    #     # healthy_evoked.times()
    #     healthy_data = np.array([e.get_data(picks=[ch_name]) for e in healthy_evoked]).squeeze()
    #     ms_data = np.array([e.get_data(picks=[ch_name]) for e in ms_evoked]).squeeze()
    #     # Compute permutation test
    #     def test_statistic(x, y):
    #         return np.mean(x) - np.mean(y)  # Difference in means
        
        
    #     # Define the desired time window (0 to 0.4 seconds)
    #     time_window = (0.01, 0.200)
        
    #     # Get indices for the specified time window
    #     time_indices = np.where((evoked.times >= time_window[0]) & (evoked.times <= time_window[1]))[0]
        
    #     # Extract GFP data for the selected time window
    #     healthy_gfp_subset = healthy_data[:, time_indices]
    #     ms_gfp_subset = ms_data[:, time_indices]
    #     res = permutation_test((healthy_gfp_subset, ms_gfp_subset),
    #                            test_statistic, permutation_type='independent', n_resamples=100)        
    #     # Print test statistics and p-value
    #     print(f"Channel: {ch_name} | Test Statistic: {np.mean(res.statistic):.5f} | p-value: {np.mean(res.pvalue):.5f}")
        
    #     ax = fig.add_axes([x[i], y[i], 0.08, 0.08])  # Adjusted positions
    #     avg_healthy_evoked.plot(
    #         picks=[ch_name], 
    #         axes=ax, 
    #         show=False, 
    #         hline=[0],  # Zero-line reference
    #         spatial_colors=False
    #     )
        
    #     avg_ms_evoked.plot(
    #         picks=[ch_name], 
    #         axes=ax, 
    #         show=False, 
    #         hline=[0],  # Zero-line reference
    #         spatial_colors=True
    #     )
    #     lines = ax.get_lines()  # Get all plotted lines
    #     if len(lines) > 1:
    #         plt.setp(lines[0], color='Green', linewidth=1.5)  # First line (Healthy)
    #         plt.setp(lines[2], color='Red', linewidth=1.5)  # Second line (MS)
    #     # Remove "Nave" label if it appears
    #     if len(ax.texts) > 1:
    #         ax.texts[1].set_visible(False)
    #         ax.texts[3].set_visible(False)

    #     ax.set_title(ch_name, fontsize=10)
    #     ax.axvline(0, color='k', linestyle='--', linewidth=1)  # Stimulus onset
    #     ax.set_xlim(-0.1, 0.4)  # Time range
    #     ax.set_ylim(-1, 1)  # Amplitude limits
    #     ax.axis('off')  # Hide axis for cleaner display
    
    # # Global title
    # plt.show()
    # plt.tight_layout()
    # fig.savefig('Compared_Topotep.svg')
    
    # plt.close('all')
    
    
    # Generate boxplots for N15, P30, N100, and P180 components
    fig, ax = plt.subplots(1,1, figsize=(2,4))
    components_to_plot = ['PCI']
    palette = {'Controls': 'Green', 'Patients': 'Red'}
    
    sns.boxplot(x='Group', y='PCI', data=metrics, ax=ax, showfliers=False, palette=palette)
    
    # annotator = Annotator(ax, [("Controls", "Patients")], data=metrics, x='Group', y='PCI', test='Mann-Whitney')
    # annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
    # annotator.apply_and_annotate()
    # Styling
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    ax.set_ylabel('PCI', fontsize = 12)
    plt.tight_layout()
    plt.show()
    ax.set_xticklabels(['Healthy','MS'], fontsize = 12)
    ax.set_xlabel('')
    fig.savefig('PCI.svg', bbox_inches = 'tight')
    plt.show()
    
    
    # Split PCI into two groups
    group_controls = metrics[metrics['Group'] == 'Controls']['PCI']
    group_patients = metrics[metrics['Group'] == 'Patients']['PCI']
    
    # Perform Mann-Whitney U test
    u_statistic, p_value = mannwhitneyu(group_controls, group_patients, alternative='two-sided')
    
    # Calculate effect size (rank-biserial correlation)
    n1 = len(group_controls)
    n2 = len(group_patients)
    
    rank_biserial_r = 1 - (2 * u_statistic) / (n1 * n2)

    
def plot_correlation_and_scatter(df, selected_columns=None, alpha=0.05, add='_'):
    """Plots a correlation heatmap and scatterplots with regression lines for PCI vs other variables.
    
    Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        selected_columns (list, optional): List of column names to include in the correlation matrix and scatterplots.
        alpha (float): Significance level for masking non-significant correlations.
    """
    
    # If specific columns are selected, filter the DataFrame
    if selected_columns:
        df = df[selected_columns]  

    numerical_df = df.select_dtypes(include=['number'])
    cols = numerical_df.columns
    num_vars = len(cols)

    # Compute correlation matrix and p-values
    corr_matrix = np.zeros((num_vars, num_vars))
    p_values = np.ones((num_vars, num_vars))

    for i in range(num_vars):
        for j in range(num_vars):
            if i != j:
                # Drop NaNs for this pair of variables
                valid_idx = numerical_df.iloc[:, [i, j]].dropna()
                if len(valid_idx) > 1:  # Ensure at least two valid data points
                    corr_matrix[i, j], p_values[i, j] = spearmanr(valid_idx.iloc[:, 0], valid_idx.iloc[:, 1])
                else:
                    corr_matrix[i, j], p_values[i, j] = np.nan, np.nan  # Not enough data
            else:
                corr_matrix[i, j] = 1  # Correlation of a variable with itself is 1

    # Convert to DataFrame
    corr_matrix = pd.DataFrame(corr_matrix, index=cols, columns=cols)
    p_values = pd.DataFrame(p_values, index=cols, columns=cols)
    pdb.set_trace()
    # Mask non-significant correlations
    mask = p_values >= alpha
    corr_matrix_masked = corr_matrix.mask(mask)

    # Extract R² values for PCI vs other variables
    r2_values = corr_matrix.loc["PCI"]## ** 2  # Squaring Spearman's R to get R²
    
    # --- Figure 1: Correlation Heatmap ---
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix_masked, annot=True, cmap='coolwarm', 
                fmt='.2f', linewidths=0.8, mask=mask,
                annot_kws={"size": 24}, ax=ax)
    
    ax.set_aspect("equal")  # Square aspect ratio
    plt.xticks(fontsize=24, rotation = 90)
    plt.yticks(fontsize=24, rotation = 0)

    plt.tight_layout()
    plt.savefig(add+"Correlation.svg", format="svg")
    plt.show()

    # --- Figure 2: Scatterplots with Regression (PCI vs Significant Variables) ---
    if "PCI" not in df.columns:
        raise ValueError("The column 'PCI' is not present in the dataframe.")
    
    # Select only significant variables (p < alpha)
    significant_vars = [col for col in selected_columns if col != "PCI" and p_values.loc["PCI", col] < alpha]
    num_significant_vars = len(significant_vars)
    
    if num_significant_vars > 0:
        fig, axes = plt.subplots(nrows=1, ncols=num_significant_vars, figsize=(3 * num_significant_vars, 3), sharey=True)
        if num_significant_vars == 1:
            axes = [axes]  # Ensure axes is iterable for a single subplot

        for ax, col in zip(axes, significant_vars):
            x = df[col]
            y = df["PCI"]
            
            # Drop NaNs before regression
            valid_data = df[[col, "PCI"]].dropna()
            if len(valid_data) > 1:
                slope, intercept, _, _, _ = linregress(valid_data[col], valid_data["PCI"])
                
                # Use Spearman's R² from heatmap
                # r_squared = r2_values[col] if not np.isnan(r2_values[col]) else 0  # Default to 0 if NaN

                ax.scatter(valid_data[col], valid_data["PCI"], alpha=0.6, s = 50, label=f"R = {r2_values[col]:.2f}")
                
                # Regression line
                x_vals = np.linspace(valid_data[col].min(), valid_data[col].max(), 100)
                y_vals = slope * x_vals + intercept
                ax.plot(x_vals, y_vals, color='red', linestyle='dashed', linewidth = 1.4)

            # Formatting
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel("PCI", fontsize=12)
            ax.legend()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            if col == 'L_per_lat' or col == 'R_per_lat':
                ax.set_xlim(20,36)
        plt.tight_layout()
        plt.savefig(add+"ScatterPlots_PCI.svg", format="svg")
        plt.show()
        
    return fig

def merge_patient_data():
    # Load the datasets
    metrics_df = pd.read_excel(os.path.join(data_elab_dir, 'MetricsTEP.xlsx'))
    clinical_df = pd.read_excel(os.path.join(data_elab_dir, 'MetricsClinical.xlsx'))
    # Filter out only patients (SubId not starting with 'S')
    metrics_df = metrics_df[~metrics_df['SubId'].astype(str).str.startswith('S')]
    # Convert SubId to match the format of ID in clinical data
    metrics_df['SubId'] = metrics_df['SubId'].astype(int).astype(str).str.zfill(3)
    metrics_df['SubId'] = 'SM' + metrics_df['SubId']
    # Merge on ID column from clinical data and SubId from metrics
    merged_df = metrics_df.merge(clinical_df, left_on='SubId', right_on='ID', how='inner')


    selected_columns = ['PCI','CMI_cogn_single_task_esatte', 'CMI_cogn_single_task_totali',
                        'CMI_mot_Over_single _task_nprova', 'CMI_mot_Over_single _task_metri',
                        'CMI_Over_dual_task_nprova', 'CMI_Over_dual_task_esatte',
                        'CMI_Over_dual_task_totali', 'CMI_Over_dual_task_metri',
                        'CMI_mot_CMill_single _task_ nprova', 'CMI_Cmill_dual_task_esatte',
                        'CMI_Cmill_dual_task_totali']
    plot_correlation_and_scatter(merged_df, selected_columns, add = 'CMI')
    
    selected_columns = ['PCI','6mWT_overground_min1', '6mWT_overground_min2',
                        '6mWT_overground_min3', '6mWT_overground_min4',
                        '6mWT_overground_min5', '6mWT_overground_min6',
                        '6mWT_overground_totmetri',
                        'DTC_motor_overground', 'DTC_cognitive_overground', 'DTC_cognitive_overground_2']
    plot_correlation_and_scatter(merged_df, selected_columns, add = '6MWT_DTC')

    selected_columns = ['PCI','Area_ST_over', 'Ant/Post_ST_over', 'Sin/Dx_ST_over',
                        'Passo_ST_over', 'Passo_CV_ST_over', 'Passo _norm_ST_over',
                        'Stance Phase_ST_over', 'perc_Stance Phase_ST_over', 'Swing_Phase_ST_over',
                        'perc_Swing_Phase_ST_over', 'Single_Sup_ST_over', 'perc_Single_Sup_ST_over',
                        'Double_Sup_ST_over', 'perc_Double_Sup_ST_over', 'Tempi_PassoST_over',
                        'Load_Response_ST_over', 'perc_Load_Response_ST_over', 'Pre-Swing_ST_over',
                        'perc_PreSwing_ST_over', 'Gait_cycle_Stover', 'Falcata_Stover', 'Velocità_STover',
                        'Cadence_Stover', 'Contact_phase_Stover', 'perc_Contact_phase_STover',
                        'Foot_ flatST_over', 'perc_Foot_ flat_ST_over']
    plot_correlation_and_scatter(merged_df, selected_columns, add = 'gait1')

    selected_columns =  ['PCI','Propulsive_phase_ST_over', 'perc_Propulsive_phase_ST_over',
                          'WALK_RATIO_ST', 'Area_DT_over', 'Ant/Post_DT_over', 'Sin/Dx_DT_over',
                          'Passo_DT_over', 'Passo_CV_DT_over', 'Passo _norm_DT_over', 'Stance Phase_DT_over',
                          'perc_Stance Phase_DT_over', 'Swing_Phase_DT_over', 'perc_Swing_Phase_DT_over',
                          'Single_Sup_DT_over', 'perc_Single_Sup_DT_over', 'Double_Sup_DT_over',
                          'perc_Double_Sup_DT_over', 'Tempi_Passo_DT_over', 'Load_Response_DT_over',
                          'perc_Load_Response_DT_over', 'Pre-Swing_DT_over', 'perc_PreSwing_DT_over',
                          'Gait_cycle_DT_over', 'Falcata_DT_over', 'Velocità_DT_over', 'Cadence_DT_over',
                          'Contact_phase_DT_over', 'perc_Contact_phase_DT_over', 'Foot_ flat_DT_over',
                          'perc_Foot_ flat_DT_over']
    plot_correlation_and_scatter(merged_df, selected_columns, add = 'gait2')
    
    selected_columns = ['PCI','Propulsive_phase_DT_over', 'perc_Propulsive_phase_DT_over',
                        'WALK_RATIO_DT_over', 'DTCpasso', 'DTCdoubsupp', 'DTCfalcata',
                        'DTCvelocita', 'DTCcadenza', 'Area_ST_over_6MWT', 'Ant/Post_ST_over_6MWT', 
                        'Sin/Dx_ST_over_6MWT', 'Passo_ST_over_6MWT', 'Passo _norm_ST_over_6MWT', 
                        'Stance Phase_ST_over_6MWT', 'perc_Stance Phase_ST_over_6MWT', 'Swing_Phase_ST_over_6MWT', 
                        'perc_Swing_Phase_ST_over_6MWT', 'Single_Sup_ST_over_6MWT', 'perc_Single_Sup_ST_over_6MWT']
    plot_correlation_and_scatter(merged_df, selected_columns, add = 'gait3')
    
    selected_columns = ['PCI' , 'Double_Sup_ST_over_6MWT', 'perc_Double_Sup_ST_over_6MWT', 'Tempi_PassoST_over_6MWT',
                        'Load_Response_ST_over_6MWT', 'perc_Load_Response_ST_over_6MWT', 'Pre-Swing_ST_over_6MWT', 
                        'perc_PreSwing_ST_over_6MWT', 'Gait_cycle_Stover_6MWT', 'Falcata_Stover_6MWT', 
                        'Velocità_STover_6MWT', 'Cadence_Stover_6MWT', 'Contact_phase_Stover_6MWT',
                        'perc_Contact_phase_STover_6MWT', 'Foot_ flatST_over_6MWT', 'perc_Foot_ flat_ST_over_6MWT',
                        'Propulsive_phase_ST_over_6MWT', 'perc_Propulsive_phase_ST_over_6MWT']
    plot_correlation_and_scatter(merged_df, selected_columns, add = 'gait4')

    selected_columns = ['PCI','ICV','BV','GMV','WMV','T2LV','T1LV','TV','HP']  
    fig = plot_correlation_and_scatter(merged_df, selected_columns, add = 'MRI')

    selected_columns = ['PCI', 'Age', 'EDSS', 'BDI-II', 'MFIS',
                        '9HPT (dominant)', '9HPT (not dominant)']
    fig = plot_correlation_and_scatter(merged_df, selected_columns, add = 'Clinical')

    selected_columns = ['PCI','Latency (L)',	'Amplitude (L)',
                        'CMAP/MEP (L)','CMCT (L)','Latency (R)', 	'Amplitude (R)' ,	'CMAP/MEP (R) ',	'CMCT (R)']
    fig = plot_correlation_and_scatter(merged_df, selected_columns, add = 'Mep')


    # Save the merged dataframe
    merged_df.to_excel(os.path.join(data_elab_dir, 'MetricsMerged.xlsx'), index=False)
    
    return "Merged file saved successfully!"

#%% 

runfile(os.path.dirname(__file__) + '\support_code.py',
        wdir=os.path.dirname(__file__))
raw_dir = os.path.join(os.path.dirname(__file__), '..', 'Raw')
data_dir = os.path.join(os.path.dirname(__file__), '..', 'Data')
data_elab_dir = os.path.join(os.path.dirname(__file__), '..', 'Results')
opt_ex = {'OrganizeData':0,
          'ComputeMetrics': 0,
          'TEP_COMPONENTS' : {'N15': {'time': (10, 40), 'polarity': 'negative'},
                              'P30': {'time': (20, 50), 'polarity': 'negative'},
                              'N45': {'time': (40, 50), 'polarity': 'negative'},
                              'P60': {'time': (55, 70), 'polarity': 'positive'},
                              'N100': {'time': (85, 140), 'polarity': 'negative'},
                              'P180': {'time': (150, 250), 'polarity': 'positive'},
                              'N280':{'time':(250,350),'polarity':'negative'}}
          }


# Prepare for TMSEEGPY
if opt_ex['OrganizeData'] == 1:
    message = elaborate_files(data_dir, opt_ex)
    ## Run here TMSEEGPY GUI and execute pre-processing

# # Compute TEP metrics
if opt_ex['ComputeMetrics'] == 1:
    # Compute metrics
    metrics, gfp, evoked = compute_metrics(data_dir, data_elab_dir, opt_ex)
    # Save metrics  
    metrics.to_excel(os.path.join(data_elab_dir, 'MetricsTEP.xlsx'), index=False)
    # Save gfp  
    np.save(os.path.join(data_elab_dir, 'GFP.npy'), gfp)
    # Save evoked
    np.save(os.path.join(data_elab_dir, 'Evoked.npy'), evoked)
else:
    # Load metrics
    metrics = pd.read_excel(os.path.join(data_elab_dir, 'MetricsTEP.xlsx'))
    # Load gfp
    gfp = np.load(os.path.join(data_elab_dir, 'gfp.npy'), allow_pickle=True)
    # Load evoked
    evoked = np.load(os.path.join(data_elab_dir, 'evoked.npy'), allow_pickle=True)

#%%
# Plot results
figures = plot_results(metrics, gfp, evoked, opt_ex)

# Merge patient data
merge_patient_data()


























