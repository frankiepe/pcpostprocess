import sys
import csv
import os
sys.path.insert(1, "/Users/pmxfp1/Desktop/PhD-Frankie/cardiac_modelling_repos/Nanion_Data_Process")
from pcpostprocess.trace import Trace as tr
from pcpostprocess import leak_correct as lc
from matplotlib import pyplot as plt
import numpy as np

### specify filepath and json
filepath = "/Users/pmxfp1/Library/CloudStorage/OneDrive-TheUniversityofNottingham/Newton Syncropatch/Frankie_Experiments_Nov_Dec_2023/08122023_MW2/3_drug_protocol_14.34.49/"
json_file = "3_drug_protocol_14.34.49"
trace_data = tr(filepath, json_file)

### specify location of ramp and pulse in protocol
ramp = [850, 1250]
pulses = [[1350, 11350], [17930, 27930]]

### specify results output location
outloc = "/Users/pmxfp1/Desktop/PhD-Frankie/Code/2024/Jan/test2"
if not os.path.exists(outloc):
    os.makedirs(outloc)

### specify drug, concentrations, columns, and colours
drug = "quinidine"
concs = [0.15, 0.5, 1.5]
col_pairs = [['11', '12', '13'], ['14', '15'], ['16', '17']]
colours = ['#1f77b4', '#ff7f0e', '#2ca02c']

### specify number of sweeps to plot, min current, drug sweep range, and crop index
num_swps = 10
min_curr = 200
drug_swps = [6, 19]
crop_ind = 2

### choose additional wells to exclude
exc_wells = []

### specify what plots to produce
protocol = True
leak_corr = True
current = True
block = True
all_block = True
save_csvs = True
if save_csvs:
    all_block = True

### get times and voltages
ts = trace_data.get_times()
vs = trace_data.get_voltage()

if protocol:
    ### plot protocol
    fig, ax = plt.subplots(figsize = (7,3))
    ax.set_title('Voltage Protocol')
    ax.plot(1000*ts, 1000*vs, label = 'protocol')

    ax.axvline(ramp[0], color = 'k', linestyle = '--', linewidth = '0.5', label = 'ramp bounds')
    ax.axvline(ramp[1], color = 'k', linestyle = '--', linewidth = '0.5')
    ax.axvline(999999, color = 'r', linestyle = '--', linewidth = '0.7', label = 'pulse bounds')
    for pulse in pulses:
        ax.axvline(pulse[0], color = 'r', linestyle = '--', linewidth = '0.7')
        ax.axvline(pulse[1], color = 'r', linestyle = '--', linewidth = '0.7')

    ax.set_ylabel('Voltage (mV)')
    ax.set_xlabel('Time (ms)')
    ax.set_xlim(left = 0, right = 1000*ts[-1])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outloc}/voltage_protocol.png")

### perform initial QC
QC_data = trace_data.get_onboard_QC_values()
filt_QC = lc.get_QC_dict(QC_data)

### perform leak correction
current_data = trace_data.get_all_traces(leakcorrect=False)
lc_data = lc.get_leak_corrected(trace_data, current_data, filt_QC, ramp_bounds = [2*r for r in ramp])

store_mean_control = []
store_mean_sweeps = []
for conc, cols in zip(concs, col_pairs):
    if leak_corr:
        ### Save leak correction plots
        outdir = f"{outloc}/{drug}/{conc}/leak_correct"
        for well in filt_QC.keys():
            if well[1:] in cols and well not in exc_wells: # filter out additional wells that don't pass manual QC
                lc.fit_linear_leak(trace_data, well, drug_swps[0] - 2, 
                                   ramp_bounds = [2*r for r in ramp], plot = True, output_dir = outdir)
                lc.fit_linear_leak(trace_data, well, drug_swps[0] + 4, 
                                   ramp_bounds = [2*r for r in ramp], plot = True, output_dir = outdir)

    if current:
        ### Save current plots
        outdir = f"{outloc}/{drug}/{conc}/current"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        for well in filt_QC.keys():
            if well[1:] in cols and well not in exc_wells:
                fig, ax = plt.subplots()
                ax.plot(1000*ts, lc_data[well][drug_swps[0] - 2], label = 'control')
                ax.plot(1000*ts, lc_data[well][drug_swps[0]], label = 'sweep 1', alpha = 0.5)    
                ax.plot(1000*ts, lc_data[well][drug_swps[0] + 4], label = 'sweep 5', alpha = 0.5)
                ax.plot(1000*ts, lc_data[well][drug_swps[1]], label = f'sweep {drug_swps[1] - drug_swps[0] + 1}', alpha = 0.5)
                ax.set_ylabel('Current (pA)')
                ax.set_xlabel('Time (ms)')
                ax.set_xlim(left = 0, right = 1000*ts[-1])
                ax.set_ylim(bottom = -25)
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{outdir}/{well}.png")
                plt.close()

    ### Get mean control and mean sweeps
    all_control = []
    all_sweeps = []
    for well in filt_QC.keys():
        if well[1:] in cols and well not in exc_wells:
            multi_sweeps = []
            last = 0
            control = []
            for pulse in pulses:
                control = control + list(lc_data[well][drug_swps[0] - 2][2*pulse[0]:2*pulse[1]])
            all_control.append(control)
            for i in np.arange(drug_swps[0], num_swps+drug_swps[0]):
                sweep = []
                for pulse in pulses:
                    sweep = sweep + list(lc_data[well][i][2*pulse[0]:2*pulse[1]])
                multi_sweeps = multi_sweeps + list(sweep)
            all_sweeps.append(multi_sweeps)
        
    mean_control = np.mean(all_control, axis=0)
    mean_sweeps = np.mean(all_sweeps, axis=0)
    store_mean_control.append(mean_control)
    store_mean_sweeps.append(mean_sweeps)

    if block:
        ### plot current block
        if not os.path.exists(f"{outloc}/{drug}/{conc}"):
            os.makedirs(f"{outloc}/{drug}/{conc}")
        fig, ax = plt.subplots()
        mean_cont_crop = mean_control[(mean_control>min_curr)]
        for i in np.arange(0, num_swps):
            mean_sweep_crop = mean_sweeps[0+i*len(mean_control):len(mean_control)+
                                          i*len(mean_control)][(mean_control>min_curr)]
            sweep = 1 - (mean_cont_crop - mean_sweep_crop)/mean_cont_crop
            #sweep = sweep[(sweep<1)]
            ax.plot(np.arange(0+i*len(mean_cont_crop), len(sweep)+i*len(mean_cont_crop)), sweep, color = '#9467bd')
        ax.set_ylim(bottom = 0, top = 1.2)
        ax.set_xlim(left = 0)
        ax.set_title(f'{drug} block at {conc}uM')
        ax.set_ylabel('proportion current remaining')
        ax.set_xlabel('sweep')
        swp_len = len(sweep)
        xticks = np.arange(swp_len/2, len(sweep)*num_swps + 1, swp_len)
        for i, xval in enumerate(xticks):
            if i % 2:
                ax.axvspan(xval-swp_len/2, xval+swp_len/2, color='#CACAD2', linestyle='', alpha=0.5)
        ax.set_xticks(xticks, np.arange(1, num_swps+1))
        plt.savefig(f"{outloc}/{drug}/{conc}/block.png")

if all_block:
    ### plot current block
    block_dict = {}
    time_dict = {}
    fig, ax = plt.subplots()
    for mean_control, mean_sweeps, col, conc in zip(store_mean_control, store_mean_sweeps, colours, concs):
        mean_cont_crop = mean_control[(store_mean_control[crop_ind]>min_curr)]
        for i in np.arange(0, num_swps):
            mean_sweep_crop = mean_sweeps[0+i*len(mean_control):len(mean_control)+
                                          i*len(mean_control)][(store_mean_control[crop_ind]>min_curr)]
            sweep_prop = 1 - (mean_cont_crop - mean_sweep_crop)/mean_cont_crop
            time = np.arange(0+i*len(mean_cont_crop), len(sweep_prop)+i*len(mean_cont_crop))
            if str(conc) in block_dict:
                block_dict[str(conc)] += list(sweep_prop)
                time_dict[str(conc)] += [t/2 for t in time]
            else:
                block_dict[str(conc)] = list(sweep_prop)
                time_dict[str(conc)] = [t/2 for t in time]
            if i == 0:
                ax.plot(time, sweep_prop, color = col, label = f'{conc}nM')
            else:
                ax.plot(time, sweep_prop, color = col)
    ax.set_ylim(bottom = 0, top = 1.2)
    ax.set_xlim(left = 0)
    ax.set_title(f'{drug} block')
    ax.set_ylabel('proportion current remaining')
    ax.set_xlabel('sweep')
    swp_len = len(sweep_prop)
    xticks = np.arange(swp_len/2, len(sweep_prop)*num_swps + 1, swp_len)
    for i, xval in enumerate(xticks):
        if i % 2:
            ax.axvspan(xval-swp_len/2, xval+swp_len/2, color='#CACAD2', linestyle='', alpha=0.5)
    ax.set_xticks(xticks, np.arange(1, num_swps+1))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outloc}/{drug}/block.png")

    if save_csvs:
        ### save csvs
        for conc in concs:
            with open(f"{outloc}/{drug}/{conc}/{drug}_conc_{int(conc*1000)}_Milnes.csv", 'w') as f:
                f.write('"time","current"')
                f.write("\n")
                writer = csv.writer(f)
                writer.writerows(zip(time_dict[str(conc)], block_dict[str(conc)]))