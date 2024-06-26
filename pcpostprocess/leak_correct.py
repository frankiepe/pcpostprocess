import os

import numpy as np
from matplotlib import pyplot as plt

from .trace import Trace


def linear_reg(V, I_obs):
    # number of observations/points
    n = np.size(V)

    # mean of V and I vector
    m_V = np.mean(V)
    m_I = np.mean(I_obs)

    # calculating cross-deviation and deviation about V
    SS_VI = np.sum(I_obs*V) - n*m_I*m_V
    SS_VV = np.sum(V*V) - n*m_V*m_V

    # calculating regression coefficients
    b_1 = SS_VI / SS_VV
    b_0 = m_I - b_1*m_V

    # return intercept, gradient
    return b_0, b_1


def get_QC_dict(QC, bounds={'Rseal': (10e8, 10e12), 'Cm': (1e-12, 1e-10),
                            'Rseries': (1e6, 2.5e7)}):
    '''
    @params:
    QC: QC trace attribute extracted from the JSON file
    bounds: A dictionary of bound tuples, (lower, upper), for each QC variable
    '''

    QC_dict = {}
    for well in QC:
        for qc in QC[well]:
            if all(qc):
                if bounds['Rseal'][0] < qc[0] < bounds['Rseal'][1] and \
                   bounds['Cm'][0] < qc[1] < bounds['Cm'][1] and \
                   bounds['Rseries'][0] < qc[2] < bounds['Rseries'][1]:

                    if well in QC_dict:
                        QC_dict[well] = QC_dict[well] + [qc]
                    else:
                        QC_dict[well] = [qc]

    max_swp = max(len(QC_dict[well]) for well in QC_dict)
    QC_copy = QC_dict.copy()
    for well in QC_copy:
        if len(QC_dict[well]) != max_swp:
            QC_dict.pop(well)
    return QC_dict


def get_leak_corrected(trace, currents, QC_filt, ramp_bounds):
    leak_corrected = {}
    V = 1000*np.array(trace.get_voltage())  # mV
    for row in trace.WELL_ID:
        for well in row:
            if well in QC_filt.keys():
                leak_corrected[well] = {}
                for sweep in range(trace.NofSweeps):
                    I_obs = currents[well][sweep]  # pA
                    b_0, b_1 = linear_reg(
                        V[ramp_bounds[0]:ramp_bounds[1]+1],
                        I_obs[ramp_bounds[0]:ramp_bounds[1]+1])
                    I_leak = b_1*V + b_0
                    leak_corrected[well][sweep] = I_obs - I_leak
    return leak_corrected


def fit_linear_leak(trace: Trace, well, sweep, ramp_bounds, plot=False,
                    label='', output_dir=''):

    voltage = trace.get_voltage()
    times = trace.get_times()

    current = trace.get_trace_sweeps([sweep])[well]

    if len(current) == 0:
        return (np.nan, np.nan), np.empty(times.shape)

    current = current.flatten()

    # Convert to mV for convinience
    V = voltage * 1e3

    I_obs = current  # pA
    b_0, b_1 = linear_reg(V[ramp_bounds[0]:ramp_bounds[1]+1],
                          I_obs[ramp_bounds[0]:ramp_bounds[1]+1])
    I_leak = b_1*V + b_0

    if plot:
        # fit to leak ramp
        fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(7.5, 6))

        ax1.set_title('current vs time')
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('current (pA)')
        ax1.plot(times, I_obs)
        ax1.axvline(ramp_bounds[0]*0.0005, linestyle='--', color='k', alpha=0.5)
        ax1.axvline(ramp_bounds[1]*0.0005, linestyle='--', color='k', alpha=0.5)
        ax1.set_xlim(left=ramp_bounds[0]*0.0005 - 1,
                     right=ramp_bounds[1]*0.0005 + 1)

        ax2.set_title('voltage vs time')
        ax2.set_xlabel('time (s)')
        ax2.set_ylabel('voltage (mV)')
        ax2.plot(times, V)
        ax2.axvline(ramp_bounds[0]*0.0005, linestyle='--', color='k', alpha=0.5)
        ax2.axvline(ramp_bounds[1]*0.0005, linestyle='--', color='k', alpha=0.5)
        ax2.set_xlim(left=ramp_bounds[0]*0.0005 - 1,
                     right=ramp_bounds[1]*0.0005 + 1)

        ax3.set_title('current vs voltage')
        ax3.set_xlabel('voltage (mV)')
        ax3.set_ylabel('current (pA)')
        ax3.plot(V[ramp_bounds[0]:ramp_bounds[1]+1],
                 I_obs[ramp_bounds[0]:ramp_bounds[1]+1], 'x')
        ax3.plot(V[ramp_bounds[0]:ramp_bounds[1]+1],
                 I_leak[ramp_bounds[0]:ramp_bounds[1]+1], '--')

        ax4.set_title(
            f'current vs. time (gleak: {np.round(b_1,1)}, Eleak: {np.round(b_0/b_1,1)})')
        ax4.set_xlabel('time (s)')
        ax4.set_ylabel('current (pA)')
        ax4.plot(times, I_obs, label='I_obs')
        ax4.plot(times, I_leak, linestyle='--', label='I_leak')
        ax4.plot(times, I_obs - I_leak,
                 linestyle='--', alpha=0.5, label='Ikr')
        ax4.legend()

        fig.tight_layout()
        fname = f"{label}_{well}_sweep{sweep}" if label != '' \
            else f"{well}_sweep{sweep}"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig.savefig(os.path.join(output_dir, fname))
        plt.close(fig)

    return (b_0, b_1), I_leak
