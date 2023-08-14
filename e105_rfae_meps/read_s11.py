import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import os
import skrf as rf
from skrf import Network
from scipy.signal import find_peaks
from subprocess import check_output
from skrf.calibration import OnePort


ntwk = rf.Network('S11_100khz_1GHz.s1p')

# extract 1-port objects from multiport objects
s11 = ntwk.s11
print(s11)# extract 1-port objects from multiport objects
ntwk.write_touchstone(dir='D:/acoustoelectric_instrumentation/e57_smithchart/')
# rf.stylely() 
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121)
ntwk.s11.plot_s_smith(m=0,n=0, 
                r=1,
                chart_type='z',
                ax=ax1,
                show_legend=True,
                draw_labels=True,
                draw_vswr=True)
ax2 = fig.add_subplot(122); ax2.grid()
ntwk.s11.plot_s_db()
fig.tight_layout()
plt.savefig("smith_chart.png") 
plt.show()
