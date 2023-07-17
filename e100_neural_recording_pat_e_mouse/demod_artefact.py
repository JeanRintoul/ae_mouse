'''

time series artefact test, using the correlation metric. 

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift
from scipy import signal
import pandas as pd
from scipy.signal import iirfilter,sosfiltfilt
from scipy import stats 
from scipy.stats import ttest_ind
# 
# with violin plots?
# this data is somewhat filtered. 
ac_filename       = 'acoustic_connection_corrs.npz'
nac_filename      = 'no_acoustic_connection_corrs.npz'
# 
ac_data           = np.load(ac_filename)
ac_corr        	  = np.array(ac_data['corr_data'])
ac_sel            = np.array(ac_data['s_data'])
# 
nac_data          = np.load(nac_filename)
nac_corr          = np.array(nac_data['corr_data'])
nac_sel           = np.array(nac_data['s_data'])
# 
print ('ac sel/nac sel',ac_sel,nac_sel)
print ('ac corr/nac corr',ac_corr,nac_corr)
# 
# 
ac_sel = ac_sel[ac_sel>0]
#
print ('out',ac_sel)
# 
# T-test statistics. 
sample1 = nac_corr
sample2 = ac_corr  
t_stat, p_value = ttest_ind(ac_corr, nac_corr)
print("T-statistic value: ", t_stat)  
print("P-Value: ", p_value)
# 
t_stat, p_value = ttest_ind(ac_sel, nac_corr)
print("T-statistic value: ", t_stat)  
print("P-Value: ", p_value)

# 
# Now do violin plots. 
# 
# Data for violin plot. 
X 		 = [ac_sel,ac_corr,nac_corr]
x_labels = ['ac + threshold','ac','not ac']
print ('number of repeats ac_sel,ac,nac:',len(ac_sel),len(ac_corr),len(nac_corr) )

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
# Create the violin plot
plots = ax.violinplot(X, showmeans=True, showextrema=False, widths=0.8)
colors = ['Red','Black','Grey']
# Set the color of the violin patches
for pc, color in zip(plots['bodies'], colors):
    pc.set_facecolor(color)
plots['cmeans'].set_color('black')

ax.set_ylabel('Correlation coefficient',fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks([1,2,3])
ax.set_xticklabels(x_labels,fontsize=14,rotation=45,ha='right')
fig.tight_layout()
plot_filename = 'violin_plot.png'
plt.savefig(plot_filename)
plt.show()


