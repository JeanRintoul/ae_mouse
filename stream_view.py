# 
# 
# 
# 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib
# 
prefix = str(10)
filename = prefix+'_stream.npy'
# print ('filename is',filename)
d = np.load(filename,allow_pickle = True)
a,b,c = d.shape
# print ('data shape',a,b,c)
data = d.transpose(1,0,2).reshape(b,-1) 
a,b = data.shape

# 
print (a,b)
# 
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
plt.plot(data[0])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()