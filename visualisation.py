import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np


fig, ax = plt.subplots()

df = pd.read_csv('planets_coordinates.csv', header=None)
dfs = pd.read_csv('sources_coordinates.csv', header=None)
scat = ax.scatter(df.loc[0][::2], df.loc[0][1::2], c=np.random.rand(len(df.loc[0][::2]),3), s=1, linewidths=0)
def frame(i):
    #scat.set_offsets(np.c_[list(df.loc[i][::2]), list(df.loc[i][1::2]), [0]*10000])
    scat.set_offsets(np.c_[list(df.loc[i][::2]), list(df.loc[i][1::2])])
ax.set_xlim(-1e13,1e13)
ax.set_ylim(-1e13,1e13)
#plt.scatter(0, 0,0,marker='o',color='k')
plt.scatter(dfs.loc[0][::2], dfs.loc[0][1::2], marker='o',color='k')
animation = FuncAnimation(fig, frame, frames=range(len(df)), interval=0, repeat=True)
plt.show()
