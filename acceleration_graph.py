# import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl # in python
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from operator import itemgetter
from more_itertools import unique_everseen


j=13
# 31 - 0002
# 90 - 0003
# 13 - 0004
# 31 - 0004
annotations_0 = 'data/train/00004/annotations_0.csv'
acceleration = 'data/train/00004/acceleration.csv'
df = pd.read_csv(annotations_0)
start = df['start'][j]
end = df['end'][j]
print(df['name'][j])

def check(value,a,b):
    if a <= value <= b:
        return True
    return False

acceleration_df = pd.read_csv(acceleration)
time = acceleration_df['t']
x = acceleration_df['x']
y = acceleration_df['y']
z = acceleration_df['z']
X = []
Y = []
Z = []

for i,t in enumerate(time):
	if check(t,start,end):
		X.append(x[i])
		Y.append(y[i])
		Z.append(z[i])



# fig stuff
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot3D(X,Y,Z,'#f4a261')

for i,x in enumerate(X):
	y = Y[i]
	z = Z[i]
	ax.scatter3D(X,Y,Z,color='#f4a261')
	ax.text(x,y,z+0.006,i,size=6)


t = len(X)
print(t)
ax.text(X[0]+0.09,Y[0],Z[0]+0.03,'Start',color='#f4a261')
ax.text(X[t-1],Y[t-1]+0.01,Z[t-1]-0.02,'End',color='#f4a261')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
# plt.axis('off')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# ax.xaxis._axinfo['label']['space_factor'] = 1
ax.grid(False)


ax.xaxis.pane.set_edgecolor('#b1b4b5')
ax.yaxis.pane.set_edgecolor('#b1b4b5')
ax.zaxis.pane.set_edgecolor('#b1b4b5')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False


# ax.xaxis.line.set_color('gray')	
# ax.zaxis.line.set_linewidth(0.01)	
# ax.yaxis.line.set_color('gray')	
# ax.zaxis.line.set_color('gray')	
# ax.yaxis._axinfo['label']['space_factor'] = 2.8
for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
    axis.line.set_linewidth(0.01)
ax.view_init(30, 30)
# ax.contour3D(X,Y,Z,cmap='autumn')

plt.title('Acceleration: Walk',y=1.05)
plt.savefig('figures/walk_plot.png',bbox_inches='tight',dpi=300)
