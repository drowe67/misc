#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.txt")
print(df.head())

lat_max=-34.5676
lat_min=-35.2052
lon_min=138.4113
lon_max=139.0073
BBox=(lon_min, lon_max, lat_min, lat_max)
print(BBox)
ruh_m = plt.imread("map.png")

lon=df.longitude
lat=df.latitude
#bearing=df.bearing
#line_deg = 0.1
noise=df.noise
noise_floor=-146
noise_scale = 0.01

fig, ax = plt.subplots()
ax.scatter(df.longitude, df.latitude, zorder=1, alpha= 1, c='b', s=10)
for i in range(lon.size):
    '''
    # relate compass bearing to theta on cos-sine plane
    theta = -(bearing[i]-90)*np.pi/180
    dx = line_deg*np.cos(theta)
    dy = line_deg*np.sin(theta)
    print(lon[i],lat[i], bearing[i], dx, dy)
    '''
    dy = noise_scale*(noise[i]-noise_floor)
    print(noise[i], noise[i]+146, dy)
    ax.plot((lon[i],lon[i]), (lat[i], lat[i]+dy),'r')
ax.set_title('Plotting Noise Data on Adelaide Map')
ax.set_ylim(lat_min, lat_max)
ax.set_xlim(lon_min, lon_max)
ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= 'equal')
plt.savefig("test.png", bbox_inches='tight', dpi=600)

