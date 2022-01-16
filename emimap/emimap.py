#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, atan2, radians
from sklearn.linear_model import LinearRegression

'''
Based On: 
  https://towardsdatascience.com/easy-steps-to-plot-geographic-data-on-a-map-python-11217859a2db
'''

# noise we measure is the sum of the noise collected by the antenna
# and the internal noise (system noise figure), this affects the
# measurements close to the system noise floor.
def correct_noise_figure(noisedB, systemNFdB):
    measured_noise_lin = 10**(noisedB/10)
    noise_system_lin = 10**(systemNFdB/10)
    noise_antenna_lin = measured_noise_lin - noise_system_lin
    noise_antenna_dB = 10*np.log10(noise_antenna_lin)
    return noise_antenna_dB

df = pd.read_csv("dataset.txt")
print(df.head())

# Map boundaries

lat_max=-34.5676
lat_min=-35.2052
lon_min=138.4113
lon_max=139.0073
BBox=(lon_min, lon_max, lat_min, lat_max)
ruh_m = plt.imread("map.png")

lon=df.longitude
lat=df.latitude
noise=df.noise
noise_floor=-146
noise_scale = 0.01
corrected_noise = np.zeros(lon.size)

# Plot measurement sites and a bar representing noise level above noise floor of measurement
# system

fig, ax = plt.subplots()
ax.scatter(df.longitude, df.latitude, zorder=1, alpha= 1, c='b', s=10)
for i in range(lon.size):
    corrected_noise[i] = correct_noise_figure(noise[i], noise_floor)
    dy = noise_scale*(corrected_noise[i]-noise_floor)
    print(noise[i], corrected_noise[i],  corrected_noise[i]+noise_floor, dy)
    ax.plot((lon[i],lon[i]), (lat[i], lat[i]+dy),'r')
ax.set_title('Noise Power around Adelaide')
ax.set_ylim(lat_min, lat_max)
ax.set_xlim(lon_min, lon_max)
ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= 'equal')
plt.savefig("emimap.png", bbox_inches='tight', dpi=600)

# Plot noise power against distance from centre of Adelaide

def haversine(lat1,lon1,lat2,lon2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

lat_adel = -34.928
lon_adel = 138.600
distance_km = np.zeros(lon.size)
for i in range(lon.size):
    distance_km[i] = haversine(lat_adel,lon_adel,lat[i],lon[i])
    print(distance_km[i],corrected_noise[i])
x = distance_km.reshape((-1, 1))
model = LinearRegression().fit(x, corrected_noise)
r_sq = model.score(x, corrected_noise)
print(model.intercept_,model.coef_, r_sq)
y_pred = model.predict(x)
plt.figure(2)
plt.scatter(distance_km,corrected_noise)
plt.plot(distance_km,y_pred)
plt.xlabel('Distance from CBD (km)')
plt.ylabel('Measured noise power (dBm/Hz)')
plt.title('Noise Power against Distance from Adelaide CBD')
plt.grid()
plt.savefig("noise_dist.png")
