import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from math import sqrt
from scipy.stats import norm

priceE = []
dateE = []
priceA = []
dateA = []

tahun = 2001
with open('AIR.PA bulanan.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    sementara = 0
    for row in csv_reader:
        if line_count == 0:
            #print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            
            if line_count % 12 == 0:
                sementara = float(sementara / 12)
                priceE.append(sementara)
                dateE.append(tahun)
                tahun += 1
                sementara = 0
            else:
                sementara += float(row[1])
            #print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
            line_count += 1
            
tahun  = 2001
with open('BA bulanan.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    sementara = 0
    for row in csv_reader:
        if line_count == 0:
            #print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            
            if line_count % 12 == 0:
                sementara = float(sementara / 12)
                priceA.append(sementara)
                dateA.append(tahun)
                tahun += 1
                sementara = 0
            else:
                sementara += float(row[1])
            #print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
            line_count += 1


def brownian(x0, n, dt, delta, out=None):

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out

import numpy
from pylab import plot, show, grid, xlabel, ylabel

# The Wiener process parameter.
deltaE = 10
deltaA = 50
# Total time.
T = 5.0
# Number of steps.
N = 5
# Time step size
dt = T/N
# Number of realizations to generate.
m = 4
# Create an empty array to store the realizations.
x = np.empty((m,N+1))
y = np.empty((m,N+1))
# Initial values of x.
x[:, 0] = priceE[len(priceE)-1]
y[:, 0] = priceA[len(priceA)-1]

brownian(x[:,0], N, dt, deltaE, out=x[:,1:])
brownian(y[:,0], N, dt, deltaA, out=y[:,1:])

tE = numpy.linspace(dateE[len(dateE)-1], dateE[len(dateE)-1]+5, N+1)
tA = numpy.linspace(dateA[len(dateA)-1], dateA[len(dateA)-1]+5, N+1)

# plotting the points 
plt.plot(dateE,priceE)
plt.plot(dateA,priceA)
plt.xlabel('Bulan') 
plt.ylabel('Harga (USD) and (EUR)') 
plt.title('Saham Boeing (Oranye) dan Airbus (Biru) per Bulan'
          '\ndari September 2001 hingga Desember 2019'
          '\nDisertakan prediksi dengan metode Geometric Brownian Motion')
for k in range(m):
    plot(tE, x[k])
    plot(tA, y[k])
xlabel('t', fontsize=16)
ylabel('x', fontsize=16)
grid(True)
show()

