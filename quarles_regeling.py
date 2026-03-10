import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


pomp2 = pd.read_csv('data/2001-BS.csv', sep='\t', parse_dates=['Time'], dayfirst=True, index_col='Time')
pomp3 = pd.read_csv('data/3001-BS.csv', sep='\t', parse_dates=['Time'], dayfirst=True, index_col='Time')
pomp4 = pd.read_csv('data/4001-BS.csv', sep='\t', parse_dates=['Time'], dayfirst=True, index_col='Time')

akz = pd.read_csv('data/AKZ-LC-0001.csv', sep='\t', parse_dates=['Time'], dayfirst=True, index_col='Time')

# drop all rows with values other than 1 or 2
pomp2 = pomp2[pomp2['Waarde'].isin([1, 2])]
pomp3 = pomp3[pomp3['Waarde'].isin([1, 2])]
pomp4 = pomp4[pomp4['Waarde'].isin([1, 2])]

fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(16, 10))
ax[1].plot(pomp2.index, pomp2['Waarde'], label='Pomp 2', drawstyle='steps-post')
ax[2].plot(pomp3.index, pomp3['Waarde'], label='Pomp 3', drawstyle='steps-post')
ax[3].plot(pomp4.index, pomp4['Waarde'], label='Pomp 4', drawstyle='steps-post')
ax[0].plot(akz.index, akz['Waarde'], label='AKZ', drawstyle='steps-post', color='k')

for idx, a in enumerate(ax):
    a.legend()
    a.set_ylabel('Status')
    a.grid()

    if idx > 0:
        a.set_yticks([1, 2])
        a.set_yticklabels(['Uit', 'Aan'])

plt.xlabel('Tijd')
plt.suptitle('Quarles regeling pompen en AKZ')
plt.savefig('quarles_regeling.png', dpi=300)
plt.show()
