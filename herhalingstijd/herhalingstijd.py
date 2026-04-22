import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# inladen
file_name = 'MPN_081851_Gorinchemse Kanaalsluis_Hlng_H.M.0.csv'
df = pd.read_csv(file_name, sep=';', index_col=0, parse_dates=True)

# verwijder afgekeurde en lege waarden
df = df[df['flag'] < 6]

# find index of max value for each year
peaks_df = df.loc[df.groupby(df.index.year)['waarde'].idxmax()]
peaks_df['rank'] = peaks_df.waarde.rank(method='max', ascending=False).astype(int)
peaks_df['weibull T'] = (1 / (peaks_df['rank'] / (len(peaks_df) + 1))).round(2)
peaks_df = peaks_df.sort_values('rank')

df[['eenheid', 'waarde']].to_csv('data/waterstand_gorinchemse_kanaalsluis.csv')
peaks_df[['eenheid', 'waarde', 'weibull T']].to_csv('data/herhalingstijd_gorinchemse_kanaalsluis.csv')

def fit_weibull(T, a, b):
    return a * (T ** b)

t_data, h_data = peaks_df['weibull T'], peaks_df['waarde']
popt, pcov = curve_fit(fit_weibull, t_data, h_data)
t_linspace = np.linspace(1, 100, 100)
h_fit = fit_weibull(t_linspace, *popt)

# plot weibull T
fig, ax = plt.subplots()
ax.plot(peaks_df['weibull T'], peaks_df['waarde'], 'o', color='red', label='Jaarlijkse piek')
ax.plot(t_linspace, h_fit, 'r--', color='black', label='Weibull fit (y={:.2f} * T^{:.2f})'.format(*popt))
ax.set_xlabel('Weibull T (jaar)')
ax.set_ylabel('Waterstand (m+NAP)')
ax.set_title('Herhalingstijd waterstand Gorinchemse Kanaalsluis')
ax.set_xscale('log')
ax.grid()
ax.legend()
fig.savefig('data/herhalingstijd_weibull.png')

# plot
fig, ax = plt.subplots()
ax.plot(df.index, df['waarde'], label='Waterstand', linewidth=0.1, color='black')
ax.plot(peaks_df.index, peaks_df['waarde'], 'ro', label='Jaarlijkse piek', color='red')
ax.set_xlabel('Datum')
ax.set_ylabel('Waterstand (m+NAP)')
ax.set_title('Waterstand Gorinchemse Kanaalsluis (x=126763, y=427469)')
ax.grid()
ax.legend()
# different background shading for each year
for year in df.index.year.unique():
    t_start = pd.Timestamp(f'{year}-01-01')
    t_end = pd.Timestamp(f'{year}-12-31')
    ax.axvspan(t_start, t_end, alpha=0.1, color='gray' if year % 2 == 0 else 'lightgray')
fig.savefig('data/waterstand_gorinchemse_kanaalsluis.png')
