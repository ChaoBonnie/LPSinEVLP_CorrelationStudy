import pandas as pd
from scipy import stats
import numpy as np
import seaborn as sns; sns.set(style='white', context='paper')
import matplotlib.pyplot as plt


### Set up the two main dataframes ###


df_EVLPinfo = pd.read_excel('C:/Users/chaob/OneDrive/Documents/IBBME-UHN/LPS (2020)/EVLP Full Info.xlsx', sheet_name='Relevant')
df_EVLPinfo = pd.pivot_table(df_EVLPinfo, index=["EVLP_ID_NO", "TIME_LABEL"], columns="PARAMETER_NAME", values="PARAMETER_VALUE").sort_index()

df_LPS = pd.read_excel('C:/Users/chaob/OneDrive/Documents/IBBME-UHN/LPS (2020)/EVLP Full Info.xlsx', sheet_name='LPS')
df_LPS = df_LPS.set_index('EVLP_ID_NO')
df_LPS = df_LPS.stack()
df_LPS = df_LPS.rename('LPS')
df_LPS = pd.DataFrame(df_LPS)
df_LPS.index = df_LPS.index.set_names(['EVLP_ID_NO', 'TIME_LABEL'])

df_EVLP_LPS = df_EVLPinfo.join(df_LPS)
print(df_EVLP_LPS.to_string())
print(df_LPS.stack([0]).reindex())


### Create correlation heat maps for the overall LPS correlations with other EVLP parameters ###


def correlation_tables(df, drop_col):

    overall_correlations, overall_ps = [], []
    for hr in [1, 2, 4]:
        df_hour = df.loc[(slice(None), hr), :]
        correlations = df_hour.corr(method=lambda x, y: stats.spearmanr(x, y)[0])
        ps = df_hour.corr(method=lambda x, y: stats.spearmanr(x, y)[1])
        overall_correlations.append(correlations.loc[drop_col, :].drop(drop_col))
        overall_ps.append(ps.loc[drop_col, :].drop(drop_col))

    overall_correlations = pd.DataFrame(overall_correlations, index=['EVLP 1hr', 'EVLP 2hr', 'EVLP 4hr'])
    overall_ps = pd.DataFrame(overall_ps, index=['EVLP 1hr', 'EVLP 2hr', 'EVLP 4hr'])
    overall_correlations = overall_correlations
    overall_ps = overall_ps.T
    overall_stars = overall_ps.copy()
    overall_stars['EVLP 1hr'] = pd.cut(overall_stars['EVLP 1hr'], bins=[0, 0.0000994, 0.000994, 0.00994, 0.0495], include_lowest=True, labels=['****', '***', '**', '*'])
    overall_stars['EVLP 2hr'] = pd.cut(overall_stars['EVLP 2hr'], bins=[0, 0.0000994, 0.000994, 0.00994, 0.0495], include_lowest=True, labels=['****', '***', '**', '*'])
    # overall_stars['EVLP 3hr'] = pd.cut(overall_stars['EVLP 3hr'], bins=[0, 0.0000994, 0.000994, 0.00994, 0.0495], include_lowest=True, labels=['****', '***', '**', '*'])
    overall_stars['EVLP 4hr'] = pd.cut(overall_stars['EVLP 4hr'], bins=[0, 0.0000994, 0.000994, 0.00994, 0.0495], include_lowest=True, labels=['****', '***', '**', '*'])
    overall_stars = overall_stars.T
    overall_ps = overall_ps.T
    print(overall_correlations.to_string())
    print(overall_ps.to_string(), overall_stars.to_string())

    return overall_correlations, overall_ps, overall_stars

def heat_map(correlation_table, p_table, star_table, fig_name):
    fig = plt.figure(figsize=(8, 2))
    mask = np.invert(p_table < 0.05)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.set(font_scale=0.6)
    ax = sns.heatmap(correlation_table, mask=mask, linewidths=1, cmap=cmap, vmin=-0.5, vmax=0.5, annot=star_table, fmt = '')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode='anchor')
    ax.invert_yaxis()
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=200)
    plt.close()

correlations_all, ps_all, stars_all = correlation_tables(df_EVLP_LPS, 'LPS')
heat_map(correlations_all, ps_all, stars_all, 'LPS Levels vs EVLP Parameters (All)')


### LPS Correlations in Declined or Transplanted Lungs ###


df_outcome = pd.read_excel('C:/Users/chaob/OneDrive/Documents/IBBME-UHN/LPS (2020)/EVLP Full Info.xlsx', sheet_name='Outcome')
df_outcome = df_outcome.set_index('EVLP_ID_NO')
df_EVLP_LPS_outcome = df_EVLP_LPS.join(df_outcome)
df_4hr = df_EVLP_LPS_outcome.loc[(slice(None), 4), :]
df_EVLP_LPS_Dec = df_EVLP_LPS_outcome[df_EVLP_LPS_outcome['Outcome'] == 0]
df_EVLP_LPS_Tx = df_EVLP_LPS_outcome[df_EVLP_LPS_outcome['Outcome'] == 1]

correlations_Dec, ps_Dec, stars_Dec = correlation_tables(df_EVLP_LPS_Dec, 'LPS')
heat_map(correlations_Dec, ps_Dec, stars_Dec, 'LPS Levels vs EVLP Parameters (Dec)')
correlations_Tx, ps_Tx, stars_Tx = correlation_tables(df_EVLP_LPS_Tx, 'LPS')
heat_map(correlations_Tx, ps_Tx, stars_Tx, 'LPS Levels vs EVLP Parameters (Tx)')


### Intensive-Care Unit (ICU) Length of Stay Correlations with LPS Levels and Other EVLP Parameters ###


# LPS Correlation with ICU Length of Stay #

df_LPS_outcome = df_LPS.loc[(slice(None), 4), :].join(df_outcome)
print(df_LPS_outcome.to_string())
df_4hr_Tx = df_LPS_outcome[df_LPS_outcome['Outcome'] == 1]
correlations_Tx = df_4hr_Tx.corr(method=lambda x, y: stats.spearmanr(x, y)[0])
ps_Tx = df_4hr_Tx.corr(method=lambda x, y: stats.spearmanr(x, y)[1])
print(correlations_Tx.to_string())
print(ps_Tx.to_string())

# ICU-Length of Stay Correlations with Everything Else at EVLP-4hrs #

df_tx_outcome = pd.read_excel('C:/Users/chaob/OneDrive/Documents/IBBME-UHN/LPS (2020)/EVLP Full Info.xlsx', sheet_name='Tx Outcome')
df_tx_outcome = df_tx_outcome.set_index('EVLP_ID_NO')
df_EVLPinfo_outcome = df_EVLPinfo.loc[(slice(None), 4), :].join(df_tx_outcome)
correlations_Tx = df_EVLPinfo_outcome.corr(method=lambda x, y: stats.spearmanr(x, y)[0])
ps_Tx = df_EVLPinfo_outcome.corr(method=lambda x, y: stats.spearmanr(x, y)[1])
print(correlations_Tx.to_string())
print(ps_Tx.to_string())


### Correlate LPS Trends with Other EVLP Parameters ###


# Calculate individual best-fit slopes from LPS trends #

df_LPStrend = pd.read_excel(r'C:/Users/chaob/OneDrive/Documents/IBBME-UHN/LPS (2020)/LPS 2hr Test/Updated LPS Data.xlsx',
                            sheet_name='Calculate Trend')
df_LPStrend = df_LPStrend[:297]
df_LPStrend_dec = df_LPStrend[df_LPStrend['Outcome'] == 0]
df_LPStrend_tx = df_LPStrend[df_LPStrend['Outcome'] == 1]

slopes_dec, slopes_tx, slopes_all = [], [], []
for case_dec in df_LPStrend_dec['EVLP ID']:
    m_dec, _, _, _, _ = stats.linregress(df_LPStrend_dec['EVLP Time Point (hr)'][df_LPStrend_dec['EVLP ID'] == case_dec],
                                         df_LPStrend_dec['LPS Level (EU/mL)'][df_LPStrend_dec['EVLP ID'] == case_dec])
    slopes_dec.append(m_dec)
for case_tx in df_LPStrend_tx['EVLP ID']:
    m_tx, _, _, _, _ = stats.linregress(df_LPStrend_tx['EVLP Time Point (hr)'][df_LPStrend_tx['EVLP ID'] == case_tx],
                                        df_LPStrend_tx['LPS Level (EU/mL)'][df_LPStrend_tx['EVLP ID'] == case_tx])
    slopes_tx.append(m_tx)
for case in df_LPStrend['EVLP ID']:
    m_all, _, _, _, _ = stats.linregress(df_LPStrend['EVLP Time Point (hr)'][df_LPStrend['EVLP ID'] == case],
                                        df_LPStrend['LPS Level (EU/mL)'][df_LPStrend['EVLP ID'] == case])
    slopes_all.append(m_all)
slopes_dec_no_repeats = slopes_dec[::3]
slopes_tx_no_repeats = slopes_tx[::3]
slopes_dec = np.repeat(slopes_dec_no_repeats, 4)
slopes_tx = np.repeat(slopes_tx_no_repeats, 4)

df_EVLP_outcome = df_EVLPinfo.join(df_outcome)
df_EVLP_outcome = df_EVLP_outcome.drop(columns='ICU LOS')
df_EVLP_outcome = df_EVLP_outcome.loc[(slice(None), [1, 2, 3, 4]), :]
df_EVLP_Dec = df_EVLP_outcome[df_EVLP_outcome['Outcome'] == 0]
df_EVLP_Tx = df_EVLP_outcome[df_EVLP_outcome['Outcome'] == 1]
df_EVLP_Dec['LPS Trend'] = slopes_dec
df_EVLP_Tx['LPS Trend'] = slopes_tx
print(df_EVLP_Dec.to_string(), df_EVLP_Tx.to_string())

# Visualize the correlations #

correlations_trend_Dec, ps_trend_Dec, stars_trend_Dec = correlation_tables(df_EVLP_Dec, 'LPS Trend')
heat_map(correlations_trend_Dec, ps_trend_Dec, stars_trend_Dec, 'LPS Trend vs EVLP Parameters (Dec)')
correlations_trend_Tx, ps_trend_Tx, stars_trend_Tx = correlation_tables(df_EVLP_Tx, 'LPS Trend')
heat_map(correlations_trend_Tx, ps_trend_Tx, stars_trend_Tx, 'LPS Trend vs EVLP Parameters (Tx)')