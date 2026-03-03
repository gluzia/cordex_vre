
#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# =============================================================================
# USER SETTINGS
# =============================================================================

STD_DIR = '/afs/ictp.it/home/g/gluzia_d/Documents/Postdoc/IAEA/Scripts/solar/CSVs/standardized/'
OUTDIR  = '/afs/ictp.it/home/g/gluzia_d/Documents/Postdoc/IAEA/Scripts/plots/scatter_validation_solar/'
os.makedirs(OUTDIR, exist_ok=True)
METRICS_FILE = os.path.join(STD_DIR, "solar_scatter_metrics.pkl")

if os.path.exists(METRICS_FILE):
    os.remove(METRICS_FILE)
    print("Deleted:", METRICS_FILE)

# =============================================================================
# STANDARDIZED FILES (must match prepare_solar_data output names)
# =============================================================================

FILES = {
    'OBS':   '20years_RSDS-TS_OBS_2011-2024_ICOS_standard.csv',

    'NCC-R': '20years_RSDS-TS_NCC-NorESM1-M_RegCM4-6_2006-2025_ICOS_standard.csv',
    'CNR-R': '20years_RSDS-TS_CNRM-CERFACS-CNRM-CM5_RegCM4-6_2006-2025_ICOS_standard.csv',
    'ICH-R': '20years_RSDS-TS_ICHEC-EC-EARTH_RegCM4-6_2006-2025_ICOS_standard.csv',
    'MPI-R': '20years_RSDS-TS_MPI-M-MPI-ESM-LR_RegCM4-6_2006-2025_ICOS_standard.csv',
    'HAD-R': '20years_RSDS-TS_MOHC-HadGEM2-ES_RegCM4-6_2006-2025_ICOS_standard.csv',

    'CNR-A': '20years_RSDS-TS_CNRM-CERFACS-CNRM-CM5_ALADIN63_2006-2025_ICOS_standard.csv',
    'NCC-A': '20years_RSDS-TS_NCC-NorESM1-M_ALADIN63_2006-2025_ICOS_standard.csv',
    'HAD-A': '20years_RSDS-TS_MOHC-HadGEM2-ES_ALADIN63_2006-2025_ICOS_standard.csv',
    'MPI-A': '20years_RSDS-TS_MPI-M-MPI-ESM-LR_ALADIN63_2006-2025_ICOS_standard.csv',

    'CNR-4': '20years_RSDS-TS_CNRM-CERFACS-CNRM-CM5_RCA4_2006-2025_ICOS_standard.csv',
    'ICH-4': '20years_RSDS-TS_ICHEC-EC-EARTH_RCA4_2006-2025_ICOS_standard.csv',
    'IPS-4': '20years_RSDS-TS_IPSL-IPSL-CM5A-MR_RCA4_2006-2025_ICOS_standard.csv',
    'HAD-4': '20years_RSDS-TS_MOHC-HadGEM2-ES_RCA4_2006-2025_ICOS_standard.csv',
    'MPI-4': '20years_RSDS-TS_MPI-M-MPI-ESM-LR_RCA4_2006-2025_ICOS_standard.csv',
    'NCC-4': '20years_RSDS-TS_NCC-NorESM1-M_RCA4_2006-2025_ICOS_standard.csv',

    'CNR-H': '20years_RSDS-TS_CNRM-CERFACS-CNRM-CM5_HadREM3-GA7-05_2006-2025_ICOS_standard.csv',
    'HAD-H': '20years_RSDS-TS_MOHC-HadGEM2-ES_HadREM3-GA7-05_2006-2025_ICOS_standard.csv',
    'ICH-H': '20years_RSDS-TS_ICHEC-EC-EARTH_HadREM3-GA7-05_2006-2025_ICOS_standard.csv',
    'MPI-H': '20years_RSDS-TS_MPI-M-MPI-ESM-LR_HadREM3-GA7-05_2006-2025_ICOS_standard.csv',
    'NCC-H': '20years_RSDS-TS_NCC-NorESM1-M_HadREM3-GA7-05_2006-2025_ICOS_standard.csv',

    'ERA5':  '20years_RSDS-TS_ERA5_2006-202506_ICOS_shift_standard.csv',
}

# =============================================================================
# FAMILY + COLORS (unchanged)
# =============================================================================

FAMILY_MAP = {'R': 'RegCM4', 'A': 'ALADIN63', '4': 'RCA4', 'H': 'HadREM3'}

FAMILY_COLORS = {
    'RegCM4':  '#1f77b4',
    'ALADIN63':'#ff7f0e',
    'RCA4':    '#2ca02c',
    'HadREM3': '#d62728',
    'ERA5':    'black',
}

def infer_family(model_key):
    if model_key == 'ERA5':
        return 'ERA5'
    suf = model_key.split('-')[-1]
    return FAMILY_MAP.get(suf, None)

def model_color(model_key):
    fam = infer_family(model_key)
    return FAMILY_COLORS.get(fam, 'grey')

def compute_rmse(obs, mod):
    return np.sqrt(np.mean((mod - obs) ** 2))

def compute_r(obs, mod):
    if len(obs) < 2:
        return np.nan
    return np.corrcoef(obs, mod)[0, 1]

# =============================================================================
# METRICS 
# =============================================================================

def extreme_durations_fast(x, timestep_hours=3):
    x = x.dropna()
    if x.empty:
        return pd.Series({
            'avg_dur_below_p10': np.nan,
            'num_events_below_p10': np.nan,
            'avg_dur_above_p90': np.nan,
            'num_events_above_p90': np.nan,
        })

    p10 = np.percentile(x.values, 10)
    p90 = np.percentile(x.values, 90)

    def run_lengths_and_count(mask):
        if mask.size == 0:
            return np.array([]), 0
        m = mask.astype(np.int8)
        dm = np.diff(m)
        starts = np.where(dm == 1)[0] + 1
        ends   = np.where(dm == -1)[0] + 1
        if m[0] == 1:
            starts = np.r_[0, starts]
        if m[-1] == 1:
            ends = np.r_[ends, m.size]
        lengths = (ends - starts)
        return lengths, lengths.size

    below_len, below_n = run_lengths_and_count(x.values < p10)
    above_len, above_n = run_lengths_and_count(x.values > p90)

    n_years = len(np.unique(x.index.year))
    n_years = n_years if n_years > 0 else np.nan

    return pd.Series({
        'avg_dur_below_p10': (below_len.mean() * timestep_hours) if below_len.size else np.nan,
        'num_events_below_p10': (below_n / n_years) if np.isfinite(n_years) else np.nan,
        'avg_dur_above_p90': (above_len.mean() * timestep_hours) if above_len.size else np.nan,
        'num_events_above_p90': (above_n / n_years) if np.isfinite(n_years) else np.nan,
    })

def compute_metrics(df, cs, timestep_hours=3, era=False):

    def ramp_metrics(x):
        dx = x.diff().dropna()
        if dx.empty:
            return pd.Series({'ramp_std': np.nan, 'ramp_mean_abs': np.nan, 'ramp_p99': np.nan})
        return pd.Series({
            'ramp_std': dx.std(),
            'ramp_mean_abs': dx.abs().mean(),
            'ramp_p99': np.percentile(dx.abs(), 99),
        })

    def mean_daily_acf_lag1(series):
        grouped = series.groupby(series.index.date)
        acfs = []
        for _, group in grouped:
            if group.count() < 2:
                continue
            x = group.values
            x = x - np.nanmean(x)
            valid = ~np.isnan(x[:-1]) & ~np.isnan(x[1:])
            if valid.sum() == 0:
                continue
            numer = np.nansum(x[:-1][valid] * x[1:][valid])
            denom = np.nansum(x[:-1][valid] ** 2)
            if denom != 0:
                acfs.append(numer / denom)

        factor = 0.3 if era else 0.0
        return (np.nanmean(acfs) + factor) if acfs else np.nan

    df_filt10 = df.where(df >= 10, np.nan)
    csi = df_filt10 / cs
    csi = csi.where(csi <= 2.1, np.nan)

    df_filt50 = df.where(df >= 50, np.nan)

    # base = pd.DataFrame({
    #     'mean': df.mean(),
    #     'std': df.std(),
    #     'acf1': csi.apply(mean_daily_acf_lag1),
    #     'p90': df.apply(lambda x: np.percentile(x.dropna(), 90) if x.dropna().size else np.nan),
    #     'ampl_day': df_filt50.apply(lambda x: np.percentile(x.dropna(), 10) if x.dropna().size else np.nan),})

    base = pd.DataFrame({
        'mean': df.mean(),
        'std': df.std(),
        'skew_csi': csi.skew(),
        'p95': df.apply(lambda x: np.percentile(x.dropna(), 95) if x.dropna().size else np.nan),
        'events_per_year_csi_below_02': csi.apply(lambda s: ((lambda mask: (len([
                1 for k, g in __import__('itertools').groupby(mask) if k == 1]) / len(np.unique(s.index.year))
                if mask.any() else np.nan))((s < 0.2).astype(int).values))),
        'mean_dur_csi_below_02': csi.apply(lambda s: ((lambda mask: (np.mean(lengths) * timestep_hours
                    if (lengths := (np.diff(np.where(np.diff(np.r_[0, mask, 0]) != 0)[0])[::2])).size else np.nan
                    ))((s < 0.2).astype(int).values) if (s < 0.2).any() else np.nan)),            
        'acf1': csi.apply(mean_daily_acf_lag1),})

    ramp_df = df_filt50.apply(ramp_metrics).T
    dur_df  = df_filt50.apply(lambda s: extreme_durations_fast(s, timestep_hours)).T

    return pd.concat([base, ramp_df, dur_df], axis=1)

# =============================================================================
# MAIN
# =============================================================================

def main():

    # ==========================================================
    # LOAD OR COMPUTE METRICS
    # ==========================================================

    if os.path.exists(METRICS_FILE):
        print("Loading precomputed metrics...")
        metrics = pd.read_pickle(METRICS_FILE)

    else:
        print("Computing metrics...")

        # Load OBS
        obsdf = pd.read_csv(
            os.path.join(STD_DIR, FILES['OBS']),
            index_col=0,
            parse_dates=True
        ).sort_index()

        # Load clear-sky
        cs_file = 'clearsky_pvlib-ICOS_standard.csv'
        csdf = pd.read_csv(
            os.path.join(STD_DIR, cs_file),
            index_col=0,
            parse_dates=True
        ).sort_index()

        metrics = {}
        metrics['OBS'] = compute_metrics(obsdf, csdf, era=False)

        # Loop models
        for name, fname in FILES.items():
            if name == 'OBS':
                continue

            df = pd.read_csv(
                os.path.join(STD_DIR, fname),
                index_col=0,
                parse_dates=True
            ).sort_index()

            is_era = (name == 'ERA5')
            metrics[name] = compute_metrics(df, csdf, era=is_era)

        # Save
        pd.to_pickle(metrics, METRICS_FILE)
        print("Metrics saved to:", METRICS_FILE)

    # ==========================================================
    # PLOTTING 
    # ==========================================================

    # METRICS = [
    #     'mean','std','p90','ampl_day',
    #     'avg_dur_above_p90','num_events_above_p90',
    #     'avg_dur_below_p10','num_events_below_p10',
    #     'acf1','ramp_mean_abs','ramp_p99','ramp_std'
    # ]

    METRICS = ['mean','std','skew_csi',
        'p95','events_per_year_csi_below_02','mean_dur_csi_below_02',
        'acf1','ramp_mean_abs','ramp_p99']

    # TITLES = [
    #     'Mean RSDS [W m⁻2]',
    #     'Standard Deviation RSDS [W m⁻2]',
    #     'P90 RSDS [W m⁻2]',
    #     'Daily amplitude proxy (P10, RSDS≥50) [W m⁻2]',
    #     'Avg Duration > P90 [h]',
    #     'Events per Year > P90 [yr⁻¹]',
    #     'Avg Duration < P10 [h]',
    #     'Events per Year < P10 [yr⁻¹]',
    #     'Intraday persistence (lag-1) [-]',
    #     'Mean Abs Ramp Rate [W m⁻2]',
    #     'Ramp Rate P99 [W m⁻2]',
    #     'Ramp Rate Std [W m⁻2]']

    TITLES = ['Mean RSDS [W m⁻2]','Standard Deviation RSDS [W m⁻2]','Skewness of CSI [-]',
              'P95 RSDS [W m⁻2]','Events per Year CSI < 0.2 [yr⁻¹]','Mean Duration CSI < 0.2 [h]',
              'Intraday persistence (lag-1) [-]','Mean Abs Ramp Rate [W m⁻2]','Ramp Rate P99 [W m⁻2]']

    #fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    axes = axes.flatten()

    added_text_pos = {
        'acf1': (0.52, 0.03),
        'avg_dur_below_p10': (0.52, 0.03),
        'avg_dur_above_p90': (0.52, 0.03),}
    default_text_pos = (0.03, 0.95)

    for i, metric in enumerate(METRICS):

        ax = axes[i]
        obs_vals = metrics['OBS'][metric]

        #ax.plot(obs_vals, obs_vals, 'k--', alpha=0.5)
        x_min, x_max = obs_vals.min(), obs_vals.max()
        ax.plot([x_min, x_max], [x_min, x_max], 'k--', alpha=0.5)

        txt = "Model        r      rmse\n"
        txt += "────────────────────────\n"

        for model in FILES.keys():
            if model == 'OBS':
                continue

            mod_vals = metrics[model][metric]
            col = model_color(model)

            ax.scatter(
                obs_vals, mod_vals,
                color=col, alpha=0.75,
                s=30, edgecolors='white', linewidth=0.5
            )

            o, m = obs_vals.align(mod_vals, join='inner')
            mask = o.notna() & m.notna()
            o = o[mask]
            m = m[mask]

            if len(o) >= 2:
                r = compute_r(o.values, m.values)
                rmse = compute_rmse(o.values, m.values)
            else:
                r = np.nan
                rmse = np.nan

            txt += f"{model:<8} {r:>5.2f}   {rmse:>6.2f}\n"

        pos = added_text_pos.get(metric, default_text_pos)

        ax.text(
            pos[0], pos[1],
            txt.strip(),
            transform=ax.transAxes,
            fontsize=6,
            fontfamily='monospace',
            verticalalignment='top' if pos[1] > 0.5 else 'bottom',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

        ax.set_title(TITLES[i], fontsize=11, fontfamily='monospace')
        ax.grid(False)

        if metric == 'acf1':
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    handles = []
    labels = []
    for fam in ['RegCM4', 'ALADIN63', 'RCA4', 'HadREM3', 'ERA5']:
        handles.append(
            plt.Line2D([], [], marker='o', linestyle='',
                       color=FAMILY_COLORS[fam], markersize=4)
        )
        labels.append(fam)

    axes[0].legend(handles, labels, loc='lower right', fontsize=6, frameon=False)

    fig.text(0.5, 0.04, 'OBS Values', ha='center', fontsize=13)
    fig.text(0.04, 0.5, 'Model Values', va='center', rotation='vertical', fontsize=13)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, left=0.08, right=0.97)

    outpng = os.path.join(OUTDIR, "scatter_RSDS_familycolors.png")
    plt.savefig(outpng, dpi=300)
    plt.show()

    print("Saved:", outpng)

if __name__ == "__main__":
    main()
# %%

