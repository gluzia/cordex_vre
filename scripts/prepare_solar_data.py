# %%
import os
import numpy as np
import pandas as pd

# =============================================================================
# USER SETTINGS
# =============================================================================

DIR_PATH = '/afs/ictp.it/home/g/gluzia_d/Documents/Postdoc/IAEA/Scripts/solar/CSVs/'
CS_PATH  = '/afs/ictp.it/home/g/gluzia_d/Documents/Postdoc/IAEA/Data/Solar_rad/clearsky_pvlib-ICOS.csv'
GSA_PATH = '/afs/ictp.it/home/g/gluzia_d/Documents/Postdoc/IAEA/Data/GSA/gsa_meanGHI.csv'
OUTDIR   = '/afs/ictp.it/home/g/gluzia_d/Documents/Postdoc/IAEA/Scripts/solar/CSVs/standardized/'
os.makedirs(OUTDIR, exist_ok=True)

# Periods
OBS_START = '2012-01-01 00:00:00'
OBS_END   = '2025-05-31 23:00:00'

MODEL_START_FOR_SCALING = '2006'
MODEL_END_FOR_SCALING   = '2024'

# =============================================================================
# INPUT FILES 
# =============================================================================

FILES = {
    'OBS':   '20years_RSDS-TS_OBS_2011-2024_ICOS.csv',

    'NCC-R': '20years_RSDS-TS_NCC-NorESM1-M_RegCM4-6_2006-2025_ICOS.csv',
    'CNR-R': '20years_RSDS-TS_CNRM-CERFACS-CNRM-CM5_RegCM4-6_2006-2025_ICOS.csv',
    'ICH-R': '20years_RSDS-TS_ICHEC-EC-EARTH_RegCM4-6_2006-2025_ICOS.csv',
    'MPI-R': '20years_RSDS-TS_MPI-M-MPI-ESM-LR_RegCM4-6_2006-2025_ICOS.csv',
    'HAD-R': '20years_RSDS-TS_MOHC-HadGEM2-ES_RegCM4-6_2006-2025_ICOS.csv',

    'CNR-A': '20years_RSDS-TS_CNRM-CERFACS-CNRM-CM5_ALADIN63_2006-2025_ICOS.csv',
    'NCC-A': '20years_RSDS-TS_NCC-NorESM1-M_ALADIN63_2006-2025_ICOS.csv',
    'HAD-A': '20years_RSDS-TS_MOHC-HadGEM2-ES_ALADIN63_2006-2025_ICOS.csv',
    'MPI-A': '20years_RSDS-TS_MPI-M-MPI-ESM-LR_ALADIN63_2006-2025_ICOS.csv',

    'CNR-4': '20years_RSDS-TS_CNRM-CERFACS-CNRM-CM5_RCA4_2006-2025_ICOS.csv',
    'ICH-4': '20years_RSDS-TS_ICHEC-EC-EARTH_RCA4_2006-2025_ICOS.csv',
    'IPS-4': '20years_RSDS-TS_IPSL-IPSL-CM5A-MR_RCA4_2006-2025_ICOS.csv',
    'HAD-4': '20years_RSDS-TS_MOHC-HadGEM2-ES_RCA4_2006-2025_ICOS.csv',
    'MPI-4': '20years_RSDS-TS_MPI-M-MPI-ESM-LR_RCA4_2006-2025_ICOS.csv',
    'NCC-4': '20years_RSDS-TS_NCC-NorESM1-M_RCA4_2006-2025_ICOS.csv',

    'CNR-H': '20years_RSDS-TS_CNRM-CERFACS-CNRM-CM5_HadREM3-GA7-05_2006-2025_ICOS.csv',
    'HAD-H': '20years_RSDS-TS_MOHC-HadGEM2-ES_HadREM3-GA7-05_2006-2025_ICOS.csv',
    'ICH-H': '20years_RSDS-TS_ICHEC-EC-EARTH_HadREM3-GA7-05_2006-2025_ICOS.csv',
    'MPI-H': '20years_RSDS-TS_MPI-M-MPI-ESM-LR_HadREM3-GA7-05_2006-2025_ICOS.csv',
    'NCC-H': '20years_RSDS-TS_NCC-NorESM1-M_HadREM3-GA7-05_2006-2025_ICOS.csv',

    'ERA5':  '20years_RSDS-TS_ERA5_2006-202506_ICOS_shift.csv',}

SITES = [
    'Brasschaat', 'Dorinne', 'Lochristi', 'Lonzee', 'Maasmechelen',
    'Vielsalm', 'Davos', 'Bily Kriz forest', 'Lanzhot', 'Trebon',
    'Gebesee', 'Grillenburg', 'Hainich', 'Hartheim', 'Hohes Holz',
    'Hetzdorf', 'Klingenberg', 'Mooseurach', 'Rollesbroich','Selhausen Juelich',
    'Wustebach', 'Tharandt',
    'Gludsted Plantage','Skjern', 'Soroe', 'Voulundgaard',
    'Hyytiala', 'Kenttarova', 'Lettosuo', 'Siikaneva', 'Sodankyla', 'Varrio',
    'Aurade', 'Bilos', 'Col du Lautaret', 'Estrees-Mons A28',
    'Font-Blanche', 'Fontainebleau-Barbeau', 'Grignon', 'Hesse','La Guette',
    'Lamasquere', 'Laqueuille', 'Lusignan', 'Mejusseaume','Puechabon', 'Toulouse',
    'Borgo Cioffi',
    'Lison', 'Monte Bondone', 'Nivolet', 'Arca di Noe - Le Prigionette', 'Renon',
    'San Rossore 2', 'Torgnon', 'Loobos', 'Degero',
    'Hyltemossa', 'Norunda', 'Abisko-Stordalen Palsa Bog', 'Svartberget', 'Auchencorth Moss']

# =============================================================================
# PROCESSING
# =============================================================================

def process_model(df, obsdf3, gsa_path, sites, apply_gsa=True):

    points = [
        'time','BE-Bra', 'BE-Dor', 'BE-Lcr', 'BE-Lon', 'BE-Maa',
        'BE-Vie', 'CH-Dav', 'CZ-BK1', 'CZ-Lnz', 'CZ-wet',
        'DE-Geb', 'DE-Gri', 'DE-Hai', 'DE-Har', 'DE-HoH', 'DE-Hzd', 'DE-Kli', 'DE-Msr', 'DE-RuR',
        'DE-RuS', 'DE-RuW','DE-Tha',
        'DK-Gds', 'DK-Skj', 'DK-Sor', 'DK-Vng',
        'FI-Hyy', 'FI-Ken', 'FI-Let','FI-Sii', 'FI-Sod', 'FI-Var',
        'FR-Aur', 'FR-Bil', 'FR-CLt', 'FR-EM2',
        'FR-FBn', 'FR-Fon', 'FR-Gri', 'FR-Hes', 'FR-LGt',
        'FR-Lam', 'FR-Lqu', 'FR-Lus', 'FR-Mej', 'FR-Pue', 'FR-Tou',
        'IT-BCi',
        'IT-Lsn', 'IT-MBo', 'IT-Niv', 'IT-Noe', 'IT-Ren',
        'IT-SR2', 'IT-Tor', 'NL-Loo', 'SE-Deg', 'SE-Htm', 'SE-Nor', 'SE-Sto', 'SE-Svb', 'UK-AMo']

    try:
        df = df[points]
    except Exception:
        pass

    # normalize time column
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.rename(columns={'datetime': 'time'}, inplace=True)
    elif 'TIMESTAMP' in df.columns:
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
        df.rename(columns={'TIMESTAMP': 'time'}, inplace=True)
    else:
        raise ValueError("No recognized time column found.")

    df = df.set_index('time')

    # rename columns if needed
    initials = [c for c in df.columns if c != 'time']
    if len(initials) == len(obsdf3.columns):
        df = df.rename(columns=dict(zip(initials, obsdf3.columns)))

    df = df[sites]

    if apply_gsa:
        gsa = pd.read_csv(gsa_path)
        gsa = gsa.groupby('site')['GHI'].mean().reset_index().set_index('site')

        df_kwh = df.loc[MODEL_START_FOR_SCALING:MODEL_END_FOR_SCALING] * 0.003
        df_annual_kwh = df_kwh.resample('Y').sum()
        df_clim_kwh = df_annual_kwh.mean(axis=0)

        ratio = (gsa['GHI'] / df_clim_kwh).reindex(sites)

        df = df.apply(
            lambda col: (
                col * ratio.fillna(0.95)[col.name] * 1.1
                if obsdf3[col.name].mean() > 370
                else col * ratio.fillna(0.95)[col.name]
            ),
            axis=0
        )

    df = df.where(obsdf3.notna())
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():

    # ------------------ OBS ------------------
    obs_df = pd.read_csv(os.path.join(DIR_PATH, FILES['OBS']))

    if 'TIMESTAMP' not in obs_df.columns:
        raise ValueError("OBS file expected to have 'TIMESTAMP' column.")

    obs_df['TIMESTAMP'] = pd.to_datetime(obs_df['TIMESTAMP'])
    obs_df = obs_df.set_index('TIMESTAMP')
    obs_df = obs_df[OBS_START:OBS_END].resample('3H').mean()
    obs_df = obs_df[SITES]
    obs_df = obs_df.where(obs_df >= 10, np.nan)

    obsdf3 = obs_df

    # Save standardized OBS
    obs_out = os.path.join(OUTDIR, FILES['OBS'].replace('.csv', '_standard.csv'))
    obsdf3.to_csv(obs_out)

    # ------------------ CLEAR SKY ------------------
    cs_raw = pd.read_csv(CS_PATH)
    cs_df  = process_model(cs_raw.copy(), obsdf3, GSA_PATH, SITES, apply_gsa=False)
    cs_df  = cs_df.where(cs_df >= 10, np.nan)

    # Save standardized clear-sky
    cs_out = os.path.join(OUTDIR, os.path.basename(CS_PATH).replace('.csv', '_standard.csv'))
    cs_df.to_csv(cs_out)

    # ------------------ MODELS ------------------
    for name, fname in FILES.items():

        if name == 'OBS':
            continue

        fullpath = os.path.join(DIR_PATH, fname)

        df = pd.read_csv(fullpath)
        df_gsa = process_model(df.copy(), obsdf3, GSA_PATH, SITES, apply_gsa=True)
        df_gsa = df_gsa.where(df_gsa >= 10, np.nan)

        outname = fname.replace('.csv', '_standard.csv')
        outpath = os.path.join(OUTDIR, outname)

        df_gsa.to_csv(outpath)

        print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
# %%

