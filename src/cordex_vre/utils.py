##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 17:03 2024
@gluzia
"""

#%%
import pyproj
import geopandas as gpd
import xarray as xr
import numpy as np
#from calendar import calendar
import cftime
#from datetime import datetime,timedelta
import dask

#%%

def crs_latlon2xy(
    df,latname,lonname,hgtname,rcm):
    """
    Function that extracts x and y from cordex dataset given any lat and lon, and the 
    proper crs proj chars.

    Parameters
    ----------
    df: a dataframe
    latname: the latitude column name
    lonname: the longitude column name
    hgtname: the hub height column name

    Return a xarray with cordex x and y coordinates. 
    """
    #ToDo add csr projections for other models.
    if rcm=='ALADIN63':
        crs_proj='+proj=lcc +R=6370997.0 +units=km +lat_0=49.5 +lon_0=10.5 +lat_1=49.5 +lat_2=49.5 +x_0=2824897.1381365177 +y_0=2824897.1321809227'  # lambert_conformal
    if rcm=='RegCM4-6':
        crs_proj='+proj=lcc +lat_1=30.00 +lat_2=65.00 +lat_0=48.00 +lon_0=9.75 +x_0=-6000. +y_0=-6000. +ellps=sphere +a=6371229. +b=6371229. +units=m +no_defs' # lambert_conformal + rot. pole
    crs = pyproj.CRS(crs_proj)
    df_gpd = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lonname], df[latname]),
        crs='EPSG:4326')
    df_gpd_new = df_gpd.to_crs(crs=crs_proj)
    df['x'] = df_gpd_new.geometry.x.values
    df['y'] = df_gpd_new.geometry.y.values
    
    locs_xr = df.loc[:,['x','y',latname,lonname,hgtname]].to_xarray()

    return locs_xr

import re
import cartopy.crs as ccrs
import xarray as xr

def crs_latlon2rlatrlon(df, latname, lonname, hgtname, rcm):
    """
    Converts lat/lon columns in a DataFrame to rlat/rlon for a rotated pole grid,
    extracting the rotated pole parameters from a PROJ string (crs_proj).

    Parameters
    ----------
    df: pandas DataFrame 
    latname: latitude column name
    lonname: longitude column name
    hgtname: height column name
    
    Returns
    -------
    locs_xr: xarray.Dataset with columns ['rlat', 'rlon', latname, lonname, hgtname]
    """

    if rcm=='RCA4' or rcm=='HadREM3-GA7-05':
        crs_proj='+proj=ob_tran +R=6370000 +units=m +lon_0=0.0 +o_lon_p=-162 +o_lat_p=39.25 +o_proj=longlat'  #Rotated pole
    o_lon_p = float(re.search(r'o_lon_p=([-\d\.]+)', crs_proj).group(1))
    o_lat_p = float(re.search(r'o_lat_p=([-\d\.]+)', crs_proj).group(1))

    rotated_pole = ccrs.RotatedPole(pole_latitude=o_lat_p, pole_longitude=o_lon_p)
    platecarree = ccrs.PlateCarree()
    lats = df[latname].values
    lons = df[lonname].values
    rlon, rlat = rotated_pole.transform_points(platecarree, lons, lats)[..., :2].T

    df['rlon'] = rlon
    df['rlat'] = rlat

    # Select columns for output
    cols = ['rlon', 'rlat', latname, lonname]
    if hgtname in df.columns:
        cols.append(hgtname)

    locs_xr = df.loc[:, cols].to_xarray()
    return locs_xr


def datetime_to_cftime(date,calendar='proleptic_gregorian'):
    '''Returns a cftime from a datetime

    Args:
        date (string): date in the form yyyymmddhh

    Returns:
        cftime: date in cftime 
    '''

    y,m,d,h = (
        date.dt.year, date.dt.month, date.dt.day, date.dt.hour)
    
    return cftime.datetime(y,m,d,h,
        calendar=calendar)

def date_to_cftime(date,calendar='noleap'):
    '''Returns a cftime from a yyyymmddhh string

    Args:
        date (string): date in the form yyyymmddhh

    Returns:
        cftime: date in cftime 
    '''

    # print(date,len(date))
    if len(date) >= 10:
        y,m,d,h = (
            int(date[0:4]),int(date[4:6]),\
            int(date[6:8]),int(date[8:10])
        )
        return cftime.datetime(y,m,d,h,calendar=calendar)
    else:
        y,m,d = (
            int(date[0:4]),int(date[4:6]),\
            int(date[6:8])
        )
        return cftime.datetime(y,m,d,0,calendar=calendar)

def open_opendap_ds(var,urls,timecoord='False'):
        import pandas as pd
        if len(urls)==1:
            ds = xr.open_dataset(urls[0],chunks={'time': 50}) 
        else:
            ds = xr.open_mfdataset(urls, concat_dim="time",combine='nested',chunks={'time': 20, 'x':151, 'y':151}, parallel=True)  
        try:
            ds = ds.reset_coords(names=['lat','lon'], drop=False)
        except:
            ds = ds.reset_coords(names=['latitude','longitude'], drop=False)
        if var=='tas':
            ds = ds.reset_coords(names=['height'], drop=True)
        if var=='rsds':
            ds = ds.assign_coords(time=ds.time + pd.Timedelta(minutes=90))
            #ds = ds.assign_coords(time=timecoord) #rsds starts at 1:30,this shift the TS 1:30h
        #ds = ds.compute()
        print(var+' assigned', flush=True)
        return ds

def combine_ds(ua,uavar,va,vavar,attrb,h):
    ds = xr.Dataset()
    ds = ds.assign(WS=np.sqrt(ua[uavar]**2+va[vavar]**2))
    ds = ds.assign(WD=np.rad2deg(np.arctan2(ua[uavar],va[vavar])) + 180.)
    ds = ds.assign_attrs(attrb)
    try:
        ds = ds.assign_attrs(ua.Lambert_Conformal.attrs)
    except:
        try:
            ds = ds.assign_attrs(ua.crs.attrs)
        except:
            print('projections specs not found')
    ds = ds.expand_dims(dim='height', axis=0)
    del(ua,va)    
    try:
        ds = ds.reset_coords(names=['lat','lon'], drop=False)
    except:
        print('ds lat and lon are variables', flush=True)
    if h=='s':
        ds = ds.assign_coords(height=([10]))
    else:
        ds = ds.assign_coords(height=([int(h[:-1])]))
    return ds

