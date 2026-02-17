##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:49 2024
@gluzia
"""

#%%
import xarray as xr
import numpy as np
#from datetime import datetime
#import glob
#import os
#import time

# basic libraries
from numpy import newaxis as na
#import pandas as pd
#import geopandas as gpd
#import yaml
#import scipy
#import openmdao.api as om

from finitediff import get_weights
from sklearn.neighbors import NearestNeighbors
#from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter

#%%

def get_interpolation_weights(
    px, py, pz, all_x, all_y, all_z, n_stencil=4, locs_ID=[],
):
    """
    Function that creates the 3D interpolation weights using finite
    differences for multiple interpolations points (px,py,pz), given a grid of
    observed points [all_x, all_y, all_z].

    This function computes the weights for interpolation for different order
    in the horizontal dimensions (x,y), while it computes the weights for both
    linear interpolation and for piecewise logarithmic profile in z.

    Parameters
    ----------
    px: numpy.array
        Interpolation (prediction) points in x
    py: numpy.array
        Interpolation (prediction) points in y
    pz: numpy.array
        Interpolation (prediction) points in z
    all_x: numpy.array
        Observed points in x
    all_y: numpy.array
        Observed points in y
    all_z: numpy.array
        Observed points in z
    n_stencil: int, optional, default=4
        Number of points used in the horizontal interpolation
    locs_ID: list
        Names or ID to identify the locations
    """
    # Number of prediction points and observed points
    Np = len(px)
    Nx = len(all_x)
    Ny = len(all_y)
    Nz = len(all_z)

    if (Np != len(py)) or (Np != len(pz)):
        raise Exception("The len of px, py and pz should be the same")

    # get stencils for interpolations
    n_st_x = n_stencil
    n_st_y = n_stencil
    n_st_z = 2  # In z, interpolation is always based on two points
    if n_stencil > Nx:
        n_st_x = Nx
    if n_stencil > Ny:
        n_st_y = Ny
    if 2 > Nz:
        n_st_z = Nz
    nnx = NearestNeighbors(n_neighbors=n_st_x).fit(all_x[:, na])
    nny = NearestNeighbors(n_neighbors=n_st_y).fit(all_y[:, na])
    nnz = NearestNeighbors(n_neighbors=n_st_z).fit(all_z[:, na])

    # Find the indexes of the observed points to be used for interpolation
    ind_x = np.sort(nnx.kneighbors(px[:, na], return_distance=False), axis=1)
    ind_y = np.sort(nny.kneighbors(py[:, na], return_distance=False), axis=1)
    ind_z = np.sort(nnz.kneighbors(pz[:, na], return_distance=False), axis=1)

    # Find the index for nearest point selection
    nnx_1 = NearestNeighbors(n_neighbors=1).fit(all_x[:, na])
    nny_1 = NearestNeighbors(n_neighbors=1).fit(all_y[:, na])
    nnz_1 = NearestNeighbors(n_neighbors=1).fit(all_z[:, na])
    ind_x_1 = nnx_1.kneighbors(px[:, na], return_distance=False)
    ind_y_1 = nny_1.kneighbors(py[:, na], return_distance=False)
    ind_z_1 = nnz_1.kneighbors(pz[:, na], return_distance=False)

    # Allocate weight matrices
    # Horizontal interpolation weights have the size of the stencil
    weights_x = np.zeros([Np, n_st_x])
    weights_y = np.zeros([Np, n_st_y])
    # Vertical extrapolation weights are always the same size: all available
    # heights
    weights_log_z = np.zeros([Np, Nz])
    weights_z = np.zeros([Np, Nz])
    for i in range(Np):
        weights_x[i, :] = get_weights(
            grid=all_x[ind_x[i, :]],
            xtgt=px[i],
            maxorder=0)[:, 0]

        weights_y[i, :] = get_weights(
            grid=all_y[ind_y[i, :]],
            xtgt=py[i],
            maxorder=0)[:, 0]

        weights_log_z[i, ind_z[i, :]] = get_weights(
            grid=np.log(all_z[ind_z[i, :]]),
            xtgt=np.log(pz[i]),
            maxorder=0)[:, 0]

        weights_z[i, ind_z[i, :]] = get_weights(
            grid=all_z[ind_z[i, :]],
            xtgt=pz[i],
            maxorder=0)[:, 0]

    if len(locs_ID) == 0:
        locs_ID = np.arange(Np, dtype=int)

    # Build dataset
    weights_ds = xr.Dataset(
        data_vars={
            'weights_x': (
                ['locs_ID', 'ix'],
                weights_x,
                {'description': 'Interpolation weights based on finite differences'}),
            'ind_x': (
                ['locs_ID', 'ix'],
                ind_x,
                {'description': 'Indices of WRF grid to use in the interpolation'}),
            'weights_y': (
                ['locs_ID', 'iy'],
                weights_y,
                {'description': 'Interpolation weights based on finite differences'}),
            'ind_y': (
                ['locs_ID', 'iy'],
                ind_y,
                {'description': 'Indices of WRF grid to use in the interpolation'}),
            'weights_z': (
                ['locs_ID', 'iz'],
                weights_z,
                {'description': 'Interpolation weights based on finite differences'}),
            'weights_log_z': (
                ['locs_ID', 'iz'],
                weights_log_z,
                {'description': 'Logaritmic interpolation weights based on finite differences'}),
            'ind_z': (
                ['locs_ID', 'iz'],
                np.repeat(np.arange(len(all_z))[na, :], Np, axis=0),
                {'description': 'Indices of model grid to use in the interpolation'}),
            'ind_x_1': (
                ['locs_ID'],
                ind_x_1.flatten(),
                {'description': 'Indices of model grid to use in nearest point selection'}),
            'ind_y_1': (
                ['locs_ID'],
                ind_y_1.flatten(),
                {'description': 'Indices of model grid to use in nearest point selection'}),
            'ind_z_1': (
                ['locs_ID'],
                ind_z_1.flatten(),
                {'description': 'Indices of model grid to use in nearest point selection'}), },
        coords={'locs_ID': locs_ID})

    return weights_ds

def apply_interpolation_f(
    model_ds,
    weights_ds,
    vars_xy_logz=["WSPD"],
    var_x_grid='west_east',
    var_y_grid='south_north',
    var_z_grid='height',
):
    """
    Function that applies interpolation to a model simulation.

    Parameters
    ----------
    model_ds: xarray.Dataset
        Weather timeseries
    weights_ds: xarray.Dataset
        Weights for locs interpolation for several methods::

            <xarray.Dataset>
            Dimensions:        (ix: 4, iy: 4, iz: 5, loc: 14962)
            Coordinates:
            * loc            (loc) int64
            Dimensions without coordinates: ix, iy, iz
            Data variables:
                weights_x      (loc, ix) float64
                ind_x          (loc, ix) int64
                weights_y      (loc, iy) float64
                ind_y          (loc, iy) int64
                weights_z      (loc, iz) float64
                weights_log_z  (loc, iz) float64
                ind_z          (loc, iz) int64
                ind_x_1        (loc)     int64
                ind_y_1        (loc)     int64
                ind_z_1        (loc)     int64

    vars_xy_logz: list
        List of variables to be interpolated in horizontal (x,y) using finite
    var_x_grid: string, default:'west_east'
        Name of the variable in the weather data used as x in the interpolation
    var_y_grid: string, default: 'south_north'
        Name of the variable in the weather data used as y in the interpolation
    var_z_grid: string, default:'height'
        Name of the variable in the weather data used as z in the interpolation
    
    Returns
    --------
    interp: xarray.Dataset
        Dataset including meso-variables timeseries, interpolated at each locs.
        The arrays have two dimensions: ('Time', 'locs').

    """

    interp = xr.Dataset()
    #interp = pd.DataFrame()

    # power law profile in z
    for var in vars_xy_logz:
        if var not in ['WSPD', 'WS', 'ws', 'wspd']:
            interp[var] = (model_ds.get(var).isel({
                var_x_grid: weights_ds.ind_x,
                var_y_grid: weights_ds.ind_y,
                var_z_grid: weights_ds.ind_z,
            })
                * weights_ds.weights_x
                * weights_ds.weights_y
                * weights_ds.weights_log_z).sum(['ix', 'iy', 'iz'])
        else:
            interp[var] = np.exp((np.log(model_ds.get(var) + 1e-12).isel({
                var_x_grid: weights_ds.ind_x,
                var_y_grid: weights_ds.ind_y,
                var_z_grid: weights_ds.ind_z,
            })
                * weights_ds.weights_x
                * weights_ds.weights_y
                * weights_ds.weights_log_z).sum(['ix', 'iy', 'iz']))

    return interp

# %%

