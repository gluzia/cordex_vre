##!/usr/bin/env python3
##!/afs/ictp.it/home/g/gluzia_d/.anaconda3/envs/pyesgf/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:49 2024
@gluzia
"""

#%%
import numpy as np
from datetime import datetime
from pyesgf.search import SearchConnection
from cordex_vre import config
import os

#%%
'''
from pyesgf.logon import LogonManager
lm = LogonManager()
lm.logoff()
lm.is_logged_on()
OPENID = config.openid #'https://esgf-data.dkrz.de/esgf-idp/openid/gldco' cordex openID
password = config.password
lm.logon_with_openid(openid=OPENID,password=password,bootstrap=True) #enter the password 
lm.is_logged_on()
'''
os.environ["ESGF_PYCLIENT_NO_FACETS_STAR_WARNING"] = "1"

# %%
def search_esgf(var,year_s,year_e,freq,gcm,rcm,ens,node,server,dom='EUR-11',exp='historical'):
    '''Search dataset in the ESGF database'''

    date_s = datetime(year_s,1,1,0)
    date_e = datetime(year_e+1,1,1,0)
    print(date_s,date_e)

    servers = ['https://cordexesg.dmi.dk','https://esgf-node.llnl.gov/esg-search','https://esgf-data.dkrz.de/esg-search','https://esgf-index1.ceda.ac.uk/esg-search','https://esgf-node.ipsl.upmc.fr','https://esg1.umr-cnrm.fr','https://aims3.llnl.gov','https://esgf2.dkrz.de','http://esg1.umr-cnrm.fr', 'http://esgf-node.cels.anl.gov','https://esg-dn1.nsc.liu.se/esg-search']
    #server = servers[10] #2 1, 10 is good for RCA4
    print('Searching on server' ,server)
    conn = SearchConnection(server, distrib=True)
      
    #ctx = conn.new_context(facets='project,experiment_id,product,variable,time_frequency,domain,driving_model,rcm_name,ensemble',
    ctx = conn.new_context(
    project='CORDEX', 
    product = 'output',
    experiment=exp,
    variable=var,
    time_frequency = freq,
    domain = dom,
    driving_model = gcm,
    rcm_name = rcm,
    #replica=True,
    ensemble=ens,
    latest=True,
    data_node = node, #'esgf.ceda.ac.uk'
    )
    nmembers=ctx.hit_count
    urls = []
    if (nmembers):
        result = ctx.search()[0]  
        files = result.file_context().search()
        for file in files:
            start_date = file.opendap_url.split("_")[-1].split("-")[0][0:10]
            end_date = file.opendap_url.split("_")[-1].split("-")[1][0:10]
            start_time = datetime.strptime(start_date, '%Y%m%d%H')
            end_time = datetime.strptime(end_date, '%Y%m%d%H')
       
            if (start_time >= date_s) and (end_time <= date_e):
                print(start_date,end_date)
                print(file.opendap_url)
                urls.append(file.opendap_url)
        return urls
    else: 
        print("Not found")

def index_point(lats, lons, latpoint, lonpoint):

    """osmasa function: return the nearest gridpoint for a given lat lon point
    """
    xx = lons
    yy = lats

    # euclidian distance to find the nearest point
    # argmin return the position of the smaller values
    # unravel index for 2d dimension
    dist = ((yy-latpoint)**2 + (xx-lonpoint)**2)**.5
    id_min = dist.argmin()
    return np.unravel_index([id_min], xx.shape)

