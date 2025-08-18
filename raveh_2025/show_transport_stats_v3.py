import pandas as pd
import numpy as np
from scipy.stats import chi2

VERBOSE=False

def get_particle_concentration_uM(box_side_A=2000, slab_thickness_A=300):
    volume_A3 = box_side_A**2*(box_side_A-slab_thickness_A)
    volume_m3 = volume_A3 * 1e-30
    volume_L = volume_m3 * 1e3
    avogadro = 6.02214076e23
    return 1e6 / volume_L / avogadro

def get_poisson_interval(k, alpha=0.05): 
    """
    uses chisquared info to get the poisson interval with confidnce 1-alpha
    for a specified rate k (in units of time^-1). 
    """
    a = alpha
    low, high = (chi2.ppf(a/2, 2*k) / 2, chi2.ppf(1-a/2, 2*k + 2) / 2)
    if k == 0: 
        low = 0.0
    return low, high

def get_big_sister(df, row):
    if np.isclose(row['from_sec'], row['to_sec']-5e-6):
        return None
    is_from= np.isclose(df['from_sec'],row['from_sec']+5e-6)
    is_to=  (df['to_sec']==row['to_sec'])
    is_sites= np.isclose(df['n_sites'], row['n_sites'])
    is_R= (df['R']==row['R']) 
    is_type= (df['type']==row['type'])
    return df[is_from & is_to &  is_sites & is_R & is_type]

def add_conf95_to_rate(df, 
                       label='transport_per_sec_per_particle',
                       total_sim_time_label='total_sim_time_sec',
                       pseudocounts=0.001):
    N= (df[label] + pseudocounts) * df[total_sim_time_label] 
    conf95= np.array([get_poisson_interval(x) for x in N])
    conf95_low= np.minimum(N, pd.Series(conf95[:,0]))
    conf95_high= conf95[:,1] 
    df[label+'_95low']= conf95_low / df[total_sim_time_label]
    df[label+'_95high']= conf95_high / df[total_sim_time_label]
    return df

def add_5us_column(df):
    df['transport_5us_per_sec_per_particle']=np.nan
    for index, row in df.iterrows():
        icol= df.columns.get_loc('transport_5us_per_sec_per_particle')
        big_sister= get_big_sister(df, row)
        if big_sister is None:
            df.at[index, 'transport_5us_per_sec_per_particle']= row['transport_per_sec_per_particle']
            continue
        if not len(big_sister.index)==1:
            if VERBOSE:
                print("Warning: len big sister = {} row index = {}".format(len(big_sister.index), index))
    #            display(row)
            continue
        row_transport= row['transport_per_sec_per_particle']
        big_sister_transport= big_sister['transport_per_sec_per_particle']
        dT= row['to_sec']-row['from_sec']
        df.at[index,'transport_5us_per_sec_per_particle']= (row_transport * dT - big_sister_transport * (dT-5e-6)) / 5e-6
    fraction_5us= (5e-6/(df['to_sec']-df['from_sec']))
    df['total_sim_time_sec_5us']= fraction_5us * df['total_sim_time_sec']
    return df

def read_df(csv_file='all.csv'):
    df= pd.read_csv(csv_file);
    df.dropna(inplace=True);
    df= add_5us_column(df)
    df['n_sites_own']= df['n_sites']
    is_inert= (df['type']=='inert')
    df.loc[is_inert, 'n_sites_own']=0.0
    gb_columns=['n_sites_own','R','type','from_sec','to_sec']
    df= df.join(df.groupby(gb_columns)['transport_per_sec_per_particle'].mean(), on=gb_columns, rsuffix='_r')
    df= df.join(df.groupby(gb_columns)['transport_5us_per_sec_per_particle'].mean(), on=gb_columns, rsuffix='_r')
    df= df.join(df.groupby(gb_columns)['total_sim_time_sec'].sum(), on=gb_columns, rsuffix='_r')
    df= df.join(df.groupby(gb_columns)['total_sim_time_sec_5us'].sum(), on=gb_columns, rsuffix='_r')
    df= add_conf95_to_rate(df)
    df= add_conf95_to_rate(df, 
                           label='transport_5us_per_sec_per_particle', 
                           total_sim_time_label='total_sim_time_sec_5us')
    df= add_conf95_to_rate(df, 
                           label='transport_per_sec_per_particle_r', 
                           total_sim_time_label='total_sim_time_sec_r')
    df= add_conf95_to_rate(df, 
                           label='transport_5us_per_sec_per_particle_r', 
                           total_sim_time_label='total_sim_time_sec_5us_r')
    df['MW_kDa'] = 27*(df['R']/20)**3
    C_uM = get_particle_concentration_uM(box_side_A=2000, slab_thickness_A=300)
    df['transport_per_sec_per_particle_per_uM']= df['transport_per_sec_per_particle'] / C_uM
    df['transport_per_sec_per_particle_per_uM_95low']= df['transport_per_sec_per_particle_95low'] / C_uM
    df['transport_per_sec_per_particle_per_uM_95high']= df['transport_per_sec_per_particle_95high'] / C_uM
    df['n_sites_str'] = [f"{x:02.0f}" for x in df['n_sites']]
    return df

def calc_filtered_transport_stats(df):
    #diffuser='inert'    
    is_from = np.isclose(df['from_sec'], 10e-6)
    is_to = np.isclose(df['to_sec'], 65e-6)
    #is_kap= df['type']==diffuser
    is_sites = (df['n_sites']>=0) & (df['n_sites']<17)
    is_small_inert = (df['type']=='inert') & (df['R']<31)
    is_kap = (df['type']=='kap')
    is_R = df['R']<100
    is_type = (is_kap) & is_R
    slice = df[is_from & is_to & is_sites & is_type]
    return slice
    