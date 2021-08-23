#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 8 02:12:18 2020
@author: lachlan
"""

'''Enter a list of 9 number unique cusip codes to retrieve 71 quarterly financial ratios 
back to 1975 and or once data is available for the US listed firm
e.g.,
list_cusip9 = ['037833100', '594918104']
login = WRDS login
df_ratios = wrds_ratios_US(list_cusip9, 'WRDS_login')
'''

import pandas as pd 
import os
os.chdir()
data=pd.read_csv('US_sample_No_ESG_S1.csv').iloc[:, 1:]
unique_cusip = list(data['Instrument'].unique())

def wrds_ratios_US(list_cusip9, login):
    import wrds
    from pandas.tseries.offsets import MonthEnd
    from pandas.tseries.offsets import DateOffset
    import os
    import pandas as pd
    import numpy as np
    import datetime as dt

    #Functions used in WRDS SAS code
    '''https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/wrds-
    financial-ratios/financial-ratios-sas-code/?_ga=2.178422180.1063775577.1588831662
    -418423294.1588831662'''

    unique_cusip = pd.DataFrame(list_cusip9)
    unique_cusip = list(unique_cusip[0].unique()) # in case list is not unique entered
    placeholders_cusip = ','.join('%s' for i in range(len(unique_cusip)))  # '?,?'
    
    #create unique id to retrieve IBES analyst data 
    unique_cusip_8 = pd.DataFrame(unique_cusip, columns={'cusip8'})['cusip8'].astype(str).str[:-1]
    unique_cusip_IBES = list(unique_cusip_8.unique())
    #unique_cusip_IBES = ['03783310','59491810']
    placeholders_cusip_IBES = ','.join('%s' for i in range(len(unique_cusip_IBES)))  # '?,?'

    ###################
    # Connect to WRDS #
    ###################
    
    '''Connects to WRDS API, have to manually input username and password in 
    console first time'''
    
    conn=wrds.Connection(wrds_username = 'WRDS_login')

    '''Extracting data for Ratios Based on Annual Data and Quarterly Data'''
    
    ###################
       # SQL Block #
    ###################
    #/*Get pricing for primary US common shares from Security Monthly table*/
    comppricing = '''
    SELECT cusip, gvkey, iid, datadate, prccm as prc_comp_unadj, 
    (prccm/ajexm) as prc_comp_adj, cshom, dvrate,
    (cshoq*prccm) as mcap_comp, conm
    FROM comp.secm 
    WHERE tpci='0' 
    and fic='USA' 
    and primiss='P'
    and datadate between '01/01/1975' and '12/31/2019'
    and cusip IN ({})'''.format(placeholders_cusip)
    comppricing_query = conn.raw_sql(comppricing, params=(unique_cusip))
    comppricing_query=comppricing_query.rename(columns = {'datadate':'date'})
    comppricing_query['date'] = pd.to_datetime(comppricing_query['date'])
    comppricing_query['date'] = comppricing_query['date'] + MonthEnd(0)
    
    unique_gvkey = list(comppricing_query['gvkey'].unique())
    #unique_gvkey = ['001690','012141']
    placeholders_gvkey = ','.join('%s' for i in range(len(unique_gvkey)))  # '?,?'
    
    #gvkey and permno link and merge into crsp
    ccm = '''
    select gvkey, lpermno as permno, linktype, linkprim, 
    linkdt, linkenddt
    from crsp.ccmxpf_linktable
    where (linktype ='LU' or linktype='LC')
    and (linkprim = 'P' or linkprim = 'C')
    and gvkey IN ({})'''.format(placeholders_gvkey)
    ccm_query = conn.raw_sql(ccm, params=(unique_gvkey))
    
    unique_permno = list(ccm_query['permno'].unique())
    #unique_permno = ['14593', '10107']
    placeholders_permno = ','.join('%s' for i in range(len(unique_permno)))  # '?,?'
    
    #Calculate market value using CRSP
    crsp ='''
    SELECT DISTINCT date, permno, (abs(prc)*shrout)/1000 as mcap_crsp, 
    abs(prc) as prc_crsp_unadj,(abs(prc)/CFACPR) as prc_crsp_adj
    FROM crsp.msf
    WHERE date between '01/01/1975' and '12/31/2019'
    and permno IN ({})'''.format(placeholders_permno)
    crsp_query = conn.raw_sql(crsp, params=(unique_permno))
    crsp_query['date'] = pd.to_datetime(crsp_query['date'])
    crsp_query['date'] = crsp_query['date'] + MonthEnd(0)
    
    crsp_query = crsp_query.merge(ccm_query[['gvkey', 'permno']], how = 'left', on = 'permno')
    
    #/*Grab Historical GICS and merge into crsp*/
    historicalgics = '''
    SELECT gvkey, gsector, ggroup, gind
    FROM comp.co_hgic
    WHERE gvkey IN ({})'''.format(placeholders_gvkey)
    gics_query = conn.raw_sql(historicalgics, params=(unique_gvkey))
    gics_query = gics_query.drop_duplicates(subset='gvkey')

    crsp_query = crsp_query.merge(gics_query, how = 'left', on = 'gvkey')
    
    #merge CRSP and compustat
    crsp_comp =crsp_query.merge(comppricing_query, how = 'left', on=['gvkey', 'date'])
    gvkey = pd.DataFrame(crsp_comp['gvkey'].unique(), columns = {'gvkey'})
    gvkey=crsp_comp[['gvkey','cusip']]
    gvkey['IBES_cusip'] = gvkey['cusip'].astype(str).str[:-1]
    ##############################################################################
    
    #IBES actuals
    #future EPS and annual EPS growth rate from IBES
    
    ibes_actuals = '''
    SELECT cusip, pends, pdicity, anndats, value
    FROM ibes.act_epsus
    WHERE PDICITY = 'ANN' 
    and anndats between '01/01/1975' and '12/31/2019'
    and cusip IN ({})'''.format(placeholders_cusip_IBES)
    ibes_actuals_query = conn.raw_sql(ibes_actuals, params=(unique_cusip_IBES))
    ibes_actuals_query['datadate'] = ibes_actuals_query['pends'].shift(1)
    ibes_actuals_query['current_actual'] = ibes_actuals_query['value'].shift(1)
    ibes_actuals_query['current_anndate'] = ibes_actuals_query['anndats'].shift(1)
    
    ibes_actuals_1 = '''
    SELECT cusip, statpers, actual as fut_actual, meanest as fut_eps, 
    anndats_act as fut_anndate, fpedats as pends
    FROM ibes.statsum_epsus
    WHERE fpi='1' 
    and FISCALP='ANN' 
    and CURR_ACT='USD'
    and anndats_act between '01/01/1975' and '12/31/2019'
    and cusip IN ({})'''.format(placeholders_cusip_IBES)
    ibes_actuals_query_1 = conn.raw_sql(ibes_actuals_1, params=(unique_cusip_IBES))
    
    ibes_merge1 = ibes_actuals_query.merge(ibes_actuals_query_1, how = 'left', on = ['cusip','pends'])
    
    ibes_actuals_0 = '''
    SELECT cusip, meanest as ltg_eps, statpers
    FROM ibes.statsum_epsus
    WHERE statpers between '01/01/1975' and '12/31/2019'
    and fpi ='0'
    and FISCALP='LTG'
    and cusip IN ({})'''.format(placeholders_cusip_IBES)
    ibes_actuals_query_0 = conn.raw_sql(ibes_actuals_0, params=(unique_cusip_IBES))
    
    ibes = ibes_merge1.merge(ibes_actuals_query_0, how = 'left', on = ['cusip','statpers'])
    ibes['futepsgrowth'] = (ibes['fut_eps']-ibes['current_actual'])/abs(ibes['current_actual'])
    ibes.columns
    ibes['statpers_me'] = ibes['statpers'] + MonthEnd(0)
    
    ibes_gvkey = gvkey
    ibes_gvkey.drop_duplicates(subset='gvkey',inplace=True)
    ibes.rename(columns={'cusip':'IBES_cusip'},inplace=True)
        
    ibes = ibes.merge(ibes_gvkey, how = 'left', on ='IBES_cusip')
        
    for col in ['pends','pdicity','anndats', 'value','datadate','current_anndate',
                'statpers','fut_actual','fut_anndate']:
        del ibes[col]
    #ibes = ibes[['gvkey','IBES_cusip','statpers_me','ltg_eps','current_actual','fut_eps','futepsgrowth']]
    ibes['statpers_me'] = pd.to_datetime(ibes['statpers_me'])
    del ibes_actuals_query_0,ibes_actuals_query_1,ibes_actuals_query,ibes_merge1
    del comppricing_query, crsp_query, ccm_query
    ibes = ibes.drop_duplicates(subset=['gvkey','statpers_me'])
    del ibes['cusip']
    
    ##############################################################################
    '''Interested in the following annual variables'''

    #Fundamentals compustat annual
    sql_query = '''
    SELECT gvkey,cusip,conm, datadate, fyear, fyr, datafmt, indfmt, consol,popsrc, prcc_f, seq, ceq, 
    txditc, txdb, itcb, pstkrv, pstkl, pstk, csho, epsfx, epsfi, oprepsx, 
    opeps, ajex, ebit, spi, nopi, sale, ibadj, dvc, dvp, ib, oibdp, dp, oiadp, gp, 
    revt, cogs, pi, ibc, dpc, at, ni, ibcom, icapt, mib, ebitda, xsga, xido, xint, 
    mii, ppent, act, lct, dltt, dlc, che, invt, lt, rect, xopr, oancf, txp, txt, 
    ap, xrd, xad, xlr, capx
    FROM comp.funda
    WHERE indfmt='INDL' 
    and datafmt='STD' 
    and popsrc='D'
    and consol='C' 
    and datadate between '01/01/1975' and '12/31/2019' 
    and gvkey IN ({})'''.format(placeholders_gvkey)
    
    data_querya = conn.raw_sql(sql_query, params=(unique_gvkey))

    data_querya['datadate'] = pd.to_datetime(data_querya['datadate'])    
    #merge ibes and crsp_comp together
    data_querya = data_querya.merge(ibes,how='left', left_on=['datadate','gvkey'], right_on=['statpers_me','gvkey'])
    data_querya = data_querya.merge(crsp_comp,how='left', left_on=['datadate','gvkey'],
                                    right_on=['date','gvkey'])
        
    '''Interested in the following quarterly variables'''
    
    #Fundamentals compustat quarterly
    sql_query = '''
    SELECT gvkey, cusip, datadate, fyr, fyearq, fqtr, prccq, epsf12, epsfi12, ibadj12,
    oepsxq, oepsxy, oepf12, oeps12, seqq, ceqq, txditcq, txdbq, cshoq, epsfxq, epsfiq,  
    opepsq, ajexq, spiq, nopiq, saleq, saley, pstkq, ibadjq, dvy, dvpq, ibq, oibdpq, dpq, oiadpq, 
    revtq, cogsq, piq, dpcy, atq, niq, ibcomq, icaptq, mibq, xsgaq, xidoq, xintq, 
    miiq, ppentq, actq, lctq, dlttq, dlcq, cheq, invtq, ltq, rectq, xoprq, oancfy, txpq, txtq, 
    apq, xrdq, capxy, ibcy, dpy
    FROM comp.fundq
    WHERE indfmt IN ('INDL','FS')
    and datafmt='STD' 
    and popsrc='D'
    and consol='C' 
    and datadate between '01/01/1975' and '12/31/2019' 
    and gvkey IN ({})'''.format(placeholders_gvkey)
    
    data_queryq = conn.raw_sql(sql_query,  params=(unique_gvkey))
    data_queryq['datadate'] = pd.to_datetime(data_queryq['datadate'])
    
    data_queryq = data_queryq.merge(ibes,how='left', left_on=['datadate','gvkey'], right_on=['statpers_me','gvkey'])
    data_queryq = data_queryq.merge(crsp_comp,how='left', left_on=['datadate','gvkey'],
                                    right_on=['date','gvkey'])
    
    del crsp_comp
        
    dqq = data_queryq.drop_duplicates(subset=['gvkey','datadate'])
    dqq.sort_values(by=['gvkey','datadate'], ascending=True, inplace=True)
    for col in ['gvkey','gsector','ggroup','gind']:
        dqq[col] = dqq[col].astype(float)
        
    dqa = data_querya.drop_duplicates(subset=['gvkey','datadate'])
    dqa.sort_values(by=['gvkey','datadate'], ascending=True, inplace=True)
    for col in ['gvkey','gsector','ggroup','gind']:
        dqa[col] = dqa[col].astype(float)
    
    '''have to fix this bit up here'''
    del dqq['cusip_x']
    
    dqa['cusip']=dqa['cusip_x']
    del dqa['cusip_x']
    del dqa['cusip_y']
    
################################################################################ 
    # remove rows with no values
    cols = list(dqq.columns[6:])
    cols.remove('ajexq')
    dqq=dqq.dropna(subset=cols, how='all')
    len(data_queryq)
    #dqq.to_csv('US_financials_pre_ratios1.csv')
    dqq=dqq.drop_duplicates(subset=['datadate','gvkey'])
   # dqq = pd.read_csv('US_financials_pre_ratios1.csv').iloc[:,1:]

################################################################################
    '''Groupby quarterly missing data'''
    '''
    1. Fill with 0
    'dvrate', 'spiq', 'nopiq', 'spiq','pstkq',
    'dvy','dvpq','mibq','xidoq','miiq'
    
    2. Interpolate all other features
    
    3. Fill the rest of 'xrdq' with 0'''
    
    # dqq['miiq'].describe() sanity check
    
    #Step 1 - Groupby fill with 0 after first observation
    dqq[['gvkey','datadate','dvrate','nopiq', 'spiq','pstkq','dvy',
         'dvpq','mibq','xidoq','miiq']] = dqq[['gvkey','datadate','dvrate',
                                               'nopiq', 'spiq','pstkq','dvy',
                                               'dvpq','mibq','xidoq','miiq']].mask(
                                                   (dqq[['gvkey','datadate','dvrate','nopiq', 'spiq','pstkq','dvy',
         'dvpq','mibq','xidoq','miiq']].groupby('gvkey').ffill().notna() & dqq[['gvkey','datadate','dvrate','nopiq', 'spiq','pstkq','dvy',
         'dvpq','mibq','xidoq','miiq']].isna()).fillna(False), 0)    
           
    #sanity check before and after dqq['miiq'].describe()
    
    # Step 2 - Groupby interpolate linear
    
    dqq.columns
    inter_cols = ['prccq', 'epsf12',
           'epsfi12', 'ibadj12', 'oepsxq', 'oepsxy', 'oepf12', 'oeps12', 'seqq',
           'ceqq', 'txditcq', 'txdbq', 'cshoq', 'epsfxq', 'epsfiq', 'opepsq',
           'ajexq', 'saleq', 'saley','ibadjq',
           'ibq', 'oibdpq', 'dpq', 'oiadpq', 'revtq', 'cogsq', 'piq',
           'dpcy', 'atq', 'niq', 'ibcomq', 'icaptq','xsgaq',
           'xintq', 'ppentq', 'actq', 'lctq', 'dlttq', 'dlcq', 'cheq',
           'invtq', 'ltq', 'rectq', 'xoprq', 'oancfy', 'txpq', 'txtq', 'apq',
           'xrdq', 'capxy', 'ibcy', 'dpy', 'current_actual',
           'fut_eps', 'ltg_eps', 'futepsgrowth',
           'mcap_crsp', 'prc_crsp_unadj', 'prc_crsp_adj', 'gsector', 'ggroup',
           'gind', 'prc_comp_unadj', 'prc_comp_adj', 'cshom',
           'mcap_comp']
    
    for col in inter_cols:
        try:
            dqq[col] = dqq.groupby('gvkey')[col].apply(lambda x : x.interpolate())
        except ValueError:
            print(col+' Cannot impute this particular feature')
        print(col)

    # Step 3 - Fill the rest of xrdq with 0
    dqq[['gvkey','xrdq']]=dqq[['gvkey','xrdq']].mask(
        (dqq[['gvkey','xrdq']].groupby('gvkey').ffill().notna() & dqq[
            ['gvkey','xrdq']].isna()).fillna(False), 0)    

    dqq=dqq.rename(columns = {'cusip_y':'cusip'})
      
################################################################################

    #Trailing 12 months (Following WRDS)
    def ttm(var):
        ttm = var.rolling(min_periods=4, window=4).sum().round(1)
        return ttm
    
    #mean year (Following WRDS)
    def mean_year(var):
        mean_year = var.rolling(min_periods=4, window=4).mean().round(1)
        return mean_year
        
    data_querya = dqa.copy()
    data_queryq = dqq.copy()
    
    #del dqa, dqq
    data_queryq = data_queryq.groupby(by=['gvkey', 'cusip','datadate','conm'], as_index=False).sum(min_count=1)

    ####################   Financial ratios  ######################################
    
    '''define mktcap and price'''
    
    '''Annual mktcap'''

    data_querya['mktcap'] = data_querya['mcap_crsp']
    data_querya['mktcap'] = data_querya['mktcap'].fillna(data_querya['mcap_comp'])

    '''Quarterly mktcap'''

    data_queryq['mktcap'] = data_queryq['mcap_crsp']
    data_queryq['mktcap'] = data_queryq['mktcap'].fillna(data_queryq['mcap_comp'])

    '''Annual unadjusted price'''

    data_querya['price_unadj'] = data_querya['prc_crsp_unadj']
    data_querya['price_unadj'] = data_querya['price_unadj'].fillna(data_querya['prc_comp_unadj'])

    '''Quarterly unadjusted price'''

    data_queryq['price_unadj'] = data_queryq['prc_crsp_unadj']
    data_queryq['price_unadj'] = data_queryq['price_unadj'].fillna(data_queryq['prc_comp_unadj'])
    
    '''Annual adjusted price'''

    data_querya['price_adj'] = data_querya['prc_crsp_adj']
    data_querya['price_adj'] = data_querya['price_adj'].fillna(data_querya['prc_comp_adj'])

    '''Quarterly adjusted price'''

    data_queryq['price_adj'] = data_queryq['prc_crsp_adj']
    data_queryq['price_adj'] = data_queryq['price_adj'].fillna(data_queryq['prc_comp_adj'])

    '''oancfy to oancfq
    QTR 1, oanfcq=oanfcy
    
    For example, month3(Q1) = month6-(Q2)-month 9(Q3)-month12(Q4)'''
    data_queryq['oancfq'] = (data_queryq['oancfy']-data_queryq['oancfy'].shift(1))
    data_queryq.loc[data_queryq.fqtr == 1, 'oancfq'] = data_queryq['oancfy']
    
    '''capxy to capxq
    QTR 1, capxq=capxy
    
    For example, month3(Q1) = month6-(Q2)-month 9(Q3)-month12(Q4)'''
    data_queryq['capxq'] = (data_queryq['capxy']-data_queryq['capxy'].shift(1))
    data_queryq.loc[data_queryq.fqtr == 1, 'capxq'] = data_queryq['capxy']
    
    '''dpcy to dpcq
    QTR 1, dpcq=dpcy
    
    For example, month3(Q1) = month6-(Q2)-month 9(Q3)-month12(Q4)'''
    data_queryq['dpcq'] = (data_queryq['dpcy']-data_queryq['dpcy'].shift(1))
    data_queryq.loc[data_queryq.fqtr == 1, 'dpcq'] = data_queryq['dpcy']
    
    '''ibcy to ibcq
    QTR 1, ibcq=ibcy
    
    For example, month3(Q1) = month6-(Q2)-month 9(Q3)-month12(Q4)'''
    data_queryq['ibcq'] = (data_queryq['ibcy']-data_queryq['ibcy'].shift(1))
    data_queryq.loc[data_queryq.fqtr == 1, 'ibcq'] = data_queryq['ibcy']
    
    '''dpy to dpq
    QTR 1, dpq=dpy
    
    For example, month3(Q1) = month6-(Q2)-month 9(Q3)-month12(Q4)'''
    data_queryq['dpq'] = (data_queryq['dpy']-data_queryq['dpy'].shift(1))
    data_queryq.loc[data_queryq.fqtr == 1, 'dpq'] = data_queryq['dpy']
    
    '''dvy to dvq
    QTR 1, dvq=dvy
    
    For example, month3(Q1) = month6-(Q2)-month 9(Q3)-month12(Q4)'''
    data_queryq['dvq'] = (data_queryq['dvy']-data_queryq['dvy'].shift(1))
    data_queryq.loc[data_queryq.fqtr == 1, 'dvq'] = data_queryq['dvy']

    ###############################################################################    
    ###############################
    # Category 1: Capitalization #
    ###############################
    
    ###############################################################################
    #1.Capitalization Ratio (#DONE)
    '''Total Long-term Debt as a fraction of the sum of Total Long-term
    Debt, Common/Ordinary Equity and Preferred Stock'''
    
    data_querya['capital_ratioa'] = data_querya['dltt']/(data_querya['dltt'].fillna(0) \
    +data_querya['ceq'].fillna(0)+data_querya['pstk'].fillna(0)) #annual
        
    data_queryq['capital_ratioq'] = mean_year(data_queryq['dlttq']) \
    /(mean_year(data_queryq['dlttq'].fillna(0))+mean_year(data_queryq['ceqq'].fillna(0)\
                                                          + data_queryq['pstkq'].fillna(0))) #quarterly

    ###############################################################################
    #2.Common Equity/Invested Capital (#DONE) 
    '''Common Equity as a fraction of Invested Capital'''
    
    data_querya['equity_invcapa'] = data_querya['ceq']/data_querya['icapt'] #annual
    data_queryq['equity_invcapq']=mean_year(data_queryq['ceqq'])/mean_year(data_queryq['icaptq']) #quarterly

    ##############################################################################
    #3.Long-term Debt/Invested Capital (#DONE)
    '''Long-term Debt as a fraction of Invested Capital'''
    
    data_querya['debt_invcapa'] = data_querya['dltt']/data_querya['icapt'] #annual
    data_queryq['debt_invcapq'] = mean_year(data_queryq['dlttq'])/mean_year(data_queryq['icaptq']) #quarterly

    ##############################################################################
    #4.Total Debt/Invested Capital (#DONE)
    '''Total Debt (Long-term and Current) as a fraction of Invested Capital'''
    
    data_querya['totdebt_invcapa'] = (data_querya['dltt'].fillna(0)+data_querya['dlc'].fillna(0))\
        /data_querya['icapt'].fillna(0) #annual
        
    data_queryq['totdebt_invcapq'] = (mean_year(data_queryq['dlttq'].fillna(0))+mean_year(data_queryq['dlcq'].fillna(0))) \
    /mean_year(data_queryq['icaptq'].fillna(0)) #quarterly

    ##############################################################################
    
    ##########################
    # Category 2: Efficiency #
    ##########################
    
    ##############################################################################
    #5.Asset Turnover (#DONE)
    '''Sales as a fraction of the average Total Assets based on the most
    recent two periods'''
    
    data_querya['at_turna'] = data_querya['sale'] \
    / ((data_querya['at'].fillna(0)+data_querya['at'].fillna(0).shift(1))/2)#annual
    
    data_queryq['at_turnq'] = ttm(data_queryq['saleq'])/mean_year(data_queryq['atq']) #quarterly

    ##############################################################################
    #6.Inventory Turnover (#DONE)
    '''COGS as a fraction of the average Inventories based on the most
    recent two periods'''
    
    data_querya['inv_turna'] = data_querya['cogs'] \
    / ((data_querya['invt'].fillna(0)+data_querya['invt'].fillna(0).shift(1))/2)#annual

    data_queryq['inv_turnq'] = ttm(data_queryq['cogsq'])/mean_year(data_queryq['invtq'].replace(0,np.nan)) #quarterly   
    
    ##############################################################################
    #7.Payables Turnover (#DONE)
    
    '''COGS and change in Inventories as a fraction of the average of
    Accounts Payable based on the most recent two periods'''
    
    data_querya['pay_turna'] = (data_querya['cogs'].fillna(0) + data_querya['invt'].fillna(0).diff(1)) \
    / ((data_querya['ap'].fillna(0)+data_querya['ap'].fillna(0).shift(1))/2)#annual

    data_queryq['pay_turnq']=(ttm(data_queryq['cogsq'].fillna(0)) \
    + data_queryq['invtq'].fillna(0).diff(4))/mean_year(data_queryq['apq'].replace(0,np.nan)) #quarterly
        
    ##############################################################################
    #8.Receivables Turnover (#DONE)
        
    '''Sales as a fraction of the average of Accounts Receivables based on
    the most recent two periods'''
    
    data_querya['rect_turna']=data_querya['sale'] \
    / ((data_querya['rect'].fillna(0)+data_querya['rect'].fillna(0).shift(1))/2)#annual

    data_queryq['rect_turnq']=ttm(data_queryq['saleq'])/mean_year(data_queryq['rectq'].replace(0,np.nan)) #quarterly
    
    ##############################################################################
    #9.Sales/Stockholders Equity (#DONE) 
    
    '''Sales per dollar of total Stockholders’ Equity'''
    
    data_querya['sale_equitya']=data_querya['sale']/data_querya['seq'] #annual
    
    data_queryq['sale_equityq']=ttm(data_queryq['saleq'])/mean_year(data_queryq['seqq']) #quarterly
          
    ##############################################################################
    #10.Sales/Invested Capital (#DONE)
    
    '''Sales per dollar of Invested Capital'''
    
    data_querya['sale_invcapa']=data_querya['sale']/data_querya['icapt'] #annual
    
    data_queryq['sale_invcapq'] = ttm(data_queryq['saleq'])/mean_year(data_queryq['icaptq']) #quarterly

    ##############################################################################
    #11.Sales/Working Capital (#DONE) 
    
    '''Sales per dollar of Working Capital, defined as difference between
    Current Assets and Current Liabilities'''
    
    data_querya['sale_nwc']=data_querya['sale']/(data_querya['act'].fillna(0)\
                                                 -data_querya['lct'].fillna(0)) #annual

    data_queryq['sale_nwcq']=ttm(data_queryq['saleq'])/mean_year(data_queryq['actq'].\
                                                                  sub(data_queryq['lctq'], fill_value=0)) #quarterly

    ##############################################################################
    ####################################
    # Category 3: Financial Soundness #
    ####################################
    ##############################################################################
    #12.Inventory/Current Assets (#DONE) 
    
    '''Inventories as a fraction of Current Assets'''
    
    data_querya['invt_acta'] =data_querya['invt']/data_querya['act'] #annual
    
    data_queryq['invt_actq']=mean_year(data_queryq['invtq'])/mean_year(data_queryq['actq']) #quarterly

    ##############################################################################
    #13.Receivables/Current Assets (#DONE)
    
    '''Accounts Receivables as a fraction of Current Assets'''
    
    data_querya['rect_acta']=data_querya['rect']/data_querya['act'] #annual
    
    data_queryq['rect_actq']=mean_year(data_queryq['rectq'])/mean_year(data_queryq['actq']) #quarterly
    
    ##############################################################################
    #14.Free Cash Flow/Operating Cash Flow (#DONE)
    
    '''Free Cash Flow as a fraction of Operating Cash Flow, where Free
    Cash Flow is defined as the difference between Operating Cash Flow
    and Capital Expenditures'''
    
    '''Annual'''

    data_querya['ocf'] = data_querya['oancf']
    data_querya['ocf'] = data_querya['ocf'].fillna((data_querya['ib'].fillna(0)) \
    -((data_querya['act'].fillna(0).diff(1))-(data_querya['che'].fillna(0).diff(1)) \
    -(data_querya['lct'].fillna(0).diff(1))+(data_querya['dlc'].fillna(0).diff(1)) 
    +(data_querya['txp'].fillna(0).diff(1))-(data_querya['dp'].fillna(0))))

    data_querya['fcf_ocfa']=(data_querya['ocf'].fillna(0)-data_querya['capx'].fillna(0))\
        /data_querya['ocf'] #annual
            
    '''Quarterly'''

    data_queryq['ocf'] = ttm(data_queryq['oancfq'])
    data_queryq['ocf'] = data_queryq['ocf'].fillna(ttm(data_queryq['ibq'].fillna(0)) \
    -((data_queryq['actq'].fillna(0).diff(4))+(-data_queryq['cheq'].fillna(0).diff(4)) \
    +(-data_queryq['lctq'].fillna(0).diff(4))+(data_queryq['dlcq'].fillna(0).diff(4)) 
    +(data_queryq['txpq'].fillna(0).diff(4))+(-data_queryq['dpq'].fillna(0))))   
    
    data_queryq['fcf_ocfq'] = (data_queryq['ocf'].fillna(0)-ttm(data_queryq['capxq'].fillna(0)))\
        /(data_queryq['ocf']) #quarterly

    ##############################################################################
    #15.Operating CF/Current Liabilities (#DONE)
    '''Operating Cash Flow as a fraction of Current Liabilities'''
    
    data_querya['ocf_lcta']=data_querya['ocf']/data_querya['lct'] #annual
    
    data_queryq['ocf_lctq']=data_queryq['ocf']/mean_year(data_queryq['lctq']) #quarterly

    ##############################################################################
    #16.Cash Flow/Total Debt (#DONE)
    '''Operating Cash Flow as a fraction of Total Debt'''
    
    '''Annual'''

    data_querya['tda'] = data_querya['lt']
    data_querya['tda'] = data_querya['tda'].fillna(data_querya['dltt'].fillna(0)\
                                                   +data_querya['dlc'].fillna(0))
    
    data_querya['cash_debta']=data_querya['ocf']/data_querya['tda'] #annual
    
    '''Quarterly'''
    
    data_queryq['cash_debtq'] = data_queryq['ocf'] / mean_year(data_queryq['ltq']) #quarterly
      
    ##############################################################################
    #17.Cash Balance/Total Liabilities (#DONE)
    
    '''Cash Balance as a fraction of Total Liabilities'''
    
    data_querya['cash_lta']=data_querya['che']/data_querya['lt'] #annual
    
    data_queryq['cash_ltq']= mean_year(data_queryq['cheq'])/ mean_year(data_queryq['ltq']) #quarterly

    ##############################################################################
    #18.Cash Flow Margin (#DONE)
    '''Income before Extraordinary Items and Depreciation as a fraction of
    Sales'''
    
    '''Annual'''

    data_querya['incomea'] = (data_querya['ibc'].fillna(0)+data_querya['dpc'].fillna(0))
    data_querya['incomea'] = data_querya['incomea'].fillna((data_querya['ib'].fillna(0)\
    +data_querya['dp'].fillna(0)))

    data_querya['cfma']=data_querya['incomea']/data_querya['sale'] #annual
    
    '''Quarterly'''

    data_queryq['incomeq'] = ttm(data_queryq['ibcq'].fillna(0)+data_queryq['dpcq'].fillna(0))
    data_queryq['incomeq']=data_queryq['incomeq'].fillna(ttm(data_queryq['ibq'].fillna(0)) \
                                                             +data_queryq['dpq'].fillna(0))

    data_queryq['cfmq']=data_queryq['incomeq']/ttm(data_queryq['saleq']) #quarterly

    ##############################################################################
    #19.Short-Term Debt/Total Debt (#DONE)
    
    '''Short-term Debt as a fraction of Total Debt'''
    
    data_querya['short_debta']=data_querya['dlc']/(data_querya['dltt'].fillna(0)\
                                                   +data_querya['dlc'].fillna(0)) #annual
    
    data_queryq['short_debtq']=mean_year(data_queryq['dlcq'])/mean_year(data_queryq['dlttq'].fillna(0)\
                                                                        +data_queryq['dlcq'].fillna(0)) #quarterly
  
    ##############################################################################
    #20.Profit Before Depreciation/Current Liabilities (#DONE)
    
    '''Annual'''

    data_querya['opibda'] = (data_querya['oibdp'])
    data_querya['opibda'] = data_querya['opibda'].fillna(data_querya['sale'].fillna(0) \
                                                         -data_querya['xopr'].fillna(0))
            
    '''Quarterly'''

    data_queryq['opibd'] = ttm(data_queryq['oibdpq'])
    data_queryq['opibd'] = data_queryq['opibd'].fillna(ttm(data_queryq['saleq'].fillna(0)) \
                                                         -ttm(data_queryq['xoprq'].fillna(0)))
         
    '''Operating Income before D&A as a fraction of Current Liabilities'''
       
    data_querya['profit_lcta']=data_querya['opibda']/data_querya['lct'] #annual
    
    data_queryq['profit_lctq']=data_queryq['opibd']/mean_year(data_queryq['lctq']) #quarterly
   
    ##############################################################################
    #21.Current Liabilities/Total Liabilities (#DONE)
    
    '''Current Liabilities as a fraction of Total Liabilities'''
    
    data_querya['curr_debta']= data_querya['lct']/data_querya['lt'] #annual
    
    data_queryq['curr_debtq']=mean_year(data_queryq['lctq'])/mean_year(data_queryq['ltq']) #quarterly
    
    ##############################################################################
    #22.Total Debt/EBITDA (#DONE)
    
    '''Annual'''

    data_querya['ebitdaa'] = (data_querya['ebitda'])
    data_querya['ebitdaa'] = data_querya['ebitdaa'].fillna((data_querya['oibdp']))
    data_querya['ebitdaa'] = data_querya['ebitdaa'].fillna((data_querya['sale'].fillna(0)\
                                                            -data_querya['cogs'].fillna(0)\
                                                                -data_querya['xsga'].fillna(0)))
    
    '''Quarterly'''

    data_queryq['ebitdaq'] = ttm(data_queryq['oibdpq'])
    data_queryq['ebitdaq'] = data_queryq['ebitdaq'].fillna(ttm((data_queryq['saleq'].fillna(0).\
                                                                sub(data_queryq['cogsq'], fill_value=0)).\
                                                                    sub(data_queryq['xsgaq'], fill_value=0)))
         
    '''Gross Debt as a fraction of EBITDA'''
    
    data_queryq['debt_ebitda_num1'] = (data_queryq['oibdpq'].fillna(0))
    data_queryq['debt_ebitda_num2'] = ttm(data_queryq['debt_ebitda_num1'])

    data_querya['debt_ebitda']=(data_querya['dltt'].fillna(0)+data_querya['dlc'].fillna(0))/data_querya['ebitdaa'] #annual
    
    data_queryq['debt_ebitdaq']=mean_year(data_queryq['dlttq'].add(data_queryq['dlcq'], fill_value=0))\
    / data_queryq['ebitdaq'] #quarterly

    ##############################################################################
    #23.Long-term Debt/Book Equity (#DONE)
    '''Long-term Debt to Book Equity'''
    
    '''Annual'''
    data_querya['bea'] = (data_querya['seq'].fillna(0) + data_querya['txditc'].fillna(0)\
                          - data_querya['pstk'].fillna(0))
    data_querya['dltt_bea']=data_querya['dltt']/data_querya['bea'] #annual
    
    '''Quarterly'''
    data_queryq['beq'] = (data_queryq['seqq'].fillna(0) + data_queryq['txditcq'].fillna(0)\
                          - data_queryq['pstkq'].fillna(0))
    data_queryq['dltt_beq']=mean_year(data_queryq['dlttq'])/mean_year(data_queryq['beq']) #quarterly
 
    ##############################################################################
    #24.Interest/Average Long-term Debt (#DONE)
    
    '''Interest as a fraction of average Long-term debt based on most
    recent two periods'''
    
    data_querya['int_debta']=data_querya['xint']/((data_querya['dltt'].fillna(0)\
                                                 +data_querya['dltt'].fillna(0).shift(1))/2) #annual
  
     #quarterly
    data_queryq['int_debtq']=ttm(data_queryq['xintq'])/mean_year(data_queryq['dlttq'].replace(0, np.nan))
     
    ##############################################################################
    #25.Interest/Average Total Debt (#DONE)
     
    '''Interest as a fraction of average Total Debt based on most recent
    two periods'''
    
    data_querya['int_totdebta']=data_querya['xint']/(((data_querya['dltt'].fillna(0) \
    + data_querya['dltt'].fillna(0).shift(1))/2)+((data_querya['dlc'].fillna(0)\
                                                  +data_querya['dlc'].fillna(0).shift(1))/2)) #annual
    
    data_queryq['int_totdebtq']=ttm(data_queryq['xintq']) \
    /mean_year((data_queryq['dlttq'].replace(0, np.nan))+(data_queryq['dlcq'].replace(0, np.nan))) #quarterly

    ##############################################################################
    #26.Long-term Debt/Total Liabilities (#DONE)
    
    '''Long-term Debt as a fraction of Total Liabilities'''
    
    data_querya['lt_debta']=data_querya['dltt']/data_querya['lt'] #annual
    
    data_queryq['lt_debtq']=mean_year(data_queryq['dlttq'])/mean_year(data_queryq['ltq']) #quarterly
     
    ##############################################################################
    #27.Total Liabilities/Total Tangible Assets (#DONE)
    
    '''Total Liabilities to Total Tangible Assets'''
    
    data_querya['lt_ppenta']=data_querya['lt']/data_querya['ppent']
    
    data_queryq['lt_ppentq']=mean_year(data_queryq['ltq'])/mean_year(data_queryq['ppentq'].\
                                                                     replace(0,np.nan))

    ##############################################################################
    #########################
    # Category 3: Liquidity #
    #########################
    ##############################################################################
    #28.Cash Conversion Cycle (Days) (#DONE)
    
    '''Inventories per daily COGS plus Account Receivables per daily Sales
    minus Account Payables per daily COGS'''
    
    '''Annual'''
    data_querya['cash_conversiona'] = ((data_querya['invt'].fillna(0)+data_querya['invt'].fillna(0).shift(1))/2) \
    /(data_querya['cogs'].fillna(0)/365)+ ((data_querya['rect'].fillna(0)+data_querya['rect'].fillna(0).shift(1))/2) \
    /(data_querya['sale'].fillna(0)/365) - ((data_querya['ap'].fillna(0)+data_querya['ap'].fillna(0).shift(1))/2) \
    /(data_querya['cogs'].fillna(0)/365) #annual
        
    '''Quarterly'''   
    data_queryq['cash_conversionq']=(mean_year(data_queryq['invtq'])/(ttm(data_queryq['cogsq'].fillna(0))/365)) \
    +(mean_year(data_queryq['rectq'].fillna(0))/(ttm(data_queryq['saleq'].fillna(0))/365))\
        -(mean_year(data_queryq['apq'].fillna(0))/(ttm(data_queryq['cogsq'].fillna(0))/365)) #quarterly
    
    '''#if cash conversion is < 0, make NaN
    data_queryq['cash_conversionq'] = data_queryq['cash_conversionq'].\
        mask(data_queryq.cash_conversionq < 0, np.nan)'''
            
    ##############################################################################
    #29.Cash Ratio (#DONE)
    '''Cash and Short-term Investments as a fraction of Current Liabilities'''
    
    data_querya['cash_ratioa']=data_querya['che']/data_querya['lct'] #annual
    
    data_queryq['cash_ratioq']=mean_year(data_queryq['cheq'])/mean_year(data_queryq['lctq']) #quarterly
    
    ##############################################################################
    #30.Current Ratio (#DONE)
    
    '''Annual'''

    data_querya['currenta'] = (data_querya['act'])
    data_querya['currenta']=data_querya['currenta'].fillna((data_querya['che'].fillna(0)\
    +data_querya['rect'].fillna(0)+data_querya['invt'].fillna(0)))
    
    '''Quarterly'''

    data_queryq['currentq'] = mean_year(data_queryq['actq'])
    data_queryq['currentq']=data_queryq['currentq'].fillna(mean_year(data_queryq['cheq'].fillna(0)\
    +data_queryq['rectq'].fillna(0)+data_queryq['invtq'].fillna(0)))
         
    '''Current Assets as a fraction of Current Liabilities'''
    
    data_querya['curr_ratioa']=data_querya['currenta']/data_querya['lct'] #annual
    
    data_queryq['curr_ratioq']= data_queryq['currentq']/mean_year(data_queryq['lctq']) #quarterly
        
    ##############################################################################
    #31.Quick Ratio (Acid Test) (#DONE)
    
    '''Annual'''

    data_querya['cania'] = (data_querya['act'].fillna(0)-data_querya['invt'].fillna(0))
    data_querya['cania']=data_querya['cania'].fillna((data_querya['che'].fillna(0)\
    +data_querya['rect'].fillna(0)))
    
    '''Quarterly'''

    data_queryq['caniq'] = mean_year(data_queryq['actq'].fillna(0)-data_queryq['invtq'].fillna(0))
    data_queryq['caniq']=data_queryq['caniq'].fillna(mean_year(data_queryq['cheq'].fillna(0)\
    +data_queryq['rectq'].fillna(0)))
    
    '''Quick Ratio: Current Assets net of Inventories as a fraction of
    Current Liabilities'''
    
    data_querya['quick_ratioa'] = data_querya['cania']/data_querya['lct'] #annual
    
    data_queryq['quick_ratioq'] = data_queryq['caniq'] / mean_year(data_queryq['lctq']) #quarterly

    ##############################################################################
    #####################
    # Category 4: Other #
    #####################
    ##############################################################################
    #32.Accruals/Average Assets (#DONE)
     
    '''Accruals as a fraction of average Total Assets based on most recent
    two periods'''
    
    '''Annual'''

    data_querya['accra'] = (data_querya['oancf'].fillna(0)-data_querya['ib'].fillna(0))
    data_querya['accra']=data_querya['accra'].fillna(-((data_querya['act'].fillna(0).diff(1))\
    +(-data_querya['che'].fillna(0).diff(1))+(-data_querya['lct'].fillna(0).diff(1))\
        + (data_querya['dlc'].fillna(0).diff(1))+(data_querya['txp'].fillna(0).diff(1))\
            + (-data_querya['dp'].fillna(0))))
    
    '''Quarterly'''

    data_queryq['accrq'] = (data_queryq['oancfq']-data_queryq['ibq'])
    data_queryq['accrq']=data_queryq['accrq'].fillna(-((data_queryq['actq'].diff(4)) 
    +(-data_queryq['cheq'].diff(4))+(-data_queryq['lctq'].diff(4))+(data_queryq['dlcq'].diff(4)) \
    +(data_queryq['txpq'].diff(4))+(ttm(-data_queryq['dpq']))))
        
    data_queryq['accrq'] = (data_queryq['oancfq'].fillna(0)-data_queryq['ibq'].fillna(0))
    data_queryq['accrq']=data_queryq['accrq'].fillna(-((data_queryq['actq'].fillna(0).diff(4)) 
                                                       +(-data_queryq['cheq'].fillna(0).diff(4))\
                                                           +(-data_queryq['lctq'].fillna(0).diff(4))\
                                                           +(data_queryq['dlcq'].fillna(0).diff(4))\
                                                               +(data_queryq['txpq'].fillna(0).diff(4))\
                                                                   +(ttm(-data_queryq['dpq'].fillna(0)))))
        
    data_querya['accruala'] = data_querya['accra']/ ((data_querya['at'].fillna(0)\
                                                      + data_querya['at'].fillna(0).shift(1))/2) #annual
    
    data_queryq['at5'] = ((data_queryq['atq'].fillna(0) + data_queryq['atq'].fillna(0).shift(1)\
                           + data_queryq['atq'].fillna(0).shift(2) \
    + data_queryq['atq'].fillna(0).shift(3) + data_queryq['atq'].fillna(0).shift(4))/5)
    data_queryq['accrualq']= data_queryq['accrq']/data_queryq['at5'] #quarterly
     
    ##############################################################################
    #33.Research and Development/Sales (#DONE)
    
    '''R&D expenses as a fraction of Sales'''
    
    data_querya['rd_salea'] = (data_querya['xrd'] + 0) / data_querya['sale'] #annual
    
    data_queryq['rd_saleq']=ttm(data_queryq['xrdq'] + 0)/ttm(data_queryq['saleq']) #quarterly
 
    ##############################################################################
    #34.Avertising Expenses/Sales (#DONE) ONLY ANNUAL
    '''Advertising Expenses as a fraction of Sales'''
    
    data_querya['adv_salea'] = (data_querya['xad'] + 0)/data_querya['sale'] #annual
    
    ##############################################################################
    #35.Labor Expenses/Sales (#DONE) ONLY ANNUAL
    
    '''Labor Expenses as a fraction of Sales'''
    #NaN for everything (0 everything in ratios for apple)
    data_querya['staff_salea']= (data_querya['xlr'] + 0)/data_querya['sale'] #annual
    
    ##############################################################################
    #############################
    # Category 5: Profitability #
    #############################
    ##############################################################################
    #36.Effective Tax Rate (#DONE)
    
    '''Annual'''

    data_querya['incometaxa'] = (data_querya['pi'])
    data_querya['incometaxa'] = (data_querya['incometaxa']).fillna((data_querya['oiadp'].fillna(0)\
    -data_querya['xint'].fillna(0)+data_querya['spi'].fillna(0)+data_querya['nopi'].fillna(0)))
    
    '''Quarterly'''

    data_queryq['incometaxq'] = ttm(data_queryq['piq'])
    data_queryq['incometaxq'] = data_queryq['incometaxq'].fillna((ttm(data_queryq['oiadpq'].fillna(0))
    -ttm(data_queryq['xintq'].fillna(0))+ttm(data_queryq['spiq'].fillna(0))+ttm(data_queryq['nopiq'].fillna(0))))
    
    '''Income Tax as a fraction of Pretax Income'''
      
    data_querya['efftaxa'] = data_querya['txt'] / data_querya['incometaxa'] # annual
          
    data_queryq['efftaxq'] = ttm(data_queryq['txtq'])/data_queryq['incometaxq'] #quarterly

    ##############################################################################
    #37.Gross Profit/Total Assets (#DONE)
    
    '''Annual'''

    data_querya['grosspa'] = (data_querya['gp'])
    data_querya['grosspa'] = data_querya['grosspa'].fillna((data_querya['revt'].fillna(0)\
                                                            -data_querya['cogs'].fillna(0)))
    data_querya['grosspa'] = data_querya['grosspa'].fillna(data_querya['sale'].fillna(0)\
                                                           -data_querya['cogs'].fillna(0))
    
    '''Gross Profitability as a fraction of Total Assets'''
    data_querya['GProfa'] = data_querya['grosspa']/data_querya['at'] #annual
    
    data_queryq['GProfq'] = (ttm(data_queryq['revtq'].fillna(0)-data_queryq['cogsq'].fillna(0)))\
        /mean_year(data_queryq['atq']) #quarterly

    ##############################################################################
    #38.After-tax Return on Average Common Equity (#DONE)
    
    '''Annual'''

    data_querya['niafta'] = (data_querya['ibcom'])
    data_querya['niafta'] = (data_querya['niafta']).fillna((data_querya['ib'].fillna(0)\
                                                            -data_querya['dvp'].fillna(0)))
    '''Quarterly'''

    data_queryq['niaftq'] = ttm(data_queryq['ibcomq'])
    data_queryq['niaftq'] = data_queryq['niaftq'].fillna(ttm(data_queryq['ibq'].fillna(0)-data_queryq['dvpq'].fillna(0)))  
        
    '''Net Income as a fraction of average of Common Equity based on
    most recent two periods'''
      
    data_querya['aftret_eqa']=data_querya['niafta']/((data_querya['ceq'].fillna(0) \
    + data_querya['ceq'].fillna(0).shift(1))/2) #annual
        
    data_queryq['aftret_eqq']=data_queryq['niaftq']/mean_year(data_queryq['ceqq']).shift(1) #quarterly
        
    ##############################################################################
    #39.After-tax Return on Total Stockholders’ Equity (#DONE) 
    
    '''Net Income as a fraction of average of Total Shareholders’ Equity
    based on most recent two periods'''
    
    data_querya['aftret_equitya']=data_querya['ib']/((data_querya['seq'].fillna(0)\
                                                      + data_querya['seq'].fillna(0).shift(1))/2) #annual
    data_queryq['aftret_equityq']=ttm(data_queryq['ibq'])/mean_year(data_queryq['seqq']).shift(1) #quarterly

    ##############################################################################
    #40.After-tax Return on Invested Capital (#DONE)
    
    '''Net Income plus Interest Expenses as a fraction of Invested Capital'''
    
    '''Annual'''
    data_querya['aftret_invcapxa'] = (data_querya['ib'].fillna(0)+data_querya['xint'].fillna(0)\
                                      + data_querya['mii'].fillna(0)) \
    / (data_querya['icapt'].fillna(0).shift(1) + data_querya['txditc'].fillna(0).shift(1) \
       + (-data_querya['mib']).fillna(0).shift(1)) #annual
    
    '''Quarterly'''
    data_queryq['lagcapq'] = (data_queryq['icaptq'].fillna(0) + data_queryq['txditcq'].fillna(0)\
                              + (-data_queryq['mibq'].fillna(0)))
    data_queryq['lagicapt4']=(mean_year(data_queryq['lagcapq'])).shift(1)
    data_queryq['aftret_invcapxq']=ttm((data_queryq['ibq'].fillna(0)+data_queryq['xintq'].fillna(0)\
                                        +data_queryq['miiq'].fillna(0)))/data_queryq['lagicapt4'] #quartertly
    
    ##############################################################################
    #41.Gross Profit Margin (#DONE)
    
    '''Gross Profit as a fraction of Sales'''
    
    data_querya['gpma']= data_querya['grosspa']/data_querya['sale'] #annual
    
    data_queryq['gpmq']=(ttm(data_queryq['revtq']-data_queryq['cogsq']))/ttm(data_queryq['saleq']) #quarterly

    ##############################################################################
    #42.Net Profit Margin (#DONE)
                      
    '''Net Income as a fraction of Sales'''
    
    data_querya['npma']=data_querya['ib']/data_querya['sale'] #annual
    
    data_queryq['npmq']=ttm(data_queryq['ibq'])/ttm(data_queryq['saleq']) #quarterly
  
    ###############################################################################
    #43.Operating Profit Margin After Depreciation (#DONE)
    
    '''Annual'''

    data_querya['oiada'] = (data_querya['oiadp'])
    data_querya['oiada'] = (data_querya['oiada']).fillna((data_querya['oibdp'].fillna(0)\
                                                          -data_querya['dp'].fillna(0)))
    data_querya['oiada'] = (data_querya['oiada']).fillna((data_querya['sale'].fillna(0)\
                                                          -data_querya['xopr'].fillna(0)\
                                                              -data_querya['dp'].fillna(0)))
    data_querya['oiada'] = (data_querya['oiada']).fillna((data_querya['revt'].fillna(0)\
                                                          -data_querya['xopr'].fillna(0)\
                                                              -data_querya['dp'].fillna(0)))
    
    '''Quarterly'''

    data_queryq['oiadq'] = ttm(data_queryq['oiadpq'])
    data_queryq['oiadq'] = data_queryq['oiadq'].fillna(ttm(data_queryq['oibdpq'].fillna(0)\
                                                           -data_queryq['dpq'].fillna(0)))
    data_queryq['oiadq'] = data_queryq['oiadq'].fillna(ttm(data_queryq['saleq'].fillna(0)\
                                                           -data_queryq['xoprq'].fillna(0)\
                                                               -data_queryq['dpq'].fillna(0)))
        
    '''Operating Income After Depreciation as a fraction of Sales'''
    data_querya['opmada'] = data_querya['oiada'] / data_querya['sale'] #annual
    data_queryq['opmadq'] = data_queryq['oiadq'] / ttm(data_queryq['saleq']) #quarterly

    ###############################################################################
    #44.Operating Profit Margin Before Depreciation (#DONE)
    '''Annual'''

    data_querya['oibda'] = data_querya['oibdp']
    data_querya['oibda'] = data_querya['oibda'].fillna(data_querya['sale'].fillna(0)\
                                                       -data_querya['xopr'].fillna(0)) 
    data_querya['oibda'] = data_querya['oibda'].fillna(data_querya['revt'].fillna(0)\
                                                       -data_querya['xopr'].fillna(0))

    '''Quarterly'''

    data_queryq['opmbdq'] = ttm(data_queryq['oibdpq'])
    data_queryq['opmbdq'] = (data_queryq['opmbdq']).fillna(ttm(data_queryq['saleq'].fillna(0))\
                                                           -ttm(data_queryq['xoprq'].fillna(0)))

    '''Operating Income Before Depreciation as a fraction of Sales'''
    data_querya['opmbda'] = data_querya['oibda'] / data_querya['sale'] #annual
    data_queryq['opmbdq'] = data_queryq['opmbdq'] / ttm(data_queryq['saleq']) #quarterly
  
    ###############################################################################
    #45.Pre-tax Return on Total Earning Assets (#DONE)
    '''Annual'''

    data_querya['oiada'] = data_querya['oiadp']
    data_querya['oiada'] = data_querya['oiada'].fillna(data_querya['oiadp'].fillna(0)\
                                                       -data_querya['dp'].fillna(0))
    data_querya['oiada'] = data_querya['oiada'].fillna(data_querya['sale'].fillna(0)\
    -data_querya['xopr'].fillna(0)-data_querya['dp'].fillna(0))
    
    data_queryq['oiadq'] = ttm(data_queryq['oiadpq'])
    data_queryq['oiadq'] = data_queryq['oiadq'].fillna(ttm((data_queryq['oibdpq']).\
                                                       sub(data_queryq['dpq'], fill_value=0)))
    data_queryq['oiadq'] = data_queryq['oiadq'].fillna(ttm((data_queryq['saleq'].\
                                                           sub(data_queryq['xoprq'], fill_value=0)).\
                                                               sub(data_queryq['dpq'], fill_value=0)))
        
    '''Operating Income After Depreciation as a fraction of average Total
    Earnings Assets (TEA) based on most recent two periods, where
    TEA is defined as the sum of Property Plant and Equipment and
    Current Assets'''
    
    data_querya['pretret_earnata'] = data_querya['oiada'] / (((data_querya['ppent'].fillna(0).shift(1) \
    + data_querya['act'].fillna(0).shift(1)) + (data_querya['ppent'].fillna(0)\
                                                +data_querya['act'].fillna(0)))/2)  #annual
    
    #demoninator quarterly
    data_queryq['lagppent_alt4']=(mean_year(data_queryq['ppentq'].\
                                            add(data_queryq['actq'], fill_value=0))).shift(1)
        
    data_queryq['pretret_earnatq'] = data_queryq['oiadq'] / data_queryq['lagppent_alt4'] #quarterly

    ###############################################################################
    #46.Pre-tax return on Net Operating Assets (#DONE)
    '''Annual'''

    data_querya['oibda'] = data_querya['oiadp']
    data_querya['oibda'] = data_querya['oibda'].fillna(data_querya['oiadp'].fillna(0)\
                                                       -data_querya['dp'].fillna(0))
    data_querya['oibda'] = data_querya['oibda'].fillna(data_querya['sale'].fillna(0)\
    -data_querya['xopr'].fillna(0)-data_querya['dp'].fillna(0))
    data_querya['oibda'] = data_querya['oibda'].fillna(data_querya['revt'].fillna(0)\
    -data_querya['xopr'].fillna(0)-data_querya['dp'].fillna(0))
            
    '''Quarterly'''

    data_queryq['oibdq'] = (ttm(data_queryq['oibdpq']))
    data_queryq['oibdq'] = data_queryq['oibdq'].fillna(ttm(data_queryq['oiadpq'].\
                                                       sub(data_queryq['dpq'], fill_value=0)))
    data_queryq['oibdq'] = data_queryq['oibdq'].fillna(ttm((data_queryq['saleq'].\
                                                           sub(data_queryq['xoprq'], fill_value=0)).\
                                                               sub(data_queryq['dpq'], fill_value=0)))
        
    '''Operating Income After Depreciation as a fraction of average Net
    Operating Assets (NOA) based on most recent two periods, where
    NOA is defined as the sum of Property Plant and Equipment and
    Current Assets minus Current Liabilities'''
    
    '''Annual'''
    data_querya['noa1a'] = ((data_querya['ppent'].add(data_querya['act'], fill_value=0)).\
                            sub(data_querya['lct'], fill_value=0)).shift(1)
    data_querya['noa2a'] = ((data_querya['ppent'].add(data_querya['act'], fill_value=0)).\
                            sub(data_querya['lct'], fill_value=0))
    data_querya['noaa'] = (data_querya['noa1a'].fillna(0) + data_querya['noa2a'].fillna(0))/2
    
    data_querya['pretret_noa'] = data_querya['oibda'] / data_querya['noaa']  #annual
    
    '''Quarterly'''
    data_queryq['lagppent4'] = mean_year((data_queryq['ppentq'].add(data_queryq['actq'], fill_value=0)).\
                                         sub(data_queryq['lctq'], fill_value=0)).shift(1)
    data_queryq['pretret_noq'] = data_queryq['oibdq'] / data_queryq['lagppent4'] #quarterly

    ###############################################################################
    #47.Pre-tax Profit Margin (#DONE)
    '''Annual'''

    data_querya['pretaxia'] = data_querya['pi']
    data_querya['pretaxia'] = data_querya['pretaxia'].fillna(data_querya['oiadp'].fillna(0)\
    -data_querya['xint'].fillna(0)+data_querya['spi'].fillna(0)+data_querya['nopi'].fillna(0))
    
    '''Quarterly'''

    data_queryq['pretaxiq'] = ttm(data_queryq['piq'])
    data_queryq['pretaxiq'] = (data_queryq['pretaxiq']).fillna(ttm(data_queryq['oiadpq'].fillna(0))\
    -ttm(data_queryq['xintq'].fillna(0)+ttm(data_queryq['spiq'].fillna(0))\
         +ttm(data_queryq['nopiq'].fillna(0))))
    
    '''Pretax Income as a fraction of Sales'''
    
    data_querya['ptpma'] = data_querya['pretaxia'] / data_querya['sale'] #annual
    data_queryq['ptpmq']=data_queryq['pretaxiq']/ttm(data_queryq['saleq']) #quarterly
  
    ###############################################################################
    #48.Return on Assets (#DONE)
    '''Annual'''

    data_querya['oibdp_roa'] = data_querya['oibdp']
    data_querya['oibdp_roa'] = data_querya['oibdp_roa'].fillna(data_querya['sale'].fillna(0)\
    -data_querya['xopr'].fillna(0))
    data_querya['oibdp_roa'] = data_querya['oibdp_roa'].fillna(data_querya['revt'].fillna(0)\
    -data_querya['xopr'].fillna(0))
        
    '''Quarterly'''

    data_queryq['oiadq_roa'] = ttm(data_queryq['oibdpq'])
    data_queryq['oiadq_roa'] = (data_queryq['oiadq_roa']).fillna(ttm(data_queryq['saleq'].fillna(0))\
    -ttm(data_queryq['xoprq'].fillna(0)))
    
    '''Operating Income Before Depreciation as a fraction of average Total
    Assets based on most recent two periods'''
    
    data_querya['roaa'] = data_querya['oibdp_roa'] / ((data_querya['at'].fillna(0)\
                                                       +data_querya['at'].fillna(0).shift(1))/2) #annual
    data_queryq['roaq'] = data_queryq['oiadq_roa'] / (mean_year(data_queryq['atq'])).shift(1) #quarterly
 
    ###############################################################################
    #49.Return on Capital Employed (#DONE)
    '''Annual'''

    data_querya['earningsa'] = data_querya['ebit']
    data_querya['earningsa'] = data_querya['earningsa'].fillna(data_querya['sale'].fillna(0)\
    -data_querya['cogs'].fillna(0)-data_querya['xsga'].fillna(0)-data_querya['dp'].fillna(0))
            
    '''Quarterly'''

    data_queryq['earningsq'] = (ttm(data_queryq['oiadpq']))
    data_queryq['earningsq'] = (data_queryq['earningsq']).fillna(ttm(data_queryq['oibdpq'].fillna(0))\
    -ttm(data_queryq['dpq'].fillna(0)))
    data_queryq['earningsq'] = (data_queryq['earningsq']).fillna(ttm(data_queryq['saleq'].fillna(0))\
    -ttm(data_queryq['xoprq'].fillna(0))-ttm(data_queryq['dpq'].fillna(0)))    
    data_queryq['earningsq'] = (data_queryq['earningsq']).fillna(ttm(data_queryq['saleq'].fillna(0))\
    -ttm(data_queryq['cogsq'].fillna(0))-ttm(data_queryq['xsgaq'].fillna(0))-ttm(data_queryq['dpq'].fillna(0)))
        
    '''Earnings Before Interest and Taxes as a fraction of average Capital
    Employed based on most recent two periods, where Capital
    Employed is the sum of Debt in Long-term and Current Liabilities
    and Common/Ordinary Equity'''
    
    data_querya['capitala'] = (data_querya['dltt'].fillna(0)+data_querya['dltt'].fillna(0).shift(1) \
    +data_querya['dlc'].fillna(0)+data_querya['dlc'].fillna(0).shift(1)+data_querya['ceq'] \
    +data_querya['ceq'].fillna(0).shift(1))/2
    data_querya['rocea'] = data_querya['earningsa'] / data_querya['capitala'] #annual
    
    data_queryq['capitalq'] = mean_year(data_queryq['dlttq'].fillna(0)\
                                        +data_queryq['dlcq'].fillna(0)\
                                            +data_queryq['ceqq'].fillna(0)).shift(1)
    data_queryq['roceq'] = data_queryq['earningsq'] / data_queryq['capitalq'] #quarterly
   
    ###############################################################################
    #50.Return on Equity (#DONE) 
    
    '''Net Income as a fraction of average Book Equity based on most
    recent two periods, where Book Equity is defined as the sum of
    Total Parent Stockholders' Equity and Deferred Taxes and
    Investment Tax Credit'''
    
    '''Annual'''
    data_querya['bea'] = (data_querya['seq'].fillna(0) + data_querya['txditc'].fillna(0)\
                          - data_querya['pstk'].fillna(0))
    data_querya['roea']=data_querya['ib']/np.mean(data_querya['bea'].fillna(0)\
                                                  +data_querya['bea'].fillna(0).shift(1)) #annual
    
    '''Quarterly'''
    data_queryq['beq'] = (data_queryq['seqq'].fillna(0) + data_queryq['txditcq'].fillna(0)\
                          - data_queryq['pstkq'].fillna(0))
    data_queryq['roeq']=ttm(data_queryq['ibq'])/mean_year(data_queryq['beq']).shift(1) #quarterly
     
    ###############################################################################
                         
    ########################
    # Category 6: Solvency #
    ########################
    
    ###############################################################################
    #51.Total Debt/Equity (#DONE)
    '''Total Liabilities to Shareholders’ Equity (common and preferred)'''
    data_querya['de_ratioa']=(data_querya['dltt'].fillna(0)+data_querya['dlc'].fillna(0))\
        /(data_querya['ceq'].fillna(0)+ data_querya['pstk'].fillna(0)) #annual
    data_queryq['de_ratioq']=mean_year(data_queryq['ltq'])/mean_year(data_queryq['ceqq']+\
                                                                     data_queryq['pstkq']).fillna(0) #quarterly
             
    ###############################################################################
    #52.Total Debt/Total Assets (#DONE) 
    '''Total Debt as a fraction of Total Assets'''
    data_querya['debt_ata'] = (data_querya['dltt'].fillna(0)+data_querya['dlc'].fillna(0))\
        /data_querya['at'] #annual
    data_queryq['debt_atq']=mean_year(data_queryq['dlttq'].fillna(0)+data_queryq['dlcq'].fillna(0))\
        /mean_year(data_queryq['atq']) #quarterly
    
    ###############################################################################
    # 53.Total Liabilities/Total Assets (#DONE) 
    '''Total Liabilities as a fraction of Total Assets'''
    
    data_querya['lt_ata']=(data_querya['lt'])/data_querya['at'] #annual
    data_queryq['lt_atq']=mean_year(data_queryq['ltq'])/mean_year(data_queryq['atq']) #quarterly

    ###############################################################################
    #54.Total Debt/Capital (#DONE)
    '''Total Debt as a fraction of Total Capital, where Total Debt is defined
    as the sum of Accounts Payable and Total Debt in Current and Long-term 
    Liabilities, and Total Capital is defined as the sum of Total Debt
    and Total Equity (common and preferred)'''
    
    data_querya['debt_capitala']=(data_querya['ap'].fillna(0)+data_querya['dlc'].fillna(0)\
                                  +data_querya['dltt'].fillna(0)) \
    /(data_querya['ap'].fillna(0)+data_querya['dlc'].fillna(0)+data_querya['dltt'].fillna(0)) \
    +(data_querya['ceq'].fillna(0)+ data_querya['pstk'].fillna(0)) #annual
    
    data_queryq['debt_capitalq']=(mean_year(data_queryq['apq'].fillna(0)) \
    +mean_year(data_queryq['dlcq'].fillna(0)+data_queryq['dlttq'].fillna(0))) \
    /(mean_year(data_queryq['apq'].fillna(0))+ mean_year(data_queryq['dlcq'].fillna(0)\
                                                         +data_queryq['dlttq'].fillna(0)) \
      +mean_year(data_queryq['ceqq'].fillna(0)+data_queryq['pstkq'].fillna(0))) #quarterly

    ###############################################################################
    #55.After-tax Interest Coverage (#DONE)
    '''Multiple of After-tax Income to Interest and Related Expenses'''
    
    data_querya['intcova']=(data_querya['xint'].fillna(0)+data_querya['ib'].fillna(0))\
        /data_querya['xint'] #annual
    data_queryq['intcovq']=ttm(data_queryq['xintq'].fillna(0)+data_queryq['ibq'].fillna(0))\
        /ttm(data_queryq['xintq'].replace(0,np.nan)) #quarterly
   
    ###############################################################################
    #56.Interest Coverage Ratio (#DONE)
    
    '''Annual'''

    data_querya['earningsa'] = data_querya['ebit']
    data_querya['earningsa'] = data_querya['earningsa'].fillna(data_querya['oiadp'])
    data_querya['earningsa'] = data_querya['earningsa'].fillna(data_querya['sale'].fillna(0)\
    -data_querya['cogs'].fillna(0)-data_querya['xsga'].fillna(0)-data_querya['dp'].fillna(0))
            
    '''Quarterly'''

    data_queryq['earningsq'] = (ttm(data_queryq['oiadpq']))
    data_queryq['earningsq'] = (data_queryq['earningsq']).fillna(ttm(data_queryq['saleq'].fillna(0))\
    -ttm(data_queryq['cogsq'].fillna(0))-ttm(data_queryq['xsgaq'].fillna(0))-ttm(data_queryq['dpq'].fillna(0)))
    
    '''Multiple of Earnings Before Interest and Taxes to Interest and
    Related Expenses'''
            
    data_querya['intcov_ratioa']=data_querya['earningsa']/data_querya['xint'] #annual
    data_queryq['intcov_ratioq']=data_queryq['earningsq']/ttm(data_queryq['xintq'].replace(0,np.nan)) #quarterly
             
    ###############################################################################
    #########################
    # Category 7: Valuation #
    #########################
    ###############################################################################
    '''define pe_exi'''
    
    '''Annual'''
    data_querya['pe_exi']=data_querya['epsfx']/data_querya['ajex']
    
    '''Quarterly'''

    data_queryq['pe_exi']=data_queryq['epsf12']
    data_queryq['pe_exi']=data_queryq['pe_exi'].fillna(ttm(data_queryq['epsfxq']/data_queryq['ajexq']))
    
    ###############################################################################
    #57.Dividend Payout Ratio (#DONE) 
    '''Dividends as a fraction of Income Before Extra. Items'''
    
    '''Annual'''
    data_querya['dpra']=data_querya['dvc']/data_querya['ibadj'] #annual
    
    '''Quarterly'''

    data_queryq['dprq'] = ttm(data_queryq['dvq'].fillna(0)+data_queryq['dvpq'].fillna(0))\
        /data_queryq['ibadj12']
    data_queryq['dprq'] = data_queryq['dprq'].fillna(ttm(data_queryq['dvq'].fillna(0)\
                                                         +data_queryq['dvpq'].fillna(0))\
                                                     /ttm(data_queryq['ibadjq']))
       
    ###############################################################################
    #58.Forward P/E to 1-year Growth (PEG) ratio (#DONE)
    '''Price-to-Earnings, excl. Extraordinary Items (diluted) to 1-Year EPS
    Growth rate'''
    
    '''Price-to-Earnings, excl. Extraordinary Items (diluted)'''
    
    '''Annual'''
    data_querya['pe_exia']=data_querya['price_adj'].shift(-1)/data_querya['pe_exi']
    data_querya['PEG_1yrforward']=abs(data_querya['pe_exia'])/(data_querya['futepsgrowth']*100)
    
    '''Quarterly'''
    data_queryq['pe_exiq']=data_queryq['price_adj'].shift(-1)/data_queryq['pe_exi']
    data_queryq['PEG_1yrforward']=abs(data_queryq['pe_exiq'])/(data_queryq['futepsgrowth']*100)

    ###############################################################################
    #59.Forward P/E to Long-term Growth (PEG) ratio (#DONE)
    '''Price-to-Earnings, excl. Extraordinary Items (diluted) to Long-term
    EPS Growth rate'''
    
    '''Annual'''
    data_querya['pe_exia']=data_querya['price_adj'].shift(-1)/data_querya['pe_exi']
    data_querya['PEG_ltgforward']=abs(data_querya['pe_exia'])/(data_querya['ltg_eps'])
    
    '''Quarterly'''
    data_queryq['pe_exiq']=data_queryq['price_adj'].shift(-1)/data_queryq['pe_exi']
    data_queryq['PEG_ltgforward']=abs(data_queryq['pe_exiq'])/(data_queryq['ltg_eps'])
    
    ###############################################################################
    #60.Trailing P/E to Growth (PEG) ratio (#DONE)
    '''Price-to-Earnings, excl. Extraordinary Items (diluted) to 3-Year past
    EPS Growth'''
    '''Annual'''
    
    #3-yr past EPS growth
    data_querya['epsgrowthy1'] = data_querya['pe_exi']/data_querya['pe_exi'].shift(12)-1
    
    data_querya['epsgrowthy2'] = data_querya['pe_exi'].shift(12)/data_querya['pe_exi'].shift(24)-1
    data_querya['epsgrowthy3'] = data_querya['pe_exi'].shift(24)/data_querya['pe_exi'].shift(36)-1
    data_querya['eps3yr_growth'] = ((data_querya['epsgrowthy1'].fillna(0)+ 
                                    data_querya['epsgrowthy2'].fillna(0)+
                                    data_querya['epsgrowthy3'].fillna(0))/3)
    
    data_querya['PEG_trailing']=(data_querya['price_adj']/data_querya['pe_exi']) \
    / (100*data_querya['eps3yr_growth'])
    
    #in the definition of trailing PEG ratio in the line below PE_EXI is the adjusted diluted EPS excluding EI, not PE*/
    '''Quarterly'''  
    #3-yr past EPS growth
    data_queryq['epsgrowthy1'] = data_queryq['pe_exi']/data_queryq['pe_exi'].shift(4)-1
    data_queryq['epsgrowthy2'] = data_queryq['pe_exi'].shift(4)/data_queryq['pe_exi'].shift(8)-1
    data_queryq['epsgrowthy3'] = data_queryq['pe_exi'].shift(8)/data_queryq['pe_exi'].shift(12)-1
    data_queryq['eps3yr_growth'] = ((data_queryq['epsgrowthy1'].fillna(0)+ 
                                    data_queryq['epsgrowthy2'].fillna(0)+
                                    data_queryq['epsgrowthy3'].fillna(0))/3)
    
    data_queryq['PEG_trailing']=(data_queryq['price_adj']/data_queryq['pe_exi']) \
    /(100*data_queryq['eps3yr_growth'])
    
    ###############################################################################
    #61.Book/Market (#DONE)
    '''Book Value of Equity as a fraction of Market Value of Equity'''
    
    '''Annual'''

    data_querya['bea'] = (data_querya['seq'].fillna(0) + data_querya['txditc'].fillna(0)\
                          - data_querya['pstk'].fillna(0))
    data_querya['bea'] = data_querya['bea'].fillna((data_querya['seq'].fillna(0)\
                                                    + (data_querya['txdb'].fillna(0)\
                                                       +data_querya['itcb'].fillna(0)) \
                                                    - data_querya['pstk'].fillna(0)))
        
    data_querya['bma'] = data_querya['bea']/(data_querya['prcc_f']*data_querya['csho']) #annual
    
    '''Quarterly'''
    data_queryq['beq'] = (data_queryq['seqq'].fillna(0) + data_queryq['txditcq'].fillna(0)\
                          - data_queryq['pstkq'].fillna(0))
    data_queryq['bmq'] = data_queryq['beq']/(data_queryq['prccq']*data_queryq['cshoq']) #quarterly

    #mask NaN demominator with NaN in ratio column
    data_queryq['bmq'] = data_queryq['bmq'].\
        mask((data_queryq['prccq']*data_queryq['cshoq']).fillna(0).lt(0)) 
        
    ###############################################################################
    #62.Shillers Cyclically Adjusted P/E Ratio (#DONE)
    '''Multiple of Market Value of Equity to 5-year moving average of Net
    Income'''
    
    data_querya['capeia'] = data_querya['ib']
    #/*Compute the moving average income before EI over the last 5 years fo r Shiller's P/E Ratio*/
    #/*Calculate moving average income before EI over previous 20 quarters (5 years)*/
    
    #convert CAPEI=CAPEI / transformout=(MOVAVE 5 trimleft 3);
    data_querya['capeia'] = data_querya['capeia'].rolling(window=5).mean()
    data_querya['capeia2'] = data_querya['mktcap'].shift(-1)/data_querya['capeia'] 
    
    #fill ibq with niq-xidoq if ibq missing
    data_queryq['ibq'] = data_queryq['ibq'].fillna(data_queryq['niq'].fillna(0)\
                                                   +data_queryq['xidoq'].fillna(0))
    
    data_queryq['capeiq']=ttm(data_queryq['ibq']) #Shiller's P/E*/
    data_queryq['capeiq'] = data_queryq['capeiq'].rolling(window=20).mean()
    data_queryq['capeiq'] = data_queryq['mktcap'].shift(-1)/data_queryq['capeiq'] 

    ###############################################################################
    #63. Dividend Yield (#DONE)
    '''Indicated Dividend Rate as a fraction of Price'''
    
    '''Annual'''
    data_querya['divyielda']=data_querya['dvrate']/data_querya['price_unadj']
    
    '''Quarterly'''
    data_queryq['divyieldq']=mean_year(data_queryq['dvrate'])/mean_year(data_queryq['price_unadj'])
    data_queryq['divyieldq'] = data_queryq['divyieldq'].fillna(0)
    ###############################################################################
    #64. Enterprise Value Multiple (#DONE)
    '''Multiple of Enterprise Value to EBITDA'''
    
    '''Annual'''
    data_querya['eva'] = (data_querya['dltt'].fillna(0) + data_querya['dlc'].fillna(0)\
                          + data_querya['mib'].fillna(0) \
                              + data_querya['pstk'].fillna(0)\
                                  + (data_querya['prcc_f'].fillna(0)*data_querya['csho'].fillna(0)))

    data_querya['evma'] = data_querya['eva']/data_querya['ebitda']
    data_querya['evma'] = data_querya['evma'].fillna(data_querya['eva']/data_querya['oibdp'])
    data_querya['evma'] = data_querya['evma'].fillna(data_querya['eva'].fillna(0)\
    /(data_querya['sale'].fillna(0)-data_querya['cogs'].fillna(0)-data_querya['xsga'].fillna(0)))
                                                         
    '''Quarterly'''
    data_queryq['evq'] = (mean_year(data_queryq['dlttq'].fillna(0)) + mean_year(data_queryq['dlcq'].fillna(0)) \
    + mean_year(data_queryq['mibq'].fillna(0)) + mean_year(data_queryq['pstkq'].fillna(0)) \
    + mean_year(data_queryq['prccq'].fillna(0)*data_queryq['cshoq'].fillna(0)))

    data_queryq['evmq'] = data_queryq['evq']/ttm(data_queryq['oibdpq'])
    data_queryq['evmq'] = data_queryq['evmq'].fillna(data_queryq['evq']/(ttm(data_queryq['saleq'].fillna(0))\
    -ttm(data_queryq['cogsq'].fillna(0))-ttm(data_queryq['xsgaq'].fillna(0))))
  
    ###############################################################################
    #65. Price/Cash flow (#DONE)
    '''Multiple of Market Value of Equity to Net Cash Flow from Operating
    Activities'''
    
    '''Annual'''
    pcfa=data_querya['ocf']
    data_querya['pcfa'] = data_querya['mktcap'].shift(-1)/data_querya['oancf'] #annual
    
    '''Quarterly'''

    data_queryq['ocf'] = ttm(data_queryq['oancfq'])
    data_queryq['ocf'] = data_queryq['ocf'].fillna(ttm(data_queryq['ibq'].fillna(0)) \
                                                   -(data_queryq['actq'].fillna(0).diff(4))+(-data_queryq['cheq'].fillna(0).diff(4)) \
        +(-data_queryq['lctq'].fillna(0).diff(4))+(data_queryq['dlcq'].fillna(0).diff(4)) 
        +(data_queryq['txpq'].fillna(0).diff(4))+(-data_queryq['dpq'].fillna(0)))   
    
    data_queryq['pcfq']=data_queryq['mktcap'].shift(-1)/data_queryq['ocf'] #quarterly
    
    ###############################################################################
    #66. P/E (Diluted, Excl. EI) (#DONE)
    
    '''Price-to-Earnings, excl. Extraordinary Items (diluted)'''
    
    '''Annual'''
    data_querya['pe_exia']=data_querya['epsfx']/data_querya['ajex']
    data_querya['pe_exia']=(data_querya['price_adj'].shift(-1)/data_querya['pe_exia']) #annual
    
    '''Quarterly'''

    data_queryq['pe_exiq']=data_queryq['price_adj'].shift(-1)/data_queryq['epsf12']
    data_queryq['pe_exiq']=data_queryq['pe_exiq'].fillna(data_queryq['price_adj'].shift(-1)\
    /ttm(data_queryq['epsfxq']/data_queryq['ajexq']))
    
    ###############################################################################
    #67. P/E (Diluted, Incl. EI) (#DONE)
    '''Price-to-Earnings, incl. Extraordinary Items (diluted)'''
    
    pe_inca=data_querya['epsfi']/data_querya['ajex'] #annual
    
    data_querya['pe_inca']=data_querya['price_adj'].shift(-1)/(data_querya['epsfi']/data_querya['ajex'])
    
    '''Quarterly'''

    data_queryq['pe_incq']=data_queryq['price_adj'].shift(-1)/data_queryq['epsfi12']
    data_queryq['pe_incq']=data_queryq['pe_incq'].fillna(data_queryq['price_adj'].shift(-1) \
                                                         /ttm(data_queryq['epsfiq']/data_queryq['ajexq']))
     
    ###############################################################################
    #68. Price/Operating Earnings (Basic, Excl. EI) (#DONE)
    '''Price to Operating EPS, excl. Extraordinary Items (Basic)'''
    
    '''Quarterly'''

    data_queryq['pe_op_basicq']=data_queryq['price_adj'].shift(-1)/data_queryq['oeps12']
    data_queryq['pe_op_basicq']=data_queryq['pe_op_basicq'].fillna(data_queryq['price_adj'].shift(-1)\
    /ttm(data_queryq['opepsq']/data_queryq['ajexq']))
    
    ###############################################################################
    #69. Price/Operating Earnings (Diluted, Excl. EI) (#DONE) 
    '''Price to Operating EPS, excl. Extraordinary Items (Diluted)'''
    
    '''Quarterly'''

    data_queryq['pe_op_dilq']=data_queryq['price_adj'].shift(-1)/data_queryq['oepf12']
    data_queryq['pe_op_dilq']=data_queryq['pe_op_dilq'].fillna(data_queryq['price_adj'].shift(-1)\
    /ttm(data_queryq['oepsxq']/data_queryq['ajexq']))

    ############################################################################### 
    #70. Price/Sales (#DONE)
    '''Multiple of Market Value of Equity to Sales'''
    
    data_queryq['psq']=data_queryq['mktcap'].shift(-1)/ttm(data_queryq['saleq']) #quarterly
 
    ###############################################################################
    #71. Price/Book (#DONE)
    '''Multiple of Market Value of Equity to Book Value of Equity'''
    
    data_queryq['beq'] = (data_queryq['seqq'].fillna(0) + data_queryq['txditcq'].fillna(0)\
                          - data_queryq['pstkq'].fillna(0))
    data_queryq['ptbq']=data_queryq['mktcap'].shift(-1)/(data_queryq['beq'])
    
    ###############################################################################
    '''Fill inf with NaN'''
    
    data_queryq = data_queryq.replace([np.inf, -np.inf], np.nan)
    #data_queryq.to_csv('US_sample.csv',sep=',')
    #data_queryq = pd.read_csv('US_sample.csv',sep=',').iloc[:,1:]
    
    
    '''Some ratios need 0 imputing for some features that exhbi'''
    ###############################################################################
    
    #groupby for forward fill
    data_queryq = data_queryq.groupby(by=['gvkey','datadate', 'conm','cusip'], as_index=False).sum(min_count=1)
    data_queryq['qtrcheck'] = ttm(data_queryq['fqtr']) #qtr check as some data entered is not in quarters in WRDS
    
    #reindex and forward fill the quarterly data in each month
    data_queryq['qdate']=data_queryq['datadate']
    new_date_idx = pd.date_range(data_queryq.datadate.min(), data_queryq.datadate.max(), freq = 'M')
    data_queryq.set_index(['gvkey', 'datadate'], inplace=True)

    mux = pd.MultiIndex.from_product([data_queryq.index.levels[0], new_date_idx], 
                                 names=data_queryq.index.names)
    data_queryq=data_queryq.reindex(mux)
    data_queryq.reset_index(inplace=True)

    f = lambda x: x.ffill(limit=2) #lambda function to forward fill to qtr, only 2 after last obs
    data_queryq = data_queryq.groupby("gvkey")[data_queryq.columns].apply(f)
    data_queryq.loc[data_queryq['qtrcheck'] !=10, data_queryq.columns] = np.nan
    data_queryq = data_queryq[data_queryq.gvkey.notnull()].reset_index(drop=True)

    #create 2 month delayed date as WRDS does
    data_queryq['publicdate'] = data_queryq['datadate']+DateOffset(months=2) + MonthEnd(0)
    
    #round to 4 decimals
    data_queryq = data_queryq.round(4)

    #data_queryq.to_csv('US_sample.csv')
    
    ###############################################################################
    #ratios to be outputted 
    '''
    data_queryq = data_queryq[['gvkey','cusip','conm','qdate','publicdate','fyr','fyearq','fqtr',
                             'accrualq','aftret_eqq','aftret_equityq','aftret_invcapxq',
                             'at_turnq','bmq','capeiq','capital_ratioq','cash_conversionq',
                             'cash_debtq','cash_ltq','cash_ratioq','cfmq','curr_debtq',
                             'curr_ratioq','debt_atq', 'de_ratioq','debt_capitalq','debt_ebitdaq',
                             'debt_invcapq','divyieldq','dltt_beq','dprq','efftaxq',
                             'equity_invcapq','evmq','fcf_ocfq','gpmq','GProfq',
                             'gsector','ggroup','gind','int_debtq','int_totdebtq',
                             'intcov_ratioq','intcovq','inv_turnq','invt_actq',
                             'lt_atq','lt_debtq','lt_ppentq','mktcap','npmq',
                             'ocf_lctq','opmadq','opmbdq','pay_turnq','pcfq','pe_exiq',
                             'pe_incq','pe_op_basicq','pe_op_dilq','PEG_1yrforward',
                             'PEG_ltgforward','PEG_trailing','pretret_earnatq',
                             'pretret_noq','price_adj','price_unadj','profit_lctq',
                             'psq','ptbq','ptpmq','quick_ratioq','rd_saleq',
                             'rect_actq','rect_turnq','roaq','roceq','roeq','sale_equityq',
                             'sale_invcapq','sale_nwcq','short_debtq','totdebt_invcapq'
                             ]]   ''' 
    return data_queryq

del_col = ['cusip','conm','publicdate','fyr','fyearq','fqtr',
 'accrualq','aftret_eqq','aftret_equityq','aftret_invcapxq',
 'at_turnq','bmq','capeiq','capital_ratioq','cash_conversionq',
 'cash_debtq','cash_ltq','cash_ratioq','cfmq','curr_debtq',
 'curr_ratioq','debt_atq', 'de_ratioq','debt_capitalq','debt_ebitdaq',
 'debt_invcapq','divyieldq','dltt_beq','dprq','efftaxq',
 'equity_invcapq','evmq','fcf_ocfq','gpmq','GProfq',
 'gsector','ggroup','gind','int_debtq','int_totdebtq',
 'intcov_ratioq','intcovq','inv_turnq','invt_actq',
 'lt_atq','lt_debtq','lt_ppentq','mktcap','npmq',
 'ocf_lctq','opmadq','opmbdq','pay_turnq','pcfq','pe_exiq',
 'pe_incq','pe_op_basicq','pe_op_dilq','PEG_1yrforward',
 'PEG_ltgforward','PEG_trailing','pretret_earnatq',
 'pretret_noq','price_adj','price_unadj','profit_lctq',
 'psq','ptbq','ptpmq','quick_ratioq','rd_saleq',
 'rect_actq','rect_turnq','roaq','roceq','roeq','sale_equityq',
 'sale_invcapq','sale_nwcq','short_debtq','totdebt_invcapq'
 ]  

#data_queryq.drop([del_col],axis=1)

data_us= data_us[data_us.columns.difference(del_col)]
data_us['qdate'] = pd.to_datetime(data_us['qdate'])

data_queryq['qdate']
