import pandas as pd
import numpy as np
from functools import reduce

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import time
from os import walk
from bs4 import BeautifulSoup

###############################################################################

def CapIQlogin():
    download_dir = '' #enter
    chromedriver = '' #enter
    driver = webdriver.Chrome(chromedriver)
    driver.get("https://www.capitaliq.com/CIQDotNet/Login.aspx")
    username = driver.find_element_by_id("username")
    password = driver.find_element_by_id("password")
    submit = driver.find_element_by_id("myLoginButton")
    username.send_keys("") #enter cap iq username (email)
    password.send_keys("") #enter cap iq password
    submit.click()
    time.sleep(10)
    return driver

def SearchDownloadFinancials(driver, exchangeticker):
    
    '''Search and returns financial data for required companies'''
    
    search = driver.find_element_by_id("SearchTopBar")
    search.send_keys(exchangeticker)
    time.sleep(1)
    search.send_keys(Keys.ARROW_DOWN)
    search.send_keys(Keys.ENTER)
    #companysearch = driver.find_element_by_id("ciqSearchSearchButton").click()
    keystats = driver.find_element_by_id("ll_7_123_2083").click()
    viewall = driver.find_element_by_id("_keyFinSection__rangeSlider_viewAll").click()
    Usdollar = driver.find_element_by_xpath("//select[@id='_pageHeader_fin_dropdown_currency']/option[@value='160']").click()
    historical = driver.find_element_by_xpath("//select[@id='_pageHeader_fin_dropdown_conversion']/option[@value='0']").click()   
    Go = driver.find_element_by_id("_pageHeader_TopGoButton").click()
    downloadfinancials = driver.find_element_by_id("_pageHeader__excelReport").click()
    wait_for_files_to_download()

def check_if_download_folder_has_unfinished_files():
    
    '''Check if the financial data file is downloaded or not'''
    download_dir = '' #enter
    for (dirpath, dirnames, filenames) in walk(download_dir):
        return str(filenames)

def wait_for_files_to_download():
    
    ''' If not downloaded, wait for the file to download'''
    
    time.sleep(5)  # let the driver start downloading
    file_list = check_if_download_folder_has_unfinished_files()
    while 'Unconfirmed' in file_list or 'crdownload' in file_list:
        file_list = check_if_download_folder_has_unfinished_files()
        time.sleep(1)

if __name__ == '__main__':
    wait_for_files_to_download()  
    
def statementdata(Statement, ISBSCFSdate, dataframedate):
    
    '''Collects the required dates and data for financial statements'''
    
    headers = Statement.iloc[0]
    IS_df  = pd.DataFrame(Statement.values[1:], columns=headers)
    IS_df.set_index(ISBSCFSdate, inplace = True) #have quarterly historical cap
    ISintersect = IS_df.index.intersection(dataframedate.index)
    IS_required = IS_df.loc[ISintersect].reset_index() # Capitalisation dates same as IS, BS and CFS
    IS_required['Date'] = pd.to_datetime(IS_required['index']).apply(lambda x: x.strftime('%Y'))
    IS_required = IS_required.set_index(IS_required['Date'])
    IS_required = IS_required.iloc[:,1:-1]
    
    return IS_required

def createDataset(file):
    
    '''Creates the entire dataset with IS, BS, CFS, Ratios, Capital Structure 
    and Market Cap sheets into one DataFrame'''
    
    xls = pd.ExcelFile(file) #read in excel file
    #print(xls.sheet_names)        
    dfs = {sheet:xls.parse(sheet) for sheet in xls.sheet_names} # dict of sheets
    
    dates_to_use = datestouse(dfs['Income Statement']) #function to collect correct dates (no duplicates)
    
    '''Income Statement '''
    
    ISBSCFS = dfs['Income Statement'].iloc[13,1:].dropna() #IS line start in excel file
    ISBSCFSdate = pd.DataFrame([x[-11:] for x in ISBSCFS]) # Remove strings from date
    #ISBSCFSdate.columns
    ISBSCFSdate['Date'] = ISBSCFSdate.loc[:]
    ISBSCFSdate = ISBSCFSdate[ISBSCFSdate != 'urce Change'].dropna() #remove if urce Change error occurs
    ISBSCFSdate = ISBSCFSdate[ISBSCFSdate != 'Year\nChange'].dropna() #remove if urce Change error occurs
    ISBSCFSdate['Date'] = pd.to_datetime(ISBSCFSdate[0], infer_datetime_format=True) # set to datetime
    ISBSCFSdate['Date'] = ISBSCFSdate['Date'].apply(lambda x: x.strftime('%m-%d-%Y'))
    
    #retrieve income statement information and transpose and merge date index
    IS = dfs['Income Statement'].iloc[16:]
    NameIS = ['For the Fiscal Period Ending', 'Note:', 'Currency','NaN','Exchange Rate',
            'Conversion Method','Filing Date', 'Occasionally', 'Restatement Type',
            'Calculation Type', '\n', 'FYC', 'Compustat']
    for Names in NameIS:
        IS = IS[IS["Unnamed: 0"].str.contains(Names)==False] #NM in effective tax rate, not meaningful
    IS = IS.loc[:, ~(IS == 'DSC').any()] #remove if DSC error occurs
    IS = IS.loc[:, ~(IS == 'FYC').any()].T #remove if DSC error occurs
    IS = statementdata(IS, ISBSCFSdate['Date'], dates_to_use) # IS desired
    IS = IS.loc[~IS.index.duplicated(keep='first')]
    
    # retrieve company name, exchange and ticker
    companyname = dfs['Income Statement'].iloc[3,0].split(' (')[0]
    exc_tick = dfs['Income Statement'].iloc[3,0]
    char1, char2, char3 = '(',':',')'
    exchange = exc_tick[exc_tick.find(char1)+1 : exc_tick.find(char2)]
    ticker = exc_tick[exc_tick.find(char2)+1 : exc_tick.find(char3)]
    
    '''Balance Sheet'''
    BS = dfs['Balance Sheet'].iloc[16:] # BS start line in excel file
    NameBS = ['For the Fiscal Period Ending', 'Note:', 'Currency','NaN','Exchange Rate',
            'Conversion Method','Filing Date', 'Occasionally', 'Restatement Type',
            'Calculation Type', '\n', 'LIABILITIES', 'Supplemental Items', 'FYC',
            'Compustat']
    for Names in NameBS:
        BS = BS[BS["Unnamed: 0"].str.contains(Names)==False] #NM in effective tax rate, not meaningful
    BS = BS.loc[:, ~(BS == 'DSC').any()] #remove if DSC error occurs
    BS = BS.loc[:, ~(BS == 'FYC').any()].T #remove if DSC error occurs
    BS = statementdata(BS, ISBSCFSdate['Date'], dates_to_use) # BS desired
    BS = BS.loc[~BS.index.duplicated(keep='first')]

    '''Cash Flow Statement'''
    CFS = dfs['Cash Flow'].iloc[16:] # CFS start line in excel file
    NameCFS = ['For the Fiscal Period Ending', 'Note:', 'Currency','NaN','Exchange Rate',
            'Conversion Method','Filing Date', 'Occasionally', 'Restatement Type',
            'Calculation Type', '\n', 'Supplemental Items', 'FYC', 'Compustat']
    for Names in NameCFS:
        CFS = CFS[CFS["Unnamed: 0"].str.contains(Names)==False] #NM in effective tax rate, not meaningful
    CFS = CFS.loc[:, ~(CFS == 'DSC').any()] #remove if DSC error occurs
    CFS = CFS.loc[:, ~(CFS == 'FYC').any()].T #remove if DSC error occurs
    CFS = statementdata(CFS, ISBSCFSdate['Date'], dates_to_use) # CFS desired
    CFS = CFS.loc[~CFS.index.duplicated(keep='first')]

    '''Ratios'''
    Ratios = dfs['Ratios'].iloc[13:] # Ratios start line in excel file
    NameRatios = ['For the Fiscal Period Ending', 'Note:', 'Currency','NaN','Exchange Rate',
        'Conversion Method','Filing Date', 'Occasionally', 'Restatement Type',
        'Calculation Type', '\n', 'Supplemental Items', 'FYC', 'Compustat']
    for Names in NameRatios:
        Ratios = Ratios[Ratios["Unnamed: 0"].str.contains(Names)==False]
    Ratios = Ratios.loc[:, ~(Ratios == 'DSC').any()]#remove if DSC error occurs
    Ratios = Ratios.loc[:, ~(Ratios == 'FYC').any()].T #remove if DSC error occurs
    Rat = statementdata(Ratios, ISBSCFSdate['Date'], dates_to_use) # Ratio statement desired
    Rat = Rat.loc[~Rat.index.duplicated(keep='first')]
    Rat.columns = Rat.columns.str.rstrip('% ')
    Rat.columns = Rat.columns.str.lstrip() # Strip space from start of column name
    RATIOS = Rat # desired columns
    RATIOS = RATIOS[RATIOS.columns.dropna()] # remove columns whose column name is NaN

    '''Capital Structure Summary'''
    CSS = dfs['Capital Structure Summary'].iloc[16:] # CSS start in excel file
    CSS = CSS.loc[:, ~(CSS == 'FYC').any()] # Remove FYC error
    CSS = CSS[CSS.columns[1::2]].T #skip every second column, after skipping first    
    CSS = CSS.reset_index()
    CSS = CSS.iloc[:,1:] # all rows, start from second column
    CSScol = dfs['Capital Structure Summary'].iloc[16:].T
    CSS.columns = CSScol.iloc[0] # first column
    CSS = CSS.dropna(how='all')    #to drop if all values in the row are nan
    if len(dfs['Capital Structure Summary'].iloc[12,1:].dropna()) == 0: # CSS start in excel file)
        CapSumdate = dfs['Capital Structure Summary'].iloc[13,1:].dropna()
    else:
        CapSumdate = dfs['Capital Structure Summary'].iloc[12,1:].dropna()
    CapSumdate = pd.DataFrame([x[-11:] for x in CapSumdate]) # remove strings from date
    CapSumdate = CapSumdate.replace('Year Change', np.nan).dropna()
    CapSumdate = CapSumdate.replace('urce Change', np.nan).dropna()
    CapSumdate['Date'] = pd.to_datetime(CapSumdate[0], infer_datetime_format=True)
    CapSumdate['Date'] = CapSumdate['Date'].apply(lambda x: x.strftime('%m-%d-%Y'))
    CSS.set_index(CapSumdate['Date'], inplace = True)
    intersect = CSS.index.intersection(dates_to_use.index) #CSS desired
    CSS_requried = CSS.loc[intersect].reset_index() # Capitalisation dates same as IS, BS and CFS
    CSS_requried['Date'] = pd.to_datetime(CSS_requried['index']).apply(lambda x: x.strftime('%Y'))
    CSS_requried = CSS_requried.set_index(CSS_requried['Date'])
    TSD = CSS_requried['Total Senior Debt'] #Required column
    TSD = TSD.loc[~TSD.index.duplicated(keep='first')]

    '''Historical Capitalization'''
    Cap = dfs['Historical Capitalization'].iloc[16:] # start of line in excel file
    Cap = Cap.loc[:, ~(Cap == 'FYC').any()].T # remove FYC error
    Cap.columns = Cap.iloc[0]
    Cap = Cap[1:]
    Capdate = dfs['Historical Capitalization'].iloc[12,1:]
    Capdate = Capdate.replace('Fiscal\nYear\nChange', np.nan).dropna() #remove if string in date
    Capdate = pd.DataFrame(Capdate)
    Capdate['Date'] = Capdate
    Capdate['Date'] = Capdate['Date'].apply(lambda x: x.strftime('%m-%d-%Y'))
    Cap.set_index(Capdate['Date'], inplace = True) #have quarterly historical cap
    intersect = Cap.index.intersection(dates_to_use.index)
    Cap_requried = Cap.loc[intersect].reset_index() # Capitalisation dates same as IS, BS and CFS
    Cap_requried['Date'] = pd.to_datetime(Cap_requried['index']).apply(lambda x: x.strftime('%Y'))
    Cap_requried = Cap_requried.set_index(Cap_requried['Date'])
    if 'Tier 1 Capital' in Cap_requried.columns:
        CAP = Cap_requried[['Share Price', 'Shares Out.', 'Market Capitalization',
                        'Tier 1 Capital', 'Tier 2 Capital', 'Total Capital']]
    else:
            CAP = Cap_requried[['Share Price', 'Shares Out.', 'Market Capitalization',
                        '= Total Enterprise Value (TEV)', '= Total Capital']]
            req_col = {'= Total Enterprise Value (TEV)':'Total Enterprise Value',
                        '= Total Capital': 'Total Capital'} # column names changing with dict
            CAP.columns = [req_col.get(x, x) for x in CAP.columns] # map from dict, with list-comprehension and get operation
    
    CAP = CAP.loc[~CAP.index.duplicated(keep='first')]

    '''Merge all Statements into useable DataFrame'''
    
    data_frames = [IS, BS, CFS, TSD, RATIOS, CAP] # list of dataframes
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Date'],
                                                how='outer'), data_frames) 
    
    df_merged = df_merged.dropna(axis=1, how='all')
    df_merged['Ticker'] = ticker # merge into DataFrame
    df_merged['Exchange'] = exchange # merge into DataFrame
    df_merged['Company Name'] = companyname # merge into DataFrame

    df = df_merged.dropna(subset=['Total Revenue'])
    df.drop(list(df.filter(regex = '_x')), axis = 1, inplace = True) # remove _x from end of column name
    df.columns = df.columns.str.rstrip('_y') # strip suffix at the right end only of column name.
    df.columns = df.columns.str.lstrip() # strip the space at front of column name
        
    return df

def datestouse(dfIncomeStatement):
    
    ''' Creates the accurate dates to ensure there is no duplicates'''
    
    data = pd.DataFrame(dfIncomeStatement.iloc[13,1:]) #dates required for statements in excel file
    data.columns
    LTM = data[data.iloc[:,0].str.contains('LTM')] # find if LTM in dates
    LTM = [x[-11:] for x in LTM[13]] # remove from date column (LTM[13] name of column)
    LTM = pd.DataFrame(LTM) 
    
    dates_to_use = [] # create empty list to append the dates with no duplications
    
    if LTM.empty: # if no LTM in dates, append the date column of IS
        data = pd.DataFrame(dfIncomeStatement.iloc[13,1:])
        data = data[~data[13].str.contains('LTM')]
        data = [x[-11:] for x in data[13]]
        dfdate = pd.DataFrame(data) 
        dfdate = dfdate[dfdate != 'urce Change'].dropna() #remove if urce Change error occurs
        dfdate = dfdate[dfdate != 'Year\nChange'].dropna() #remove if urce Change error occurs
        dfdate['Date'] = pd.to_datetime(dfdate[0], infer_datetime_format=True)
        dfdate['Date_Hist_Comp'] = dfdate['Date'].apply(lambda x: x.strftime('%m-%d-%Y'))
        dfdate['Date'] = dfdate['Date'].apply(lambda x: x.strftime('%Y'))
        dfdate = dfdate.set_index('Date')
        dates_to_use.append(dfdate)
    else: # if there is LTM, remove duplicated dates and append correct dates
        LTM['Date'] = pd.to_datetime(LTM[0], infer_datetime_format=True)
        LTM['Date_Hist_Comp'] = LTM['Date'].apply(lambda x: x.strftime('%m-%d-%Y'))
        LTM['Date'] = LTM['Date'].apply(lambda x: x.strftime('%Y'))
        LTM = LTM.set_index('Date_Hist_Comp')
        data = pd.DataFrame(dfIncomeStatement.iloc[13,1:])
        data = data[~data[13].str.contains('LTM')]
        data = [x[-11:] for x in data[13]]
        dfdate = pd.DataFrame(data) 
        dfdate = dfdate[dfdate != 'urce Change'].dropna() #remove if urce Change error occurs
        dfdate = dfdate[dfdate != 'Year\nChange'].dropna() #remove if urce Change error occurs
        dfdate['Date'] = pd.to_datetime(dfdate[0], infer_datetime_format=True)
        dfdate['Date_Hist_Comp'] = dfdate['Date'].apply(lambda x: x.strftime('%m-%d-%Y'))
        dfdate['Date'] = dfdate['Date'].apply(lambda x: x.strftime('%Y'))
        dfdate = dfdate.set_index('Date')
        var = LTM['Date'][0] in dfdate.index
        if var == False:
            LTM = LTM.reset_index()
            LTM = LTM.set_index('Date')
            dates_to_use.append(pd.concat([dfdate, LTM]))
        else:
            dates_to_use.append(dfdate)
    
    df = dates_to_use[0]
    df = df.reset_index()
    df = df.set_index('Date_Hist_Comp')
          
    return df

def MergeAllDataFrames(list_alldf):
    
    '''Find all columns across all DataFrames so merge can work'''
    desiredlist = []
    for index, dataframe in enumerate(list_alldf):
        a = dataframe.columns.values.tolist()
        for column_name in a:
            if not column_name in desiredlist:
                desiredlist.append(column_name)
    
    '''Merge'''
    frames = []
    for index, dataframe in enumerate(list_alldf):
        df = pd.DataFrame(columns = desiredlist, index = dataframe.index)
        cols_to_use = df.columns.difference(dataframe.columns)
        df2 = pd.DataFrame(columns = cols_to_use, index = dataframe.index)
        dataframe = pd.concat([df2, dataframe], axis=1)
        dataframe = dataframe.loc[:,~dataframe.columns.duplicated()]
        dataframe = dataframe.reindex(sorted(dataframe.columns), axis=1)
        frames.append(dataframe)
    
    df = pd.concat(frames)
    
    return df

def getdata(file):
    xls = pd.ExcelFile(file) #read in excel file
    #print(xls.sheet_names)        
    dfs = {sheet:xls.parse(sheet) for sheet in xls.sheet_names} # dict of sheets
    
    dates_to_use = datestouse(dfs['Income Statement']) #function to collect correct dates (no duplicates)
    
    '''Income Statement '''
    
    ISBSCFS = dfs['Income Statement'].iloc[13,1:].dropna() #IS line start in excel file
    ISBSCFSdate = pd.DataFrame([x[-11:] for x in ISBSCFS]) # Remove strings from date
    #ISBSCFSdate.columns
    ISBSCFSdate['Date'] = ISBSCFSdate.loc[:]
    ISBSCFSdate = ISBSCFSdate[ISBSCFSdate != 'urce Change'].dropna() #remove if urce Change error occurs
    ISBSCFSdate = ISBSCFSdate[ISBSCFSdate != 'Year\nChange'].dropna() #remove if urce Change error occurs
    ISBSCFSdate['Date'] = pd.to_datetime(ISBSCFSdate[0], infer_datetime_format=True) # set to datetime
    ISBSCFSdate['Date'] = ISBSCFSdate['Date'].apply(lambda x: x.strftime('%m-%d-%Y'))
    
    #retrieve income statement information and transpose and merge date index
    IS = dfs['Income Statement'].iloc[16:]
    NameIS = ['For the Fiscal Period Ending', 'Note:', 'Currency','NaN','Exchange Rate',
            'Conversion Method','Filing Date', 'Occasionally', 'Restatement Type',
            'Calculation Type', '\n', 'FYC', 'Compustat']
    for Names in NameIS:
        IS = IS[IS["Unnamed: 0"].str.contains(Names)==False] #NM in effective tax rate, not meaningful
    IS = IS.loc[:, ~(IS == 'DSC').any()] #remove if DSC error occurs
    IS = IS.loc[:, ~(IS == 'FYC').any()].T #remove if DSC error occurs
    IS = statementdata(IS, ISBSCFSdate['Date'], dates_to_use) # IS desired
    IS = IS.loc[~IS.index.duplicated(keep='first')]
    
    # retrieve company name, exchange and ticker
    companyname = dfs['Income Statement'].iloc[3,0].split(' (')[0]
    exc_tick = dfs['Income Statement'].iloc[3,0]
    char1, char2, char3 = '(',':',')'
    exchange = exc_tick[exc_tick.find(char1)+1 : exc_tick.find(char2)]
    ticker = exc_tick[exc_tick.find(char2)+1 : exc_tick.find(char3)]
    
    '''Balance Sheet'''
    BS = dfs['Balance Sheet'].iloc[16:] # BS start line in excel file
    NameBS = ['For the Fiscal Period Ending', 'Note:', 'Currency','NaN','Exchange Rate',
            'Conversion Method','Filing Date', 'Occasionally', 'Restatement Type',
            'Calculation Type', '\n', 'LIABILITIES', 'Supplemental Items', 'FYC',
            'Compustat']
    for Names in NameBS:
        BS = BS[BS["Unnamed: 0"].str.contains(Names)==False] #NM in effective tax rate, not meaningful
    BS = BS.loc[:, ~(BS == 'DSC').any()] #remove if DSC error occurs
    BS = BS.loc[:, ~(BS == 'FYC').any()].T #remove if DSC error occurs
    BS = statementdata(BS, ISBSCFSdate['Date'], dates_to_use) # BS desired
    BS = BS.loc[~BS.index.duplicated(keep='first')]

    '''Cash Flow Statement'''
    CFS = dfs['Cash Flow'].iloc[16:] # CFS start line in excel file
    NameCFS = ['For the Fiscal Period Ending', 'Note:', 'Currency','NaN','Exchange Rate',
            'Conversion Method','Filing Date', 'Occasionally', 'Restatement Type',
            'Calculation Type', '\n', 'Supplemental Items', 'FYC', 'Compustat']
    for Names in NameCFS:
        CFS = CFS[CFS["Unnamed: 0"].str.contains(Names)==False] #NM in effective tax rate, not meaningful
    CFS = CFS.loc[:, ~(CFS == 'DSC').any()] #remove if DSC error occurs
    CFS = CFS.loc[:, ~(CFS == 'FYC').any()].T #remove if DSC error occurs
    CFS = statementdata(CFS, ISBSCFSdate['Date'], dates_to_use) # CFS desired
    CFS = CFS.loc[~CFS.index.duplicated(keep='first')]

    '''Ratios'''
    Ratios = dfs['Ratios'].iloc[13:] # Ratios start line in excel file
    NameRatios = ['For the Fiscal Period Ending', 'Note:', 'Currency','NaN','Exchange Rate',
        'Conversion Method','Filing Date', 'Occasionally', 'Restatement Type',
        'Calculation Type', '\n', 'Supplemental Items', 'FYC', 'Compustat']
    for Names in NameRatios:
        Ratios = Ratios[Ratios["Unnamed: 0"].str.contains(Names)==False]
    Ratios = Ratios.loc[:, ~(Ratios == 'DSC').any()]#remove if DSC error occurs
    Ratios = Ratios.loc[:, ~(Ratios == 'FYC').any()].T #remove if DSC error occurs
    Rat = statementdata(Ratios, ISBSCFSdate['Date'], dates_to_use) # Ratio statement desired
    Rat = Rat.loc[~Rat.index.duplicated(keep='first')]
    Rat.columns = Rat.columns.str.rstrip('% ')
    Rat.columns = Rat.columns.str.lstrip() # Strip space from start of column name
    RATIOS = Rat # desired columns
    RATIOS = RATIOS[RATIOS.columns.dropna()] # remove columns whose column name is NaN

    '''Capital Structure Summary'''
    CSS = dfs['Capital Structure Summary'].iloc[16:] # CSS start in excel file
    CSS = CSS.loc[:, ~(CSS == 'FYC').any()] # Remove FYC error
    CSS = CSS[CSS.columns[1::2]].T #skip every second column, after skipping first    
    CSS = CSS.reset_index()
    CSS = CSS.iloc[:,1:] # all rows, start from second column
    CSScol = dfs['Capital Structure Summary'].iloc[16:].T
    CSS.columns = CSScol.iloc[0] # first column
    CSS = CSS.dropna(how='all')    #to drop if all values in the row are nan
    if len(dfs['Capital Structure Summary'].iloc[12,1:].dropna()) == 0: # CSS start in excel file)
        CapSumdate = dfs['Capital Structure Summary'].iloc[13,1:].dropna()
    else:
        CapSumdate = dfs['Capital Structure Summary'].iloc[12,1:].dropna()
    CapSumdate = pd.DataFrame([x[-11:] for x in CapSumdate]) # remove strings from date
    CapSumdate = CapSumdate.replace('Year Change', np.nan).dropna()
    CapSumdate = CapSumdate.replace('urce Change', np.nan).dropna()
    CapSumdate['Date'] = pd.to_datetime(CapSumdate[0], infer_datetime_format=True)
    CapSumdate['Date'] = CapSumdate['Date'].apply(lambda x: x.strftime('%m-%d-%Y'))
    CSS.set_index(CapSumdate['Date'], inplace = True)
    intersect = CSS.index.intersection(dates_to_use.index) #CSS desired
    CSS_requried = CSS.loc[intersect].reset_index() # Capitalisation dates same as IS, BS and CFS
    CSS_requried['Date'] = pd.to_datetime(CSS_requried['index']).apply(lambda x: x.strftime('%Y'))
    CSS_requried = CSS_requried.set_index(CSS_requried['Date'])
    TSD = CSS_requried['Total Senior Debt'] #Required column
    TSD = TSD.loc[~TSD.index.duplicated(keep='first')]

    '''Historical Capitalization'''
    Cap = dfs['Historical Capitalization'].iloc[16:] # start of line in excel file
    Cap = Cap.loc[:, ~(Cap == 'FYC').any()].T # remove FYC error
    Cap.columns = Cap.iloc[0]
    Cap = Cap[1:]
    Capdate = dfs['Historical Capitalization'].iloc[12,1:]
    Capdate = Capdate.replace('Fiscal\nYear\nChange', np.nan).dropna() #remove if string in date
    Capdate = pd.DataFrame(Capdate)
    Capdate['Date'] = Capdate
    Capdate['Date'] = Capdate['Date'].apply(lambda x: x.strftime('%m-%d-%Y'))
    Cap.set_index(Capdate['Date'], inplace = True) #have quarterly historical cap
    intersect = Cap.index.intersection(dates_to_use.index)
    Cap_requried = Cap.loc[intersect].reset_index() # Capitalisation dates same as IS, BS and CFS
    Cap_requried['Date'] = pd.to_datetime(Cap_requried['index']).apply(lambda x: x.strftime('%Y'))
    Cap_requried = Cap_requried.set_index(Cap_requried['Date'])
    if 'Tier 1 Capital' in Cap_requried.columns:
        CAP = Cap_requried[['Share Price', 'Shares Out.', 'Market Capitalization',
                        'Tier 1 Capital', 'Tier 2 Capital', 'Total Capital']]
    else:
            CAP = Cap_requried[['Share Price', 'Shares Out.', 'Market Capitalization',
                        '= Total Enterprise Value (TEV)', '= Total Capital']]
            req_col = {'= Total Enterprise Value (TEV)':'Total Enterprise Value',
                        '= Total Capital': 'Total Capital'} # column names changing with dict
            CAP.columns = [req_col.get(x, x) for x in CAP.columns] # map from dict, with list-comprehension and get operation
    
    CAP = CAP.loc[~CAP.index.duplicated(keep='first')]

    '''Merge all Statements into useable DataFrame'''
    
    data_frames = [IS, BS, CFS, TSD, RATIOS, CAP] # list of dataframes
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Date'],
                                                how='outer'), data_frames) 
    
    df_merged = df_merged.dropna(axis=1, how='all')
    df_merged['Ticker'] = ticker # merge into DataFrame
    df_merged['Exchange'] = exchange # merge into DataFrame
    df_merged['Company Name'] = companyname # merge into DataFrame

    df = df_merged.dropna(subset=['Total Revenue'])
    df.drop(list(df.filter(regex = '_x')), axis = 1, inplace = True) # remove _x from end of column name
    df.columns = df.columns.str.rstrip('_y') # strip suffix at the right end only of column name.
    df.columns = df.columns.str.lstrip() # strip the space at front of column name
    
    return df
