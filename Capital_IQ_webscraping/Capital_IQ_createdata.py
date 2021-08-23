from Capital_IQ_webscraping_functions import *
import os 
import pandas as pd

download_dir = '' #enter
chromedriver = '' #enter
Companynames = '' #enter
df = pd.read_excel(Companynames) #read in excel file
company_list = df['Company'].tolist()

###############################################################################

'''Open browser, Login to CapIQ and download financial data'''
driver = CapIQlogin()
for company in company_list:
    SearchDownloadFinancials(driver, company)  
    
'''Obtain Excel files to import'''
os.chdir('') #enter
path = os.getcwd()
files = os.listdir(path)
files_xls = [f for f in files if f[-3:] == 'xls']
files_xls

'''Read in all Excel files and save all to list'''
dataframes[2]
dataframes = []
for file in files_xls:
    df = createDataset(file)
    dataframes.append(df)

'''Merge all DataFrames in list into one DataFrame'''
Merged_df = MergeAllDataFrames(dataframes)
Namedf = Merged_df['Company Name']
Merged_df.drop(labels=['Company Name'], axis=1,inplace = True)
Merged_df.insert(0,'Company Name', Namedf) # move company name column to second
Tickerdf = Merged_df['Ticker'] 
Merged_df.drop(labels=['Ticker'], axis=1,inplace = True)
Merged_df.insert(0,'Ticker', Tickerdf) # move ticker column to front

'''Output to excel'''
Merged_df.to_csv('Mergeddf.csv',sep = ',')
