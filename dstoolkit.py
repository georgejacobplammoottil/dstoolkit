# Helper functions for data analysis

# Imports
import pandas as pd
import numpy as np

# Test file
ds = pd.read_csv("/Users/georgeplammoottil/Documents/Projects/Stat101/ds_salaries.csv")
ds.head()

# Univariate distribution 
def means(ds,column, verbose, decimals):
    d='%.'+str(decimals)+'f'
    pd.set_option('display.float_format', lambda x: d % x)
    ser = ds[column].squeeze()
    if verbose==1:
        top5=ser.nlargest(5)
        bottom5=ser.nsmallest(5)
        print(f"******** Variable description- {column} ********")
        print("\nSummary & Percentiles")
        print(ser.describe(percentiles=[0.01,0.1,0.25,0.5,0.75,0.9,0.95,0.99]))
        print("\n")
        print("Bottom 5 values: ", bottom5, sep='\n')
        print("\n")
        print("Top 5 values: ", top5, sep='\n')
        print("\n")
        print("**************************************************")
    else:
        print(f"******** Variable description- {column} ********")
        print("\nSummary & Percentiles")
        print(ser.describe(percentiles=[0.25,0.5,0.75,0.95]))
        print("\n")
        print("**************************************************")


#Testing
means(ds,'salary',1,0)
means(ds,'salary',0,0)

# Create bins of a variable
