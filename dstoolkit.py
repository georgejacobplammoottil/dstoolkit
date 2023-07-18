# Helper functions for data analysis on pandas dataframes

# Imports
import pandas as pd
import numpy as np

# Test file
ds = pd.read_csv("/Users/georgeplammoottil/Documents/Projects/Stat101/ds_salaries.csv")
print(ds.head(5))

# 1. Function to describe a variable - detailed and summary
# Count of unique values
# List of distinct values if distinct values < 10
# Top and Bottom 5 values
# Univariate distribution
 
def means(ds,column, verbose, decimals):
    d='%.'+str(decimals)+'f'
    pd.set_option('display.float_format', lambda x: d % x)
    ser = ds[column].squeeze()
    if verbose==1:
        top5=ser.nlargest(5)
        bottom5=ser.nsmallest(5)
        unique_count = ser.nunique()
        unique_values = ser.unique()
        rows = len(ser)
        print(f"******** Variable description - {column} ********")
        print("\nSummary & Percentiles\n")
        if unique_count<10:
            print(f"{unique_count} Unique_values :", unique_values, " | Total rows : ",rows, "\n")
        else:
            print(f"There are {rows} rows and {unique_count} unique values for {column}\n")
        print(ser.describe(percentiles=[0.01,0.1,0.25,0.5,0.75,0.9,0.95,0.99]))
        print("\n")
        print("Bottom 5 values: ", bottom5, sep='\n')
        print("\n")
        print("Top 5 values: ", top5, sep='\n')
        print("\n")
        print("**************************************************\n")
    else:
        print(f"******** Variable description - {column} ********")
        print("\nSummary & Percentiles")
        print(ser.describe(percentiles=[0.25,0.5,0.75,0.95]))
        print("\n")
        print("**************************************************\n")

#Testing means outcome on continuous variables
means(ds,'salary',1,2)
means(ds, 'remote_ratio',1,2)



# 2. Function that creates bins of a variable for bivariate analysis
# Allow user to specify number of bins
# Check for distinct values available for binning
# Allow user to exclude specific values from X
# Account for tied values

def get_bucket(df, x_column, y_column, ):
    df1 = df.groupby(x_column).aggregate(
        Observations=(x_column, 'count'), 
        X_min=(x_column,'min'), 
        X_max=(x_column, 'max'),
        X_avg = (x_column,'mean'), 
        Y_avg = (y_column, 'mean'))
    
    df1['cumsum'] = df1['Observations'].cumsum()

    return df1

def bucket_label(df, x_column, no_of_bins):

    df = df.sort_values(by=x_column,ascending=True, na_position='first')
    totalobs = df['cumsum'].max()
    bucket_size = totalobs/no_of_bins
    df['bin'] = no_of_bins-1
    cum_bucket_last = df['cumsum'].min()

    for bin in range(no_of_bins):
        cum_bucket_max = (bin+1)*(totalobs/no_of_bins)
        df.loc[(cum_bucket_last<df['cumsum']) & (df['cumsum']<cum_bucket_max), 'bin'] = bin
        cum_bucket_last = df.loc[df['bin'] == bin, 'cumsum'].max()
    return df


def binner(ds, y_column, x_column, no_of_bins, exclusions):
    print("\n**************************************************\n")
    print(f"*** Bivariate table - {x_column} and {y_column} ****\n")
    # Keep relevant columns
    bin_ds = ds[[x_column, y_column]]

    # Apply exclusions for X
    for exclusion in exclusions:
        bin_ds = bin_ds[bin_ds[x_column] != exclusion]

    # Sort and Group rows based on value
    bin_ds_sorted = bin_ds.sort_values(by=x_column,ascending=True, na_position='first')
    bin_ds_distinct = get_bucket(bin_ds_sorted, x_column, y_column)
    
    # Handling repeated values - Check if distinct values < number of bins specified
    distinct_values = bin_ds[x_column].nunique()
    if distinct_values < no_of_bins:
        print(f" {distinct_values} unique values available to create {no_of_bins} bins")
        no_of_bins = distinct_values
        res = bin_ds_distinct.groupby(x_column).aggregate(
            Observations=('Observations', 'sum'), 
            min=('X_min','min'), 
            max=('X_max', 'max'),
            avg = ('X_avg','mean'), 
            yavg = ('Y_avg', 'mean'))
        res['cumsum'] = res['Observations'].cumsum()

    else: 
        # Handling repeated values - if distinct values>no of bins, roll up to distinct values
        bin_ds_bucket = bucket_label(bin_ds_distinct, x_column, no_of_bins)
        
        res = bin_ds_bucket.groupby('bin').aggregate(
            Observations=('Observations', 'sum'), 
            min=('X_min','min'), 
            max=('X_max', 'max'),
            avg = ('X_avg','mean'), 
            yavg = ('Y_avg', 'mean'))
        res['cumsum'] = res['Observations'].cumsum()
    
    print(res)
    print("\n**************************************************\n")


binner(ds, 'remote_ratio','salary',10,[])
binner(ds, 'salary','remote_ratio',10,[])


