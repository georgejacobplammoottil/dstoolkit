# Helper functions for data analysis on pandas dataframes

# Imports
import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

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
        if unique_count<20:
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

def assign_count_for_column_unique_values(df, col):
    df1 = df[[col]]
    df2 = df1.groupby(col).aggregate(observations = (col,'count'))
    df3 = pd.merge(df,df2, on=col, how='left')
    df4 = df3.sort_values(by=col,ascending=True, na_position='first')
    df4['cumsum'] = df4['observations'].cumsum()
    return df4

def assign_binlabel(df, cols, no_of_bins, exclusions, categorical):
    # Keep relevant columns
    if len(cols)==1:
        x_column = cols[0]
    else:
        y_column = cols[0]
        x_column = cols[1]
    
    bin_ds = df[cols]
    distinct_values = bin_ds[x_column].nunique()

    if bin_ds[x_column].dtype.kind not in 'biufc':
        bin_ds[x_column] = bin_ds[x_column].astype(str)
        cateogrical=1

    # Apply exclusions for X
    for exclusion in exclusions:
        bin_ds = bin_ds[bin_ds[x_column] != exclusion]

    res1 = assign_count_for_column_unique_values(bin_ds, x_column)

    # If no of buckets < unique count OR variable is categorical, binlabel = x_xolumn
    if distinct_values<no_of_bins or categorical==1:
        res1['binlabel']=res1[x_column]
        if len(cols)==2:
            aggregations = {x_column: ['count'], y_column: 'mean'}
            colnames = ['observations', 'y_mean']
        else:
            aggregations = {x_column: ['count']}
            colnames = ['observations']
    
    # If no of buckets greater than unique count, use logic
    else: 
        totalobs = res1['cumsum'].max()
        res1['binlabel'] = no_of_bins-1
        cum_bucket_last = res1['cumsum'].min()

        for bin in range(no_of_bins):
            cum_bucket_max = (bin+1)*(totalobs/no_of_bins)
            res1.loc[(cum_bucket_last<=res1['cumsum']) & (res1['cumsum']<cum_bucket_max), 'binlabel'] = bin
            cum_bucket_last = res1.loc[res1['binlabel'] == bin, 'cumsum'].max()
        
        if len(cols)==2:
            aggregations = {x_column: ['count','min','max','mean'], y_column: 'mean'}
            colnames = ['observations', 'x_min', 'x_max', 'x_mean', 'y_mean']
        else:
            aggregations = {x_column: ['count','min','max','mean']}
            colnames = ['observations', 'x_min', 'x_max', 'x_mean']
    
    return(res1, aggregations, colnames)

def roll_up(df, x_column, aggdict, res_col_names):
    df1 = df.groupby(x_column).aggregate(aggdict)
    df1.columns = res_col_names
    df1['cumsum'] = df1['observations'].cumsum()
    return df1

def bucketize(df,y_column, x_column, no_of_bins, exclusions, categorical):
    print("\n**************************************************\n")
    print(f"*** Bivariate table - {x_column} and {y_column} ****\n")
    cols = [y_column, x_column]
    result1 = assign_binlabel(df, cols, no_of_bins, exclusions, categorical)
    result2 = roll_up(result1[0],'binlabel', result1[1],result1[2])
    print(result2)
    print("\n**************************************************\n")
    return result2


#Testing binner outcome on continuous and string variables
bucketize(ds, 'remote_ratio', 'salary', 10, [], 0)
bucketize(ds, 'salary','remote_ratio',10,[],0)
bucketize(ds, 'salary','job_title',10,[],1)
bucketize(ds, 'remote_ratio','company_size',10,[],1)


# 3. Function to bootstrap samples from dataframe based on column values or random 
def samples(ds, partition, splits):
    if partition=='random':
        ds['splitindex'] = np.random.random(ds.shape[0])  
    else:
        ds['splitindex']= ds[partition]
        splits = ds['splitindex'].nunique()

    result = assign_binlabel(ds,['splitindex'], 10, [], 0)

    res = pd.merge(ds, result[0], left_index=True, right_index=True)
    res1=res.drop(columns=['splitindex_y'])
    res2=res1.rename(columns={"splitindex_x": "splitindex","binlabel": "partition"})

    return res2


# Testing splitting dataframe into 10 random samples
temp = samples(ds,'random',10)
means(ds, 'salary',0,0)
print(temp.groupby('partition').aggregate(obs = ('salary','count'), salary_sample_mean = ('salary','mean')))

# 4. Function to compare multiple dataframes
