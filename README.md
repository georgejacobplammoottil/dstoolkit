# dstoolkit - Helper functions for data exploration

Current functionality

## means() - Function to describe a variable - detailed and summary
1. Count of unique values
2. List of distinct values if distinct values < 10
3. Top and Bottom 5 values
4. Univariate distribution with percentiles
5. Verbose for all percentiles and default for quartiles

## binner() - Function to get a bivariate summary
1. Allows exclusions
2. Keeps tied values in same bucket
3. Provides averages, bucket definition and observation count
4. Works for string categorical variables
