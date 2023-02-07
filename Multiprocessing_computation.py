#   0owR71Mml_BDA_Assignment_1_PART_2

# import required libraries here

from pathlib import Path
from itertools import repeat
import pandas as pd
import time
from multiprocessing import Pool
import numpy as np
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

"""
In this exercise, we will utilize multiprocessing to speed-up computation of statistical estimates.

1. Task description. Given a dataset of age, we would ike to estimate standard errors for mean of the age 
and median of the age usinng bootstrap sampling. In bootstrap samping, we draw samples from a dataset with replacement.
2. Please read the description of each function to complete it.

3. Replace None with your code
"""


def get_boostrap_estimates_single_param(sample_size):
    """
    Generate a bootstrap sample but harcode df within the function
    so that we can use Pool.map() later
    :param sample_size: how many rows to sample from the input dataframe
    :returns mean and median of this particular bootstrap sample
    """

    # Load the age.csv file into a dataframe here (~ 2 lines)
    # by hard-coding the file name and using pandas to load it
    data=pd.read_csv('age.csv')
    # shuffle the dataframe by using pd.DataFrame.sample() with
    # n = df.size
    df = pd.DataFrame.sample(data, n=data.size)

    # Get bootstrap sample, ensure you sample with replacement
    df_sample = resample(df, replace=True, n_samples=sample_size, random_state=1)

    # return mean and median (~ 1 line)
    return np.mean(df_sample['age']), np.median(df_sample)


def get_boostrap_estimates_multiple_params(df, sample_size):
    """
    Generate a bootstrap sample
    :param df: input dataframe
    :param sample_size: how many rows to sample from the input dataframe
    :returns mean and median of this particular bootstrap sample
    """
    # this is same as function get_boostrap_estimates_single_param() above
    # but instead of hard-coding the dataframe, wee are passing it as an arg
    # so, please complete the function uding guidelines from get_boostrap_estimates_single_param()
    data = pd.DataFrame.sample(df, n=df.size)
    
    df_sample = resample(data, replace=True, n_samples=sample_size, random_state=1)
    
    return np.mean(df_sample['age']), np.median(df_sample)

def compute_std_err_sequentially(df1, size, repetitions):
    """
    We can estimate standard error of an estimate but using bootstrap sampling
    :param df: input dataframe
    :param size: how many rows to sample from the input dataframe
    :param repetitions: how many times to run the bootstrap sample
    :returns prints out the standard errors and running time
    """

    # create list to hold data
    data = []

    # set start time
    start = time.time()

    # Create a for loop to go through all repetitions and
    # get mean and median from the sample and put it in the
    # data list as a dictionary with keys 'mean' and 'median'
    # ~ 3 lines
    for i in range(repetitions):
        mean=np.mean(df1)
        median=np.median(df1)
        data.append({'mean':mean, 'median':median})

    # create a dataframe from data object above
    df = pd.DataFrame(data)

    # get standard error for mean and median
    SE_mean = np.mean(df1)-np.mean(df['mean'])
    SE_median = np.median(df1)-np.mean(df['median'])
    print('Standard error for mean: {}, Standard error for median: {}'.format(SE_mean, SE_median))

    # set endtime
    end = time.time()

    # calculate time taken in seconds
    time_taken = end-start
    print('Time taken running sequentially: {} seconds'.format(int(time_taken)))


def compute_std_err_parallel(df, size, repetitions, multiple=False):
    """
    Generate standard error estimates with multiprocessing
    :param: df: input dataframe
    :param: size: sample size for the bootstrap sample (maximum is df.size)
    :param: repetitions: How many times to run the bootstrap sample process
    :param: multiple: whether to run multiprocessing with a single parameter or multiple params
    """

    start = time.time()

    # use Pool to sete number of processors based on your computer specs
    processors = Pool(4)
    if multiple:
        message = "Time taken running using parallel processing with multiple parameters"

        # prepare params
        params = [[df, size]]
        # run either starmap or map on the processor object
        results = processors.starmap(get_boostrap_estimates_multiple_params, params)
    else:
        # follow similar instructions as above
        message = "Time taken running using parallel processing with a single parameter"
        params =[size]
        results = processors.map(get_boostrap_estimates_single_param, params)

    # create dataframe using results object
    df1 = pd.DataFrame(results)

    # calculate standard error for mean and median
    # follow instructions similar to above compute_std_err_sequentially() function
    SE_mean = np.mean(df['age'])-df1.iloc[0,0]
    SE_median = np.median(df['age'])-df1.iloc[0,1]
    print('Standard error for mean: {:.3f}, Standard error for median: {:.3f}'.format(SE_mean, SE_median))
    end = time.time()
    time_taken = end-start
    print('{} : {} seconds'.format(message, int(time_taken)))
    processors.close()


def main():
    # Replace line below with path to age.csv

    #the file age.csv have to be in the same folder thant the .py file
    #but due to the size of the file we were not able to load it on the drive using the form
    age_csv = Path.cwd().parents[0].joinpath('DATA', 'raw', 'age.csv')
    df_age = pd.read_csv('age.csv')

    # Set sample size to any number below or equal to df_age.shape[0]
    # for testing, set to smaller number (ee.g., 5,000, 000)
    sample_size = 5000000

    # This is the sample size for computing SE
    # for testing, you can sseet this to a small number (ee.g., 100)
    number_repetitions = 100

    # Get results when we run sequentially
    # ~ 1 line
    compute_std_err_sequentially(df_age, sample_size, number_repetitions)
    # Get results when we run in parallel with single param
    # ~ 1 line
    compute_std_err_parallel(df_age, sample_size, number_repetitions, multiple=False)
    # Get results when we run in parallel with multiple params
    compute_std_err_parallel(df_age, sample_size, number_repetitions, multiple=True)

if __name__ == '__main__':
    main()

