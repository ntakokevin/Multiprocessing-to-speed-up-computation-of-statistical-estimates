# Multiprocessing-to-speed-up-computation-of-statistical-estimates
This project focuses on accelerating the computation of statistical estimates in the context of big data.

The task at hand involves working with a dataset containing age records. The goal is to estimate the standard errors for the mean and median age using bootstrap sampling. In bootstrap sampling, samples are drawn from the dataset with replacement.

To tackle the computational challenges posed by the large dataset, the project adopts multiprocessing techniques. Multiprocessing involves utilizing multiple processors or cores to perform computations concurrently, thereby significantly reducing the processing time.

The project comprises several functions that need to be completed. These functions are designed to facilitate the estimation process, including tasks such as drawing bootstrap samples, calculating means and medians, and computing the standard errors.

Before executing the code, it is necessary to extract the "age.csv" file from the compressed "age.rar" file, which contains the dataset.

By leveraging multiprocessing, the project aims to expedite the computation of statistical estimates. This approach allows for parallel processing of the bootstrap sampling procedure, leading to faster and more efficient calculations of standard errors for the mean and median age. The project is particularly suited for scenarios involving big data, where traditional single-processor computations may be time-consuming or unfeasible.

Overall, the project presents an opportunity to apply multiprocessing techniques to accelerate statistical estimations, enabling faster insights and analysis on large datasets
