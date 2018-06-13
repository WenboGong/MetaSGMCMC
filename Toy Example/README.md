# Toy Example
## Introduction
This example corresponds to the toy experiment in the Meta-learning for SG-MCMC paper.
## Setup
    Sampler_Training.py
will train the meta sampler on 10D uncorrleated Gaussian with ranomly generated diagonal covariance matrix. The generated covariance is controlled

    --cov_range
The training script can be running with default settings.

    Test_Sampler.py
    
This scripts will run the meta sampler and SGHMC on 20D correlated Gaussian with randomly generated covariance. To reproduce the result, the model parameters for meta sampler are 

    saveModel/Q_state_100_dim_10_clamp_5.0_range_6.0_time_20180413-1226
    saveModel/D_state_100_dim_10_clamp_5.0_range_6.0_time_20180413-1226
and the test covariance matrix are 

    /TestResult/Cov_rand_1_scale_0.6
    /TestResult/Cov_rand_2_scale_0.6
    /TestResult/Cov_rand_3_scale_0.5
    /TestResult/Cov_rand_4_scale_0.5
The test file can run with default settings.

## Note
If NaN occurs during training or evaluation, reduce the clamp value for Q/reduce the step size/ initialized value closer to mean of test.

The above files are runned on PyTorch 0.4.0
