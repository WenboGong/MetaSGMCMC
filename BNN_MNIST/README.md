# MNIST Classification Experiment
This folder contains scripts/results for MNIST classification results.

    BNN_MNIST: contains the raw data of MNIST (if not exists, create this folder)
    
    BNN_MNIST_Result: contains the plots
    
    Dataset_Gen_Long_Run: contains the results for Dataset generalization, Acc_*_* is the test error file, NLL_*_* is the negative log likelihood and All_*_* contain the results for individual runs.
    
    ReLU_Generalization_Long_Run: contains the results for Network Architecture Generalization.
    
    Sigmoid_Generalization_Long_Run: contains the results for Activation Function Generalization.
    
    tmp_model_save: folder to temporarily save the model. The model used in the paper are not from here, they are saved root folder of BNN_MNIST/
    
    BNN_*.py: The file contains the model/samper/training function definition.
    
    NetworkTopologyGeneralization.py: The evaluation file for Network Architecture generalization.
    PlotResults.py: Plot the figure, saved at BN_MNIST_Result folder.
    
    Sampler_Training.py: Train the sampler used for Network Architecture/ Activation function generalization.
    
    Sampler_Training_DataGen.py: Train the sampler for Dataset Generalization.
    
    SigmoidGeneralization.py: Evaluation file for Activation function generalization.
    
    long_run_Dataset_Gen.py: Evaluation file for Dataset Generalization.
    
    long_run_Dataset_Gen_plot.py: Post-Process scripts to generate results files in Dataset_Gen_Long_Run folder.
    
    long_run_plot_generate.py: Post-Process scripts to generate results files in ReLU_Generalization_Long_Run or Sigmoid_Generalization_Long_Run.
    
    long_run_plot_*.py: Post-Process scripts to generate results for specific sampler (Only needed for psgld which is not included in long_run_plot_generate.py).
    
    run_PSGLD.py: Run PSGLD sampler for Network Architecture / Activation function generalization.
## Network Architecture Generalization
The model for Q and D are the following files stored in root of [BNN_MNIST/](./), which can be used directly.

    Q_state_batch_500_baseline_50D_70G_step_0.007_40ep_broad_0.2
    D_state_batch_500_baseline_50D_70G_step_0.007_40ep_broad_0.2
If you want to train new D and Q, run 

    Sampler_Training.py
    
with default settings. The trained model will be saved in [BNN_MNIST/tmp_model_save](./tmp_model_save/)

To evaluate the sampler, run 

    NetworkTopologyGeneralization.py
You can specify which sampler to run at the beginning of this file. 

Note: PSGLD is not included, you need to run 

    run_PSGLD.py
and change the BNN activation function to 'ReLU' in [run_PSGLD.py](./run_PSGLD.py#L60) and corresponding lr [run_PSGLD.py](./run_PSGLD.py#L69)

To generate the results, you can run 

        long_run_plot_generate.py
for NNSGHMC/SGHMC/SGLD. You need to specify the location of stored samples generated in the previous step.

For PSGLD, run 

        long_run_plot_psgld.py
with correct activation function in [long_run_plot_psgld.py](./long_run_plot_psgld.py#L116)
## Activation Function Generalization
You don't need to train new samplers, use the sampler trained in Network Architecture Generalization. 

**Note: Before running the each of the following experiments, check if you change to the correct stored sample location/activation functions and learning rate.**

To evaluate the sampler, run 

        SigmoidGeneralization.py
with specified sampler in [SigmoidGeneralization.py](./SigmoidGeneralization.py#L59). The samples will be stored in [Sigmoid_Generalization_Long_Run](./Sigmoid_Generalization_Long_Run). Note: you need to change the locations of stored samples and activation function to 'Sigmoid' in [long_run_plot_generate.py](./long_run_plot_generate.py#L48).

For PSGLD, it is the same as Network Architecture Generalization, but need to change the learning rate and activation function to 'Sigmoid'.

## Dataset Generalization
The stored models for D and Q are 

        Q_state_batch_500_baseline_50D_70G_step_0.007_40ep_broad_0.2_datasetGen
        D_state_batch_500_baseline_50D_70G_step_0.007_40ep_broad_0.2_datasetGen
If you want to re-train the sampler, run 
        
        Sampler_Training_DataGen.py
The trained model will be in [tmp_model_save](./tmp_model_save)

To evaluate the sampler, run 

        long_run_Dataset_Gen.py
with specified sampler in [long_run_Dataset_Gen.py](./long_run_Dataset_Gen.py#L84)

To generate the results, run 

        long_run_Dataset_Gen_plot.py
with the correct folder location. 

The above results will be saved in [Dataset_Gen_Long_Run](./Dataset_Gen_Long_Run)


## Generate Plots
If you finish the above 3 experiments, you can run 

        PlotResults.py
to generate similar plots used in the paper. You need to specify the locations of stored results for all 3 experiments in [PlotResults.py](./PlotResults.py#L133)

## Possible bugs
The above experiment are runned with **PyTorch 0.4.0** with default GPU settings. However, in the early version of 0.4.0, the randperm() function are unspported by CUDA. So it will cause errors for Dataloader. You can solve it by manually change the randperm() function. See [[pytorch] randperm lacks CUDA implementation #6874](https://github.com/pytorch/pytorch/issues/6874)

