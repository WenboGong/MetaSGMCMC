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
The model for Q and D are the following files stored in root of [BNN_MNIST/](BNN_MNIST/), which can be used directly.

    Q_state_batch_500_baseline_50D_70G_step_0.007_40ep_broad_0.2
    D_state_batch_500_baseline_50D_70G_step_0.007_40ep_broad_0.2
If you want to train new D and Q, run 

    Sampler_Training.py
    
with default settings. The trained model will be saved in [BNN_MNIST/tmp_model_save](BNN_MNIST/tmp_model_save/)

To evaluate the sampler, run 

    NetworkTopologyGeneralization.py
You can specify which sampler to run at the beginning of this file. 
Note: PSGLD is not included, you need to run 
    run_PSGLD.py
and change the BNN activation function to 'ReLU' in [run_PSGLD.py](BNN_MNIST/run_PSGLD.py#L60)
