# Salient Region Proposals for Object Detection

Code for this repository has been adapted from DeepSaliency which was written by Akwasi Akwaboah and Saliency-Based Dropout which was written by Shira Goldhaber-Gordon.

## Project overview:
Over the past few decades, many saliency models have been developed in an attempt to understand how the human brain assigns visual attention to a scene, and to predict human eye movements. We theorize that incorporating saliency into image-processing neural networks, which are also inspired by the human brain, could improve both accuracy and efficiency. In this project, we introduce a method to inform object detection models based on region proposal networks utilizing saliency information.

## File structure:
```
salRPN
├── datasets                         
│   └── README.md                    # instructions regarding dataset download and saliency generation
├── LICENSE
├── main_command_line_inputs.py      # for use with sbatch script. will call `run_experiment.py`
├── main.py                          # will call `run_experiment.py`
├── README.md
└── utils
    ├── config.yaml                  # config file for data paths. make sure the paths are correct for data download 
    ├── filter.py                    # stores filters used on saliency information
    ├── data_utils.py                # data loader function
    ├── wrapper.py                   # functions to create custom RPN
    ├── run.py                       # runs the training and testing epochs
    ├── run_experiment.py            # function with many parameters options, to be called in `main.py`
    ├── sal_utils.py                 # code for image signature saliency map generation
    └── trainer.py                   # defines training and evaluation methods and calls wrapper classes to setup model
```

## Wandb tracking:

Modify the code in salRPN/run.py to personalize wandb initialization, specifically the project-placeholder and entity-placeholder.
