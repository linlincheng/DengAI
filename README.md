# DengueAI Data Challenge
This repo stores some quick prototyping and solution code for `DengAI: Predicting Disease Spread` challenge. More info here: https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/page/80/

## Repo Structure
This repo is structured to have the following files and subdirectories:
- data: stores training and test data files downloaded from DriveAI data portal
- myproject
    - run.py: main wrapper to load dataset, train models and get predictions all together using prophet models
    - base.py: base modeling class, with basic shared methods built in and modeling interfaces definitions
    - prophet_model.py: inherits from base model class, implements prophet model with optimizations
    - sarimax.py" inherits from base model class, implements sarimax models 
    - utils
        - utils.py
- notebooks: show case EDA and intermediary modeling metrics and results
- auxiliary files: Dockerfile, poetry files, readme etc to assist with environmental alignment and run time executions
##  Run time
You can execute the codebase to reproduce the `submission.csv` on your local machine via docker by:
- first cloning the repo:
- cd into the cloned repository via your local terminal, and run:
`./run_models.sh`
This will generate a `submission.csv` file in your locally cloned repository, comments and suggestions are welcomed!