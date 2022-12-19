# DengueAI Data Challenge
This repo stores some quick prototyping and solution code for `DengAI: Predicting Disease Spread` challenge. More info here: https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/page/80/

## Repo Structure
This repo is structured to have the following files and subdirectories:
- data 
- myproject
    - run.py
    - base.py
    - prophet_model.py
    - sarimax.py
    - utils
        - utils.py
- notebooks: show case EDA and intermediary modeling metrics and results
- auxiliary files: Dockerfile, poetry files, readme etc to assist with environmental alignment and run time executions
##  Run time
You can execute the codebase to reproduce the `submission_file.csv` on your local machine via docker by:
- first cloning the repo:
- cd into the cloned repository via your local terminal, and run:
`./run_models.sh`
