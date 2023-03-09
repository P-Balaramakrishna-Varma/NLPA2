- Dependencies
    - Conda virtual environment to be used, dependencies present in "environment.yml".
    - ```conda create env --name 2020101024 --file enivronment.yml```


- pos_tagger.py 
    - It loads the "model_weights.pth" model and assings pos tags to sentences entered in terminal.
    - No command line arguments
    - To run ```python pos_tagger.py```
    - After running enter a sentence and press enter


- neural_tag.py
    - It contains code for data processing, model, training and evaluation.


- hyper_tunning.ipynb
    - A python notebook to help tune parameters.


Make sure ```UD_Englis-Atis``` directory is present in working directory.