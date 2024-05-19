# survival of patients with sepsis predictor

## Discription
A random forest model to predict the survival of patients with sepsis with a simple user interface. The dataset used is [Sepsis Survival Minimal Clinical Records](https://archive.ics.uci.edu/dataset/827/sepsis+survival+minimal+clinical+records) specifically the primary cohort. The model reached a PR AUC of 0.967 using the primary cohort dataset.

## Install requirements
```
pip install ucimlrepo scikit-learn streamlit
```
## Run the code
make sure you're in the same directory that has the python files
```
streamlit run main.py
```
