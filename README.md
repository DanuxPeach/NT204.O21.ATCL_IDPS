# NT204.O21.ATCL_IDPS

## Introduction
This project focuses on developing an Intrusion Detection and Prevention System (IDPS) using **RTIDS**, a robust transformer-based approach for network intrusion detection. The model is trained and evaluated on multiple datasets, including **CICIDS 2017, CICIDS 2018, and CICIDS 2019**.

## Dataset
The project utilizes the following datasets:
- **CICIDS 2017**: A dataset containing realistic attack scenarios and normal network traffic.
- **CICIDS 2018**: A dataset with more diverse attack types and improved traffic patterns.
- **CICIDS 2019**: A more recent version with enhanced labeling and additional attack vectors.

## Methodology
1. **Data Preprocessing**: Cleaning and normalizing data to ensure consistency.
2. **Feature Engineering**: Selecting relevant network features for training the model.
3. **Model Training**: Using RTIDS to train a transformer-based deep learning model.
4. **Evaluation**: Assessing model performance using accuracy, precision, recall, and F1-score.

## Dependencies
To set up the environment, install dependencies using:
```sh
pip install -r requirements.txt
```
Alternatively, use Conda:
```sh
conda env create -f environment.yml
```
## Usage
Run the preprocessing step:
```sh
python datasetPreprocessing.ipynb
```
Train the model and Evaluate:
```sh
python modelTraining.ipynb
```

## Results
The model is expected to achieve high detection accuracy across different attack types while minimizing false positives.

## References
This project is based on the paper:
"RTIDS: A Robust Transformer-Based Approach for Intrusion Detection System"



