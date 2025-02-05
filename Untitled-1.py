# %%
import os
import pandas as pd

datasetPath = './CICIDS2017'
allDataframe = []

for file in os.listdir(datasetPath):
    if file.endswith(".csv"):
        csvPath = os.path.join(datasetPath, file)
        dataframe = pd.read_csv(csvPath,low_memory=False)
        allDataframe.append(dataframe)

summaryDataset = pd.concat(allDataframe, axis=0)
summaryDataset.dropna(inplace=True)
summaryDataset

# %%
summaryDataset.info(show_counts=True)

# %%
def standardizeColumnNames(df):
  df.columns = df.columns.str.lower()
  df.columns = df.columns.str.strip()
  df.columns = df.columns.str.replace(' +', ' ', regex=True)
  df.columns = df.columns.str.title()
  return df

summaryDataset = standardizeColumnNames(summaryDataset)

# %%
summaryDataset.info(show_counts=True)

# %%
import matplotlib.pyplot as plt

labelCounts = summaryDataset['Label'].value_counts()

plt.figure(figsize=(15, 10))
ax = labelCounts.plot(kind='bar', color='skyblue')

for bar in ax.containers[0]:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, y + 0.1, round(y, 2), ha='center', va='bottom')

plt.xlabel('Label Value')
plt.ylabel('Count')
plt.title('The distribution of target value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import NearestNeighbors
import numpy as np

THRESHOLD = int(1e5)
randomState = 194
features, labels = summaryDataset.drop(columns='Label'), summaryDataset['Label']
maxFloat64 = np.finfo(np.float64).max
features = features.where(features <= maxFloat64, maxFloat64)

classQuantity = summaryDataset['Label'].value_counts()
classUndersample = classQuantity[classQuantity >= THRESHOLD].index

# %%
underSampler = RandomUnderSampler(sampling_strategy={label: THRESHOLD if (classQuantity[label] > THRESHOLD) else classQuantity[label] for label in np.unique(summaryDataset['Label'])},
                                  random_state=randomState)
sampledFeatures, sampledLabels = underSampler.fit_resample(features, labels)

# %%
smoteen = SMOTE(sampling_strategy={label: THRESHOLD for label in np.unique(summaryDataset['Label'])}, 
                k_neighbors=NearestNeighbors(n_neighbors=5, n_jobs=-1), 
                random_state=randomState)
balancedFeatures, balancedLabels = smoteen.fit_resample(sampledFeatures, sampledLabels)

# %%
labelCounts = balancedLabels.value_counts()

plt.figure(figsize=(15, 10))
ax = labelCounts.plot(kind='bar', color='skyblue')

for bar in ax.containers[0]:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, y + 0.1, round(y, 2), ha='center', va='bottom')

plt.xlabel('Label Value')
plt.ylabel('Count')
plt.title('The distribution of target value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

labelEncoder = LabelEncoder().fit(balancedLabels)
labelMapping = {value: label for value, label in enumerate(labelEncoder.classes_)}
balancedLabels = labelEncoder.transform(balancedLabels)

balancedFeatures = Pipeline([
    ('Quantiles transform ', QuantileTransformer(n_quantiles=THRESHOLD//4, subsample=THRESHOLD//2, output_distribution='normal')),
    ('Remove constant-value feature', VarianceThreshold()),
    ('Remove quasi-constant value feature', VarianceThreshold(0.99 * (1.0 - 0.99))),
    ('Select K-Best feature', SelectPercentile(mutual_info_classif, percentile=50))]
).fit_transform(balancedFeatures, balancedLabels)

# %%
import json

np.save('./preprocessedData/data.npy', balancedFeatures)
np.save('./preprocessedData/label.npy', balancedLabels)

with open("./preprocessedData/labelMapping.json", "w") as file:
    json.dump(labelMapping, file)


