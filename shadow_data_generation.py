import numpy as np
import pandas as pd

dataset = pd.read_csv("data/cancer-data.csv")
dataset = dataset.drop(['id', 'Unnamed: 32'], axis=1)

print(dataset.describe())

def generate_cancer_shadow(dataset, no_samples):
    y = dataset['diagnosis']
    x = dataset.drop(['diagnosis'], axis=1)
    columns = dataset.columns
    malignant_cases = y[y == "M"].count()
    benign_cases = y[y == "B"].count()
    malignant_prob = malignant_cases / (malignant_cases + benign_cases)
    benign_prob = benign_cases / (malignant_cases + benign_cases)
    statistics = x.describe()
    new_data = pd.DataFrame(index=range(0,no_samples), columns=columns)
    for index in range(0,no_samples):
        for column in columns:
            if (column != 'diagnosis'):
                new_data[column][index] = np.random.normal(statistics[column]['mean'], statistics[column]['std'], 1)[0]
            else:
                new_data['diagnosis'][index] = np.random.choice(["M", "B"], p=[malignant_prob, benign_prob])

    return new_data


new_data = generate_cancer_shadow(dataset, 500)

print(new_data.shape)
print(new_data.tail(10))