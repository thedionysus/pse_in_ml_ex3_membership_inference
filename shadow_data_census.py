import pandas as pd
import numpy as np

def generate_census_shadow(dataset, no_samples):
    columns = dataset.columns
    distribution_df = {}
    length = len(dataset.index)
    for column in columns:
        values = dataset[column].unique()
        values_prob = []
        value_counts = dataset[column].value_counts()
        for value in values:
            probability = value_counts[value]/length
            values_prob.append(probability)
        distribution_df.update({column: values_prob})
    new_data = pd.DataFrame(index=range(0,no_samples), columns=columns)
    for index in range(0,no_samples):
        for column in columns:
            new_data[column][index] = np.random.choice(dataset[column].unique(), p=distribution_df[column])

    return new_data


def predict_shadow_data(shadow_train, shadow_test, model, no_samples):
    shadow_data_train_predict = pd.DataFrame(data=model.predict(shadow_train), index=range(0, no_samples),
                                             columns=["prediction"])
    shadow_data_test_predict = pd.DataFrame(data=model.predict(shadow_test), index=range(0, no_samples),
                                            columns=["prediction"])
    return shadow_data_train_predict, shadow_data_test_predict


def create_in_out_prediction_set_census(shadow_data_train_predict, shadow_data_train_labels, shadow_data_test_predict, shadow_data_test_labels):
    in_prediction_set = pd.concat([shadow_data_train_predict, shadow_data_train_labels], axis=1, sort=False)
    in_prediction_set['in/out'] = "in"
    in_prediction_set = in_prediction_set.rename(
        columns={"prediction": "prediction", "over_50": "class_label", "in/out": "in/out"})

    out_prediction_set = pd.concat([shadow_data_test_predict, shadow_data_test_labels], axis=1, sort=False)
    out_prediction_set['in/out'] = "out"
    out_prediction_set = out_prediction_set.rename(
        columns={"prediction": "prediction", "over_50": "class_label", "in/out": "in/out"})
    return in_prediction_set, out_prediction_set

def create_attack_model(model, x_train, y_train):
    x_prediction_set = pd.get_dummies(x_train)
    attack_model = model.fit(x_prediction_set, y_train)
    return attack_model