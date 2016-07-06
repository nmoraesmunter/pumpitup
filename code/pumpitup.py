import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV


def feature_engineering(df):
    to_datetime = ['date_recorded']
    to_boolean = ['public_meeting', 'permit']
    to_category = ['scheme_management', 'extraction_type', 'management',
                   'payment', 'water_quality', 'quantity', 'source', 'waterpoint_type']
    to_drop = ['wpt_name', 'basin', 'subvillage', 'region', 'ward', 'lga', 'recorded_by',
               'scheme_name', 'extraction_type_group', 'extraction_type_class', 'management_group',
               'payment_type', 'quality_group', 'quantity_group', 'source_type', 'source_class',
               'waterpoint_type_group', 'funder', 'installer']
    df.drop(to_drop, axis=1, inplace=True)
    columns = df.columns
    for col in columns:
        if col in to_datetime:
            print col
            df[col] = pd.to_datetime(df[col])
            df[col + "_doy"] = df[col].dt.dayofyear
            df[col + "_month"] = df[col].dt.month
            df[col + "_year"] = df[col].dt.year
            df.pop(col)
        if col in to_category:
            print col
            df[col] = df[col].astype('category')
            dummies = pd.get_dummies(df[col], drop_first=True).rename(columns=lambda x: col + str(x))
            df = pd.concat([df, dummies], axis=1)
            df.pop(col)
        if col in to_boolean:
            print col
            d = {True: 1, False: 0, np.nan: 0}
            df[col] = df[col].map(d)
    return df


def submission(test_values, train_values, train_labels):
    X_train = train_values[train_values.columns.difference(['id'])]
    y_train = train_labels["status_group"]

    rf = RandomForestClassifier()
    rf.set_params(**getBestParams(X_train, y_train))
    rf.fit(X_train, y_train)

    X_test = test_values[test_values.columns.difference(['id'])]
    y_predict = rf.predict(X_test)

    submission = pd.DataFrame(data=y_predict,  # values
                              index=test_values["id"],  # 1st column as index
                              columns=["status_group"])  # 1st row as the column names

    submission.to_csv("../data/submission.csv")

def getBestParams(X_train, y_train, rerun = False):
    '''
    if rerun = True : Get best parameters using cross validated grid search
    if rerun = False: return the best parameters so far
    :param X_train:
    :param y_train:
    :param rerun:
    :return:
    '''

    if rerun:
        rf_grid = {'max_depth': [None],
                   'max_features': [20],
                   'min_samples_split': [1, 3],
                   'min_samples_leaf': [1, 3],
                   'bootstrap': [True],
                   'n_estimators': [50],
                   'random_state': [1]}

        grid_cv = GridSearchCV(RandomForestClassifier(), rf_grid, n_jobs=-1, verbose=True,
                               scoring='mean_squared_error').fit(X_train, y_train)
        best_params = grid_cv.best_params_
    else:
        best_params = {'bootstrap': True,
                     'max_depth': None,
                     'max_features': 20,
                     'min_samples_leaf': 1,
                     'min_samples_split': 3,
                     'n_estimators': 50,
                     'random_state': 1}

    return best_params


def main():
    train_labels = pd.read_csv("../data/training_set_labels.csv")
    train_values = pd.read_csv("../data/training_set_values.csv")
    test_values = pd.read_csv("../data/test_set_values.csv")
    all_values = pd.concat([train_values, test_values], axis=0)
    all_values = feature_engineering(all_values)

    mask = all_values["id"].isin(test_values["id"])
    train_values = all_values.loc[~mask]
    test_values = all_values.loc[mask]
    submission(test_values, train_values, train_labels)



if __name__ == "__main__":
    main()