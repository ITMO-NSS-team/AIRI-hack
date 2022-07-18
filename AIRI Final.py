# don't forget to install ase library
# pip install ase

import os
import pandas as pd
import numpy as np
from ase.db import connect
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root


def data_preparation(path):
    db = connect(os.path.join(path, 'train.db'))
    db_test = connect(os.path.join(path, 'test.db'))

    features = []
    target = []

    # Load total number of atoms in molecules and number of each atom type
    for row in db.select():
        features.append([row.natoms, list(row.numbers)])
        target.append(row.data['energy'])

    features_test = []

    for row in db_test.select():
        features_test.append([row.natoms, list(row.numbers)])

    #  Features vectorization

    d = DictVectorizer()
    features_onehot = d.fit_transform([Counter(x[1]) for x in features])
    features_onehot_test = d.transform([Counter(x[1]) for x in features_test])

    #  Dataset splits preparation

    X_train, X_test, y_train, y_test = train_test_split(features_onehot, target)
    X_val = features_onehot_test

    task = Task(TaskTypesEnum.regression)
    train_input = InputData(idx=np.arange(0, len(X_train.toarray())),
                            features=X_train.toarray(), target=np.array(y_train),
                            task=task, data_type=DataTypesEnum.table)
    predict_input = InputData(idx=np.arange(0, len(X_test.toarray())),
                              features=X_test.toarray(), target=np.array(y_test),
                              task=task, data_type=DataTypesEnum.table)
    val_input = InputData(idx=np.arange(0, len(X_val.toarray())),
                          features=X_val.toarray(), target=None,
                          task=task, data_type=DataTypesEnum.table)

    return train_input, predict_input, val_input


def run_AIRI_case(files_path: str, is_visualise=True) -> float:

    fit_data, predict_data, val_data = data_preparation(files_path)

    automl_model = Fedot(problem='regression', timeout=120,
                         preset='best_quality', n_jobs=2, safe_mode=False)
    automl_model.fit(features=fit_data,
                     target=fit_data.target)

    prediction = automl_model.predict(predict_data)
    metrics = automl_model.get_metrics()

    if is_visualise:
        automl_model.current_pipeline.show()

    print(f'MAE for validation sample is {round(metrics["mae"], 3)}')
    automl_model.current_pipeline.save(path='AIRI_pipeline_long')
    automl_model.history.save('AIRI_history_long.json')
    return metrics["mae"]


def create_correct_path(path: str, dirname_flag: bool = False):
    """
    Create path which was created during the testing process.
    """

    for dirname in next(os.walk(os.path.curdir))[1]:
        if dirname.endswith(path):
            if dirname_flag:
                return dirname
            else:
                file = os.path.join(dirname, path + '.json')
                return file
    return None


def show_AIRI_case(path: str):

    train_input, predict_input, val_input = data_preparation(path)

    loaded_model = Fedot(problem='regression')
    loaded_model.load(create_correct_path('AIRI_pipeline_long'))

    prediction = loaded_model.predict(predict_input)

    metrics = loaded_model.get_metrics()

    loaded_model.current_pipeline.show()

    print(f'MAE for validation sample is {round(metrics["mae"], 3)}')

    prediction_val = loaded_model.predict(val_input)

    create_submission_file(prediction_val, 'AIRI_submission.csv')

    return round(metrics["mae"], 3)


def create_submission_file(prediction, filename: str):
    submission_db = pd.DataFrame(columns=["id", "energy"])
    submission_db['id'] = list(range(1, prediction.shape[0] + 1))
    submission_db['energy'] = prediction
    submission_db.to_csv(filename, index=False)
    print(f'Submission file {filename} successfully created')


if __name__ == '__main__':
    # Training part
    # run_AIRI_case(files_path=os.path.join(str(fedot_project_root()), 'airi'), is_visualise=True)

    # Demonstration part
    show_AIRI_case(os.path.join(str(fedot_project_root()), 'airi'))
