import pytest
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import f1_score
from tools import COLUMN_NAMES
from tools import unzip_and_open_dataset
from tools import split_data_with_id_hash
import cloudpickle


@pytest.fixture
def obesity():
    """
    Fixture to load dataset from a zip file into a pandas DataFrame
    """
    try:
        data = unzip_and_open_dataset('obesity.zip')
        data.columns = COLUMN_NAMES
        return data
    except FileNotFoundError as e:
        raise FileNotFoundError('obesity.zip file not found.') from e
    except Exception as e:
        raise Exception(f'Error loading obesity: {e}') from e


@pytest.fixture
def obesity_split(obesity):
    """
    Fixture to split the obesity dataset into training and testing sets.

    Returns tuple (train_set, test_set).
    """
    data = obesity.copy()
    data['id'] = data['age'] * data['height'] * data['weight']
    train_set, test_set = split_data_with_id_hash(data, 0.2, 'id')
    for set_ in (train_set, test_set):
        set_.drop('id', axis=1, inplace=True)
    return train_set, test_set


@pytest.fixture
def obesity_train(obesity_split):
    """Fixture to provide the training subset of the obesity dataset."""
    return obesity_split[0]


@pytest.fixture
def obesity_test(obesity_split):
    """Fixture to provide the testing subset of the obesity dataset."""
    return obesity_split[1]


@pytest.mark.data
@pytest.mark.parametrize("fixture_name", ["obesity", "obesity_train", "obesity_test"])
def test_fixture_output_type(fixture_name, request):
    """
    Tests if datasets loaded, as expected, into pd.DataFrame.
    """
    fixture = request.getfixturevalue(fixture_name)
    assert isinstance(fixture, pd.DataFrame)


@pytest.fixture
def preprocessing():
    """
    Fixture to load preprocessing steps from a pickle file.
    """
    try:
        with open('preprocessing.pkl', 'rb') as f:
            return cloudpickle.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError('preprocessing.pkl not found.') from e
    except Exception as e:
        raise Exception(f'Error loading preprocessing steps: {e}') from e


@pytest.fixture
def obesity_train_preprocessed(preprocessing, obesity_train):
    """Fixture to provide preprocessed training set."""
    return preprocessing.transform(obesity_train)


@pytest.fixture
def obesity_test_preprocessed(preprocessing, obesity_test):
    """Fixture to provide preprocessed testing set."""
    return preprocessing.transform(obesity_test)


@pytest.mark.preprocessing
@pytest.mark.parametrize('fixture_name', ['obesity_train_preprocessed', 'obesity_test_preprocessed'])
def test_preprocessing(fixture_name, request):
    """Test the preprocessing steps applied to the datasets."""
    fixture = request.getfixturevalue(fixture_name)
    assert fixture.shape[1] == 31


@pytest.fixture
def model():
    """Fixture to load the final model from a pickle file."""
    with open('final_model.pkl', 'rb') as f:
        final_model = cloudpickle.load(f)
        return final_model


@pytest.mark.modelType
def test_model(model):
    """Test the type of the model instance."""
    assert isinstance(model, BaseEstimator)


@pytest.mark.modelPerformance
def test_performance(model, obesity_train, obesity_test):
    """Test the performance of the model."""
    X_train, y_train = obesity_train.drop('obesity_level', axis=1), obesity_train['obesity_level'].copy()
    X_test, y_test = obesity_test.drop('obesity_level', axis=1), obesity_test['obesity_level'].copy()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred, average='weighted')
    assert score >= 0.95
