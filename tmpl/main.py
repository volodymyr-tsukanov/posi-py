# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 18:36:46 2024

@author: volodymyr-tsukanov
"""
# Basic data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Scikit-learn base modules
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Preprocessing and decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA

# Classifiers
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC as SVM
from sklearn.tree import DecisionTreeClassifier as DT

# Metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)



### Load cvs
data_csv = pd.read_csv("../2/practice_lab_2.csv", sep=";")
d = data_csv.corr()


### Array operations
arr = data_csv.values
rowsEven = arr[1::2,:]
rowsOdd = arr[0::2,:]
colZeroCount = (arr == 0).sum(axis=0)
colMaxIndexes = np.where(arr == arr.max())[1]   #row => [0]



def draw_corr(corr_data : pd.DataFrame, title='Corr-Mtx'):
    """
    Draws correlation matrix of DataFrame

    Parameters
    ----------
    corr_data : pd.DataFrame
    title : TYPE, optional
        DESCRIPTION. The default is 'Corr-Mtx'.
    """
    cmap = LinearSegmentedColormap.from_list("WiRGn", [(0.85,0.75,0.85,1), (1,0,0,1), (0,1,0,1)])
    
    plt.figure(figsize=(corr_data.shape[0], corr_data.shape[1]))
    plt.title(title, fontsize=22)
    plt.imshow(corr_data, cmap=cmap, aspect='equal')    #or cmap='coolwarm'
    plt.colorbar()  #right bar
    plt.xticks(range(len(corr_data.columns)), corr_data.columns, rotation=45)   #axis labels
    plt.yticks(range(len(corr_data.columns)), corr_data.columns)
    for (i, j), val in np.ndenumerate(corr_data):   #data labels
        plt.text(j, i, f'{val:.2f}', ha='center', va='center', color=(0.05,0.15,0,1))
    plt.grid(False)  #disable gridlines
    plt.show()

def normalize_minmax(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Normalizes data using min-max scaling along specified axis.
    
    Args:
        data: Input array to normalize
        axis: Axis along which to normalize (0 for columns, None for global)
    
    Returns:
        np.ndarray: Normalized array between 0 and 1
    """
    return (data - data.min(axis=axis)) / (data.max(axis=axis) - data.min(axis=axis))

def standardize(data: np.ndarray, axis: int = None) -> np.ndarray:
    """
    Standardizes data using z-score normalization.
    
    Args:
        data: Input array to standardize
        axis: Axis along which to standardize (None for global, 0 for columnwise)
    
    Returns:
        np.ndarray: Standardized array with mean 0 and std 1
    """
    return (data - np.mean(data, axis=axis)) / np.std(data, axis=axis)

def calculate_coefficient_of_variation(data: np.ndarray, 
                                    column_names: list) -> tuple[np.ndarray, str]:
    """
    Calculates coefficient of variation for each column and finds column with maximum CV.
    
    Args:
        data: Input data array
        column_names: List of column names
    
    Returns:
        tuple: (coefficient_of_variation array, name of column with max CV)
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    cv = mean / (std + np.spacing(std))  # np.spacing prevents division by zero
    max_cv_column = column_names[np.argmax(cv)]
    
    return cv, max_cv_column

def test_linear_regression(X,y,n_repetitions):
    mape_scores = []
    for _ in range(n_repetitions):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mape_scores.append(mape)

    return np.mean(mape_scores)

## Usage
df = pd.read_csv('practice_lab_2.csv', sep=';')
X = df.drop('MedianowaCena', axis=1).values
y = df['MedianowaCena'].values
average_mape = test_linear_regression(X,y,100)
print(f"Średni procentowy błąd bezwzględny (MAPE): {average_mape:.2%}")

# Generacja nowych cech. Spróbuj zaproponować cechy/kombinacje cech, które mogłyby ulepszyć jakość predykcji regresji liniowej.
def generate_new_features(X):
    new_features = np.column_stack([
        X[:, 4] / X[:, 7],  # TlenkiAzotu / LPokojow
        X[:, 4] / X[:, 5],  # TlenkiAzotu / WiekMieszkania
        X[:, 4] * X[:, 3],  # TlenkiAzotu * PrzyRzece
        X[:, 4] / X[:, -1]  # TlenkiAzotu / PracFiz
    ])
    return np.column_stack([X, new_features])

def test_linear_regression(n_repetitions, handle_outliers='none', use_new_features=False):
    df = pd.read_csv('practice_lab_2.csv', sep=';')
    X = df.drop('MedianowaCena', axis=1).values
    y = df['MedianowaCena'].values

    if use_new_features:
        X = generate_new_features(X)

    mape_scores = []

    for _ in range(n_repetitions):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        if handle_outliers != 'none':
            z_scores = np.abs((y_train - np.mean(y_train)) / np.std(y_train))
            outliers = z_scores > 3

            if handle_outliers == 'remove':
                X_train = X_train[~outliers]
                y_train = y_train[~outliers]
            elif handle_outliers == 'replace':
                y_train[outliers] = np.mean(y_train)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mape_scores.append(mape)

    return np.mean(mape_scores)

## Usage
average_mape_original = test_linear_regression(100, 'none', False)
average_mape_new_features = test_linear_regression(100, 'none', True)
average_mape_new_features_remove_outliers = test_linear_regression(100, 'remove', True)

print(f"MAPE z oryginalnymi cechami: {average_mape_original:.2%}")
print(f"MAPE z nowymi cechami: {average_mape_new_features:.2%}")
print(f"MAPE z nowymi cechami i usuwaniem wartości odstających: {average_mape_new_features_remove_outliers:.2%}")

# Funkcja do testowania modelu regresji liniowej
def test_linear_regression(n_repetitions, handle_outliers='none'):
    mape_scores = []

    for _ in range(n_repetitions):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        if handle_outliers != 'none':
            z_scores = np.abs((y_train - np.mean(y_train)) / np.std(y_train))
            outliers = z_scores > 3

            if handle_outliers == 'remove':
                X_train = X_train[~outliers]
                y_train = y_train[~outliers]
            elif handle_outliers == 'replace':
                y_train[outliers] = np.mean(y_train)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mape_scores.append(mape)

    return np.mean(mape_scores)

## Usage
average_mape = test_linear_regression(100)
average_mape_remove_outliers = test_linear_regression(100, 'remove')
average_mape_replace_outliers = test_linear_regression(100, 'replace')



def qualitative_to_0_1(data, column, value_to_be_1):
    mask = data[column].values == value_to_be_1
    data.loc[mask, column] = 1
    data.loc[~mask, column] = 0
    return data

## Usage
binary_features = {
    'Gender': 'Female',
    'Married': 'Yes',
    'Education': 'Graduate',
    'Self_Employed': 'Yes',
    'Credit_History': 'Yes',
    'Loan_Status': 'Y'
}
for column, value in binary_features.items():
    data = qualitative_to_0_1(data_csv, column, value)
    

def handle_outliers(X_train: np.ndarray, y_train: np.ndarray, 
                   threshold: float = 3, method: str = 'remove') -> tuple:
    """
    Handles outliers in training data using z-score method.
    
    Args:
        X_train: Training features
        y_train: Training target values
        threshold: Z-score threshold for outlier detection
        method: 'remove' or 'replace' outliers
    
    Returns:
        tuple: (X_processed, y_processed) after outlier handling
    """
    z_scores = np.abs((y_train - y_train.mean()) / y_train.std())
    outliers = z_scores > threshold
    
    if method == 'remove':
        return X_train[~outliers], y_train[~outliers]
    elif method == 'replace':
        y_processed = y_train.copy()
        y_processed[outliers] = y_train.mean()
        return X_train, y_processed
    else:
        raise ValueError("method must be either 'remove' or 'replace'")

## Usage
### Load and split data
y = data_csv['Loan_Status'].values.astype(np.float64)
x = data_csv.drop(columns=['Loan_Status']).values.astype(np.float64)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=221, shuffle=False)

### Handle outliers by removing them
X_clean, y_clean = handle_outliers(X_train, y_train, method='remove')

### Handle outliers by replacing with mean
X_processed, y_processed = handle_outliers(X_train, y_train, method='replace')


def encode_categorical_feature(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Performs one-hot encoding on a specified categorical column.
    
    Args:
        data: Input DataFrame
        column_name: Name of the categorical column to encode
    
    Returns:
        pd.DataFrame: DataFrame with one-hot encoded column
    """
    try:
        cat_feature = pd.Categorical(data[column_name])
        one_hot = pd.get_dummies(cat_feature)
        result = pd.concat([data, one_hot], axis=1)
        result = result.drop(columns=[column_name])
        return result
    except Exception as e:
        print(f"Error during encoding: {str(e)}")
        return data

def calculate_metrics(confusion_matrix):
    """
    Oblicza metryki klasyfikacji na podstawie macierzy pomyłek

    Args:
        confusion_matrix: macierz pomyłek w formacie numpy array

    Returns:
        dict: słownik zawierający wartości metryk
    """
    TP = confusion_matrix[1, 1]
    TN = confusion_matrix[0, 0]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    specificity = TN / (FP + TN) if (FP + TN) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    return {
        'Sensitivity': sensitivity,
        'Precision': precision,
        'Specificity': specificity,
        'Accuracy': accuracy,
        'F1': f1
    }

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculates various classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        dict: Dictionary containing various metrics
    """
    try:
        conf_matrix = confusion_matrix(y_true, y_pred)
        metrics = {
            'confusion_matrix': conf_matrix,
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        return metrics
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {}

def visualize_decision_tree(model: object, feature_names: list, 
                          class_names: list, figsize: tuple = (15, 10)) -> None:
    """
    Visualizes a decision tree model.
    
    Args:
        model: Trained decision tree model
        feature_names: List of feature names
        class_names: List of class names
        figsize: Figure size as tuple (width, height)
    """
    try:
        plt.figure(figsize=figsize)
        DT(model, 
                 feature_names=feature_names,
                 class_names=class_names,
                 filled=True,
                 rounded=True)
        plt.show()
    except Exception as e:
        print(f"Error visualizing decision tree: {str(e)}")

## Usage
### Encode categorical feature
encoded_data = encode_categorical_feature(data_csv, 'Property_Area')

### Calculate metrics
metrics = calculate_classification_metrics(y_test, y_pred)

### Visualize decision tree
visualize_decision_tree(model, 
                       feature_names=X.columns.tolist(),
                       class_names=['Class_0', 'Class_1'])


class OptimalPCAComponents(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.pca = None
        self.n_components_ = None

    def fit(self, X, y=None):
        self.pca = PCA()
        self.pca.fit(X)
        self.n_components_ = np.sum(np.cumsum(self.pca.explained_variance_ratio_) < self.threshold) + 1
        return self

    def transform(self, X):
        return self.pca.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

class OutlierRemover(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_cleaned = X.copy()
        for col in X.T:
            mean = np.mean(col)
            std = np.std(col)
            X_cleaned = X_cleaned[np.abs(col - mean) < 3 * std]
        return X_cleaned


def train_and_evaluate_models(X_train: np.ndarray, X_test: np.ndarray, 
                            y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Trains and evaluates multiple classification models.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
    
    Returns:
        dict: Dictionary with model accuracies
    """
    models = {
        'kNN': kNN(),
        'SVM': SVM(),
        'DTC': DT()
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        results[name] = accuracy
    
    return results

def apply_dimension_reduction(X_train: np.ndarray, X_test: np.ndarray, 
                            method: str = 'pca', n_components: int = 7) -> tuple:
    """
    Applies dimension reduction using PCA or ICA.
    
    Args:
        X_train: Training features
        X_test: Test features
        method: 'pca' or 'ica'
        n_components: Number of components or variance ratio for PCA
    
    Returns:
        tuple: (transformed X_train, transformed X_test)
    """
    if method.lower() == 'pca':
        if isinstance(n_components, float):
            reducer = PCA(n_components)
        else:
            reducer = PCA(n_components=n_components)
    elif method.lower() == 'ica':
        reducer = FastICA(n_components=n_components, random_state=0)
    else:
        raise ValueError("Method must be either 'pca' or 'ica'")
    
    X_train_transformed = reducer.fit_transform(X_train)
    X_test_transformed = reducer.transform(X_test)
    
    return X_train_transformed, X_test_transformed

def visualize_pca_components(X_train: np.ndarray, y_train: np.ndarray, 
                           labels: list = ['Class 0', 'Class 1']) -> None:
    """
    Visualizes first two PCA components.
    
    Args:
        X_train: Training features
        y_train: Training labels
        labels: Class labels for legend
    """
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train_pca[y_train == 0, 0], X_train_pca[y_train == 0, 1], label=labels[0])
    plt.scatter(X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1], label=labels[1])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.title('First Two Principal Components')
    plt.show()

def find_optimal_components(X_train: np.ndarray, variance_threshold: float = 0.95) -> int:
    """
    Finds optimal number of PCA components for given variance threshold.
    
    Args:
        X_train: Training features
        variance_threshold: Desired explained variance ratio
    
    Returns:
        int: Optimal number of components
    """
    pca = PCA()
    pca.fit(X_train)
    
    plt.figure(figsize=(10, 6))
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(cumsum)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.axhline(y=variance_threshold, color='r', linestyle='--')
    plt.show()
    
    n_components = np.sum(cumsum < variance_threshold) + 1
    return n_components

def create_classification_pipeline(n_components: int) -> Pipeline:
    """
    Creates a classification pipeline with PCA, scaling, and kNN.
    
    Args:
        n_components: Number of PCA components
    
    Returns:
        Pipeline: Sklearn pipeline object
    """
    return Pipeline([
        ('pca', PCA(n_components=n_components)),
        ('scaler', StandardScaler()),
        ('classifier', kNN(n_neighbors=5, weights='distance'))
    ])

## Usage
### Load and split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Evaluate original models
original_results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

### Find optimal components
n_components = find_optimal_components(X_train)

### Create and evaluate pipeline
pipe = create_classification_pipeline(n_components)
pipe.fit(X_train, y_train)
pipeline_accuracy = pipe.score(X_test, y_test)

### Visualize PCA components
visualize_pca_components(X_train, y_train, labels=['Mężczyzna', 'Kobieta'])


def data_split(data: pd.DataFrame, y_col_name: str, **train_test_split_kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train and test sets
    
    Args:
        data: Dataset
        y_col_name: Name of the column containing the dependent feature
        **train_test_split_kwargs: Keyword arguments for the train_test_split function
    
    Returns:
        tuple: (x_train, x_test, y_train, y_test)
    """
    y = data[y_col_name].values.astype(np.float64)
    x = data.drop(columns=[y_col_name]).values.astype(np.float64)
    return train_test_split(x, y, **train_test_split_kwargs)

def data_prepare_multivalue(data: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Converts a column with multiple values to one-hot encoding
    
    Args:
        data: Dataset
        col_name: Name of the column to be converted to one-hot encoding
    
    Returns:
        pd.DataFrame: Dataset with the column converted to one-hot encoding
    """
    cat_feature = pd.Categorical(data[col_name])
    one_hot = pd.get_dummies(cat_feature)
    data = pd.concat([data, one_hot], axis=1)
    data = data.drop(columns=[col_name])
    return data

class NoTransform(BaseEstimator, TransformerMixin):
    """Dummy transformer that does nothing"""
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X

class NoScaler(BaseEstimator):
    """Dummy scaler that does nothing"""
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X

@dataclass
class ModelData:
    """Dataclass to store model data"""
    model: object
    y_pred: np.ndarray
    confusion_matrix: np.ndarray
    accuracy: float
    f1: np.float64

def scale_transform_fit_predict_eval(model: object, scaler: object, transformer: object, 
                                   x_train: np.ndarray, y_train: np.ndarray, 
                                   x_test: np.ndarray, y_test: np.ndarray) -> ModelData:
    """Scale, transform, fit, predict, and evaluate a model
    
    Args:
        model: Model to be used
        scaler: Scaler to be used
        transformer: Transformer to be used
        x_train: Independent features of the training set
        y_train: Dependent feature of the training set
        x_test: Independent features of the test set
        y_test: Dependent feature of the test set
    
    Returns:
        ModelData: Model data
    """
    pipe = Pipeline(
        [
            ['scaler', scaler if scaler else NoScaler()],
            ['transformer', transformer if transformer else NoTransform()],
            ['model', model]
        ]
    )
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    return ModelData(
        model=pipe,
        y_pred=y_pred,
        confusion_matrix=confusion_matrix(y_test, y_pred),
        accuracy=accuracy_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred, average='binary' if len(set(y_test)) == 2 else 'weighted')
    )

## Usage
### Data Splitting with custom parameters
data_csv = pd.read_csv("practice_lab_2.csv", sep=";")
x_train, x_test, y_train, y_test = data_split(
    data_csv, 
    'Loan_Status', 
    test_size=0.2, 
    random_state=42
)

### Multi-value categorical data preparation
data_encoded = data_prepare_multivalue(data_csv, 'Property_Area')

### Pipeline with NoTransform and NoScaler
basic_pipeline = Pipeline([
    ['scaler', NoScaler()],
    ['transformer', NoTransform()],
    ['model', kNN()]
])

### Complete model evaluation pipeline
model_evaluation = scale_transform_fit_predict_eval(
    model=kNN(),
    scaler=StandardScaler(),
    transformer=PCA(n_components=0.95),
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test
)

print(f"Model Accuracy: {model_evaluation.accuracy:.2%}")
print(f"Model F1 Score: {model_evaluation.f1:.2%}")


# keras Model Building and Modification Functions
def create_sequential_model(
    input_shape: Tuple,
    layers: List[Dict],
    activation: str = 'relu',
    optimizer: Union[str, keras.optimizers.Optimizer] = 'adam',
    loss: Union[str, keras.losses.Loss] = 'sparse_categorical_crossentropy',
    metrics: List[str] = ['accuracy'],
    dropout_rate: float = 0.0,
    use_batch_norm: bool = False
) -> keras.Model:
    """
    Creates a sequential Keras model with advanced configuration options.

    Args:
        input_shape: Tuple specifying the input shape
        layers: List of layer configurations
        activation: Default activation function
        optimizer: Model optimizer
        loss: Loss function
        metrics: List of metrics to track
        dropout_rate: Global dropout rate (if > 0)
        use_batch_norm: Whether to add batch normalization after dense layers

    Returns:
        Compiled Keras sequential model

    Example:
        layers = [
            {'type': 'Dense', 'units': 64, 'activation': 'relu'},
            {'type': 'Dense', 'units': 32},
            {'type': 'Dense', 'units': 10, 'activation': 'softmax'}
        ]
        model = create_sequential_model((28, 28, 1), layers)
    """
    try:
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=input_shape))

        supported_layers = {
            'Dense': keras.layers.Dense,
            'Conv2D': keras.layers.Conv2D,
            'MaxPooling2D': keras.layers.MaxPooling2D,
            'Flatten': keras.layers.Flatten,
            'Dropout': keras.layers.Dropout
        }

        for layer_config in layers:
            # Validate layer configuration
            if 'type' not in layer_config:
                raise ValueError("Each layer config must contain 'type'")
            
            layer_type = layer_config.pop('type')
            if layer_type not in supported_layers:
                raise ValueError(f"Unsupported layer type: {layer_type}")

            # Add layer
            if layer_type == 'Flatten':
                model.add(keras.layers.Flatten())
            else:
                # Set default activation for Dense layers if not specified
                if layer_type == 'Dense' and 'activation' not in layer_config:
                    layer_config['activation'] = activation

                model.add(supported_layers[layer_type](**layer_config))

                # Add batch normalization if requested
                if use_batch_norm and layer_type in ['Dense', 'Conv2D']:
                    model.add(keras.layers.BatchNormalization())

                # Add dropout if specified
                if dropout_rate > 0 and layer_type in ['Dense', 'Conv2D']:
                    model.add(keras.layers.Dropout(dropout_rate))

        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        return model

    except Exception as e:
        raise Exception(f"Error creating model: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Define model architecture
    layers = [
        {'type': 'Conv2D', 'filters': 32, 'kernel_size': 3, 'activation': 'relu'},
        {'type': 'MaxPooling2D', 'pool_size': 2},
        {'type': 'Flatten'},
        {'type': 'Dense', 'units': 128},
        {'type': 'Dense', 'units': 10, 'activation': 'softmax'}
    ]

    # Create model with batch normalization and dropout
    model = create_sequential_model(
        input_shape=(28, 28, 1),
        layers=layers,
        dropout_rate=0.3,
        use_batch_norm=True
    )

    # Display model summary
    model.summary()

def create_modified_model(base_model, new_layers=None, trainable=False):
    """
    Creates a modified model by adding new layers and controlling trainability
    
    Args:
        base_model: Original Keras model
        new_layers: List of layers to add
        trainable: Whether to make base model layers trainable
    
    Returns:
        Modified Keras model
    """
    # Set trainability of base model
    base_model.trainable = trainable
    
    # Get the input layer
    inputs = keras.Input(shape=base_model.input_shape[1:])
    
    # Pass input through base model
    x = base_model(inputs)
    
    # Add new layers if specified
    if new_layers:
        for layer in new_layers:
            x = layer(x)
            
    # Create new model
    model = keras.Model(inputs, x)
    return model

def freeze_layers(model, layers_to_freeze):
    """
    Freezes specified layers in the model
    
    Args:
        model: Keras model
        layers_to_freeze: List of layer indices or names to freeze
    """
    for layer in layers_to_freeze:
        if isinstance(layer, int):
            model.layers[layer].trainable = False
        else:
            model.get_layer(layer).trainable = False
    return model

def clone_and_modify_model(original_model, modifications):
    """
    Clones a model and applies modifications
    
    Args:
        original_model: Original Keras model
        modifications: Dict of layer names and new configurations
    
    Returns:
        Modified clone of original model
    """
    cloned_model = keras.models.clone_model(original_model)
    cloned_model.set_weights(original_model.get_weights())
    
    for layer_name, config in modifications.items():
        layer = cloned_model.get_layer(layer_name)
        for key, value in config.items():
            setattr(layer, key, value)
            
    return cloned_model

def create_binary_classifier(
    input_dim: int,
    hidden_layers: List[int] = [64, 32],
    dropout_rate: float = 0.2
) -> keras.Model:
    """
    Creates a simple binary classifier with specified architecture.
    Common interview task: Create a binary classifier with variable hidden layers.
    
    Args:
        input_dim: Input feature dimension
        hidden_layers: List of units in each hidden layer
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Compiled binary classification model
    """
    model = keras.Sequential()
    
    # Input layer
    model.add(keras.layers.Dense(hidden_layers[0], input_dim=input_dim, activation='relu'))
    model.add(keras.layers.Dropout(dropout_rate))
    
    # Hidden layers
    for units in hidden_layers[1:]:
        model.add(keras.layers.Dense(units, activation='relu'))
        model.add(keras.layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_mnist_cnn() -> keras.Model:
    """
    Creates a CNN for MNIST classification.
    Common interview task: Implement a CNN for image classification.
    
    Returns:
        Compiled CNN model for MNIST
    """
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def add_regularization(model: keras.Model, 
                      l1_factor: float = 0.01, 
                      l2_factor: float = 0.01) -> keras.Model:
    """
    Adds L1 and L2 regularization to Dense layers.
    Common interview task: Implement regularization to prevent overfitting.
    
    Args:
        model: Original model
        l1_factor: L1 regularization factor
        l2_factor: L2 regularization factor
    
    Returns:
        Model with added regularization
    """
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            layer.kernel_regularizer = keras.regularizers.l1_l2(l1=l1_factor, l2=l2_factor)
    
    # Recompile model
    model.compile(optimizer=model.optimizer,
                 loss=model.loss,
                 metrics=model.metrics)
    return model

def create_autoencoder(
    input_dim: int,
    encoding_dim: int = 32,
    activation: str = 'relu'
) -> Tuple[keras.Model, keras.Model]:
    """
    Creates an autoencoder model.
    Common interview task: Implement dimensionality reduction using autoencoders.
    
    Args:
        input_dim: Input dimension
        encoding_dim: Dimension of encoded representation
        activation: Activation function
    
    Returns:
        Tuple of (autoencoder, encoder) models
    """
    # Encoder
    encoder = keras.Sequential([
        keras.layers.Dense(encoding_dim, activation=activation, input_shape=(input_dim,))
    ])
    
    # Decoder
    decoder = keras.Sequential([
        keras.layers.Dense(input_dim, activation='sigmoid')
    ])
    
    # Autoencoder
    autoencoder = keras.Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, encoder

def model_evaluation_suite(model: keras.Model, 
                         X_test: np.ndarray, 
                         y_test: np.ndarray) -> Dict:
    """
    Comprehensive model evaluation.
    Common interview task: Implement model evaluation metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1) if y_pred.shape[1] > 1 else (y_pred > 0.5).astype(int)
    
    evaluation = {
        'test_loss': model.evaluate(X_test, y_test)[0],
        'test_accuracy': model.evaluate(X_test, y_test)[1],
        'predictions': y_pred,
        'predicted_classes': y_pred_classes
    }
    
    return evaluation

# Example usage
if __name__ == "__main__":
    # Binary classifier example
    binary_model = create_binary_classifier(input_dim=20)
    
    # CNN example
    cnn_model = create_mnist_cnn()
    
    # Autoencoder example
    autoencoder, encoder = create_autoencoder(input_dim=784)  # for MNIST
    
    # Add regularization to existing model
    regularized_model = add_regularization(binary_model)

## Usage
### Add new layers to existing model
new_layers = [
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
]
modified_model = create_modified_model(base_model, new_layers)

### Freeze specific layers
freeze_layers(model, [0, 1, 2])  # Freeze first three layers

### Modify layer configurations
modifications = {
    'dense_1': {'activation': 'relu', 'units': 128},
    'dropout_1': {'rate': 0.3}
}
modified_model = clone_and_modify_model(original_model, modifications)