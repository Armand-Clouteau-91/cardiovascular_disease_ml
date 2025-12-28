import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report


def grid_search(model, param_grid, X_train, y_train, X_pred=None, y_pred=None):
    """
    Run grid search on a given model using predetermined grid attached to the model

    :param model: The scikit-learn model
    :param param_grid: Dictionary of parameters to test
    :param X_train, y_train: Training data
    :param X_test, y_test: (Optional) Hold-out test set for final evaluation
    """
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator= model(),
        param_grid= param_grid,
        cv=cv,
        scoring='accuracy', 
        n_jobs=-1,          
        verbose=1)

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print(f"Best Parameters: {grid.best_params_}")
    print(f"Best CV Score: {grid.best_score_:.4f}")
    
    test_acc = None
    if X_pred is not None and y_pred is not None:
        final_preds = best_model.predict(X_pred)
        test_acc = accuracy_score(y_pred, final_preds)
        print(f"Held-out Test Accuracy: {test_acc:.4f}")
    
    return {
        "best_model": best_model,
        "best_params": grid.best_params_,
        "cv_score": grid.best_score_,
        "test_score": test_acc
    }


def XGBoosting(X_train, y_train, X_pred, y_pred=None):
    """
    Trains a XGBoost model and predicts labels
    
    :param Training_data: pandas df or numpy array
    :param Target_values
    :param Predict_data: data to predict on
    :param True_predictions: actual label of predicted data, used to calculate the accuracy of the model on unseen data
    """
    gb = GradientBoostingClassifier() # instantiation of the model
    gb.fit(X_train, y_train) # model training 
    prediction = gb.predict(X_pred) # prediction
    prediction_score = gb.score(X_pred, y_pred) if y_pred else None # prediction score

    return {f'Prediction':prediction,
            f'Accuracy Model': prediction_score ,
            f'Model Parameters': gb.get_params(),
            f'Training Score':  gb.score(X_train, y_train)}


def KNN(X_train, y_train, X_pred, y_pred=None):
    """
    Trains a KNN model and predicts labels
    
    :param Training_data: pandas df or numpy array
    :param Target_values
    :param Predict_data: data to predict on
    :param True_predictions: actual label of predicted data, used to calculate the accuracy of the model on unseen data
    """
    knn = KNeighborsClassifier() # instantiation of the model
    knn.fit(X_train, y_train) # model training 
    prediction = knn.predict(X_pred) # prediction
    prediction_score = knn.score(X_pred, y_pred) if y_pred else None # prediction score

    return {f'Prediction':prediction,
            f'Accuracy Model': prediction_score ,
            f'Model Parameters': knn.get_params(),
            f'Training Score':  knn.score(X_train, y_train)}


def LogReg(X_train, y_train, X_pred, y_pred=None):
    """
    Trains a Logistic Regression model and predicts labels
    
    :param Training_data: pandas df or numpy array
    :param Target_values
    :param Predict_data: data to predict on
    :param True_predictions: actual label of predicted data, used to calculate the accuracy of the model on unseen data
    """
    lr = LogisticRegression() # instantiation of the model
    lr.fit(X_train, y_train) # model training 
    prediction = lr.predict(X_pred) # prediction
    prediction_score = lr.score(X_pred, y_pred) if y_pred else None # prediction score

    return {f'Prediction':prediction,
            f'Accuracy Model': prediction_score ,
            f'Model Parameters': lr.get_params(),
            f'Training Score':  lr.score(X_train, y_train)}


def SVM(X_train, y_train, X_pred, y_pred=None):
    """
    Trains a SVM model and predicts labels
    
    :param Training_data: pandas df or numpy array
    :param Target_values
    :param Predict_data: data to predict on
    :param True_predictions: actual label of predicted data, used to calculate the accuracy of the model on unseen data
    """
    vm = SVC() # instantiation of the model
    vm.fit(X_train, y_train) # model training 
    prediction = vm.predict(X_pred) # prediction
    prediction_score = vm.score(X_pred, y_pred) if y_pred else None # prediction score    
    
    return {f'Prediction':prediction,
            f'Accuracy Model': prediction_score ,
            f'Model Parameters': vm.get_params(),
            f'Training Score':  vm.score(X_train, y_train)}


def run_models(grid,X_train, y_train, X_pred=None, y_pred=None):
    """
    Iterates through a dictionary of model configurations and runs GridSearch for each.
    """
    results = {}
    for model_name, config in grid.items():
        output = grid_search(
        model=config['model'],
        param_grid=config['params'],
        X_train=X_train,
        y_train=y_train,
        X_test=X_pred,
        y_test=y_pred
    )
    results[model_name] = output
    return 

if __name__ == "__main__":
    
    models_config = {
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=1000),
            'params': {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            }
        },
        'KNeighbors': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            }
        },
        'SVM': {
            'model': SVC(),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
        },
        'XGBoost': {
            'model': GradientBoostingClassifier(),
            'params': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1]
            }
        }
    }