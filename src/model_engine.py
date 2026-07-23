from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
import optuna
import logging
from config.settings import RANDOM_SEED

optuna.logging.set_verbosity(optuna.logging.WARNING)

class TimeSeriesEnsembleEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.best_ensemble = None
        self.feature_cols = []
        
    def prepare_matrices(self, df: pd.DataFrame):
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Next_Day_Return', 'Target', 'Daily_Return']
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[self.feature_cols].values
        y = df['Target'].values
        return X, y

    def train_walk_forward_validation(self, X, y):
        """ Deploys Optuna Optimization using walk-forward validation folds. """
        tscv = TimeSeriesSplit(n_splits=5)
        
        def objective(trial):
            rf_n_estimators = trial.suggest_int('rf_n_estimators', 50, 200)
            rf_max_depth = trial.suggest_int('rf_max_depth', 5, 15)
            ada_n_estimators = trial.suggest_int('ada_n_estimators', 50, 200)
            ada_lr = trial.suggest_float('ada_lr', 0.01, 0.3, log=True)
            knn_neighbors = trial.suggest_int('knn_neighbors', 3, 15)
            cart_max_depth = trial.suggest_int('cart_max_depth', 3, 12)
            bag_n_estimators = trial.suggest_int('bag_n_estimators', 10, 100)

            fold_accuracies = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                knn = KNeighborsClassifier(n_neighbors=knn_neighbors, weights='distance')
                cart = DecisionTreeClassifier(max_depth=cart_max_depth, random_state=RANDOM_SEED)
                rf = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=RANDOM_SEED)
                bag = BaggingClassifier(n_estimators=bag_n_estimators, random_state=RANDOM_SEED)
                ada = AdaBoostClassifier(n_estimators=ada_n_estimators, learning_rate=ada_lr, random_state=RANDOM_SEED)
                
                estimators = [('knn', knn), ('cart', cart), ('rf', rf), ('bag', bag), ('ada', ada)]
                voting_mod = VotingClassifier(estimators=estimators, voting='soft', weights=[1, 1, 3, 2, 2])
                
                voting_mod.fit(X_train_scaled, y_train)
                preds = voting_mod.predict(X_test_scaled)
                fold_accuracies.append(accuracy_score(y_test, preds))
                
            return np.mean(fold_accuracies)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10) # 10 trials per asset optimizes accuracy/speed trade-off
        
        best_params = study.best_value
        bp = study.best_params
        X_scaled = self.scaler.fit_transform(X)
        
        self.best_ensemble = VotingClassifier(
            estimators=[
                ('knn', KNeighborsClassifier(n_neighbors=bp['knn_neighbors'], weights='distance')),
                ('cart', DecisionTreeClassifier(max_depth=bp['cart_max_depth'], random_state=RANDOM_SEED)),
                ('rf', RandomForestClassifier(n_estimators=bp['rf_n_estimators'], max_depth=bp['rf_max_depth'], random_state=RANDOM_SEED)),
                ('bag', BaggingClassifier(n_estimators=bp['bag_n_estimators'], random_state=RANDOM_SEED)),
                ('ada', AdaBoostClassifier(n_estimators=bp['ada_n_estimators'], learning_rate=bp['ada_lr'], random_state=RANDOM_SEED))
            ],
            voting='soft', weights=[1, 1, 3, 2, 2]
        )
        self.best_ensemble.fit(X_scaled, y)
        return best_params

    def evaluate_model(self, X_test, y_test):
        X_scaled = self.scaler.transform(X_test)
        preds = self.best_ensemble.predict(X_scaled)
        
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, average='macro', zero_division=0),
            "recall": recall_score(y_test, preds, average='macro', zero_division=0),
            "f1_score": f1_score(y_test, preds, average='macro', zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist()
        }
        return metrics, preds