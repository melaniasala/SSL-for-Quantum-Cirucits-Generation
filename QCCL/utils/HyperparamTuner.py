import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from QCCL.Models import BYOLOnlineNet, BYOLTargetNet, BYOL, SimCLR, GCNFeatureExtractor
import copy
from QCCL.Data import GraphDataset, load_graphs
import numpy as np
from QCCL.utils import NTXentLoss, train, train_byol

losses = {
    'cl': NTXentLoss(),
    'byol': nn.MSELoss(reduction='sum')
}

train_fn = {
    'cl': train,
    'byol': train_byol
}

class HyperparamTuner:
    def __init__(self, experiment_configs, tuning_configs):
        self.experiment_configs = experiment_configs
        self.tuning_configs = tuning_configs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_model(self, num_layers, proj_output_size):
        if self.experiment_configs['model_type'] == 'byol':
            gnn = GCNFeatureExtractor(num_layers=num_layers, **self.experiment_configs)
            projector = nn.Sequential(nn.Linear(self.experiment_configs['embedding_size'], self.experiment_configs['hidden_size']), 
                                    nn.ReLU(), 
                                    nn.Linear(self.experiment_configs['hidden_size'], proj_output_size))
            predictor = nn.Sequential(nn.Linear(proj_output_size, self.experiment_configs['hidden_size']),
                                    nn.ReLU(),
                                    nn.Linear(self.experiment_configs['hidden_size'], proj_output_size))
            online_model = BYOLOnlineNet(gnn, projector, predictor)

            target_gnn = copy.deepcopy(gnn)  # Deep copy to ensure independent weights
            target_projector = copy.deepcopy(projector)  # Deep copy to ensure independent weights
            target_model = BYOLTargetNet(target_gnn, target_projector)

            model = BYOL(online_model, target_model)

        elif self.experiment_configs['model_type'] == 'cl':
            gnn = GCNFeatureExtractor(num_layers=num_layers, **self.experiment_configs)
            projector = nn.Sequential(nn.Linear(self.experiment_configs['embedding_size'], self.experiment_configs['hidden_size']), 
                                    nn.ReLU(), 
                                    nn.Linear(self.experiment_configs['hidden_size'], proj_output_size))
            model = SimCLR(gnn, projector)

        else:
            raise ValueError(f"Invalid model type: {self.experiment_configs['model_type']}")

        return model, self.experiment_configs['model_type']

    # Dataset splitting function
    def split_dataset(self, X):
        # get from configs
        train_size = self.experiment_configs['train_size']
        val_size = self.experiment_configs['val_size']
        use_pre_paired = self.experiment_configs['use_pre_paired_dataset']
        composite_transforms_size = self.experiment_configs['composite_transforms_size']
        
        total_size = len(X)
        
        print('Number of circuits under composite transformations:', composite_transforms_size)
        
        train_size = int(train_size * total_size)
        val_size = int(val_size * total_size)
        test_size = total_size - train_size - val_size

        composite_circuits = X[-composite_transforms_size:]  # Last n circuits (composite transformations)
        single_transform_circuits = X[:-composite_transforms_size]  # Remaining circuits (single transformations)

        composite_circuits = np.random.permutation(composite_circuits)

        val_data = composite_circuits[:val_size]
        remaining_composite = composite_circuits[min(val_size, composite_transforms_size):]

        test_data = remaining_composite[:min(test_size, len(remaining_composite))]
        if len(test_data) < test_size:  # If not enough composite circuits, take from single transformation circuits
            additional_test_data = single_transform_circuits[:(test_size - len(test_data))]
            test_data = np.concatenate([test_data, additional_test_data], axis=0)

        remaining_single = single_transform_circuits[(test_size - len(test_data)):]
        train_data = np.concatenate([remaining_single, remaining_composite], axis=0)
        train_data = np.random.permutation(train_data)

        print('Data split:')
        print('train:', len(train_data), '(', round((len(train_data) / total_size) * 100, 1), '%)')
        print('val:', len(val_data), '(', round((len(val_data) / total_size) * 100, 1), '%)')
        print('test:', len(test_data), '(', round((len(test_data) / total_size) * 100, 1), '%)')

        train_dataset = GraphDataset(train_data, pre_paired=use_pre_paired)
        val_dataset = GraphDataset(val_data, pre_paired=use_pre_paired)
        test_dataset = GraphDataset(test_data, pre_paired=use_pre_paired)
        
        return train_dataset, val_dataset, test_dataset

    # Optuna objective function
    def objective(self, trial):
        # Hyperparameters to tune
        learning_rate = get_hyperparameter_value(trial, 'learning_rate', self.tuning_configs['learning_rate'])
        n_layers = get_hyperparameter_value(trial, 'n_layers', self.tuning_configs['n_layers'])
        patience = get_hyperparameter_value(trial, 'patience', self.tuning_configs['patience'])
        projection_size = get_hyperparameter_value(trial, 'projection_size', self.tuning_configs['projection_size'])
        temperature = get_hyperparameter_value(trial, 'temperature', self.tuning_configs['temperature'])
        batch_size = get_hyperparameter_value(trial, 'batch_size', self.tuning_configs['batch_size'])

        # Model, loss function, and optimizer
        model, model_type = self.build_model(num_layers=n_layers, proj_output_size=projection_size, **self.experiment_configs)

        history = train_fn[model_type](
            model, 
            X_train, X_val, 
            batch_size=batch_size, 
            lr=learning_rate, 
            tau=temperature, 
            patience=patience,
            device=self.device,
            **self.experiment_configs
            )
        
        return min(history['ema_val_loss'])

    # Main experiment function
    def run_experiment(self, n_trials=100):
        global X_train, X_val, X_test, input_dim

        # Load the dataset
        X, _ = load_graphs()
        X_train, X_val, X_test = self.split_dataset(X)

        study = optuna.create_study(direction='minimize')  # Minimize validation loss
        study.optimize(self.objective, n_trials=n_trials)
        
        # Get the best hyperparameters
        best_params = study.best_params
        print(f"Best Hyperparameters: {best_params}")
        
        # Save table with hyperparameters and results
        df = study.trials_dataframe()
        df.to_csv('hyperparam_tuning_results.csv', index=False)

        return best_params


def get_hyperparameter_value(trial, param_name, param_config):
    if param_config['tune']:
        if param_config['type'] == 'loguniform':
            return trial.suggest_loguniform(param_name, *param_config['range'])
        elif param_config['type'] == 'int':
            return trial.suggest_int(param_name, *param_config['range'])
        elif param_config['type'] == 'categorical':
            return trial.suggest_categorical(param_name, param_config['choices'])
    else:
        return param_config['default']

