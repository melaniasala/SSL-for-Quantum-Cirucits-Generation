import copy

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from QCCL.Data import GraphDataset, from_nx_to_geometric, load_graphs
from QCCL.Models import BYOLOnlineNet, BYOLTargetNet, BYOLWrapper, GCNFeatureExtractor, SimCLRWrapper

from . import NTXentLoss, train, train_byol

losses = {"simclr": NTXentLoss(), "byol": nn.MSELoss(reduction="sum")}

train_fn = {"simclr": train, "byol": train_byol}


class HyperparamTuner:
    def __init__(self, experiment_configs, tuning_configs):
        self.experiment_configs = experiment_configs
        self.tuning_configs = tuning_configs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_splits = experiment_configs["n_splits"]

    def build_model(self, num_layers, proj_output_size):
        print(
            f"Building model with {num_layers} GCNConv layers and projection size {proj_output_size}..."
        )

        if self.experiment_configs["model_type"] == "byol":
            print("Selected model type: BYOL")
            gnn = GCNFeatureExtractor(
                **self.experiment_configs["gnn"], num_layers=num_layers
            )
            projector = nn.Sequential(
                nn.Linear(
                    self.experiment_configs["embedding_size"],
                    self.experiment_configs["hidden_size"],
                ),
                nn.ReLU(),
                nn.Linear(self.experiment_configs["hidden_size"], proj_output_size),
            )
            predictor = nn.Sequential(
                nn.Linear(proj_output_size, self.experiment_configs["hidden_size"]),
                nn.ReLU(),
                nn.Linear(self.experiment_configs["hidden_size"], proj_output_size),
            )
            online_model = BYOLOnlineNet(gnn, projector, predictor)

            target_gnn = copy.deepcopy(gnn)
            target_projector = copy.deepcopy(projector)
            target_model = BYOLTargetNet(target_gnn, target_projector)

            model = BYOLWrapper(online_model, target_model)
            print("BYOL model built successfully.")

        elif self.experiment_configs["model_type"] == "simclr":
            print("Selected model type: SimCLR")
            gnn = GCNFeatureExtractor(
                **self.experiment_configs["gnn"], num_layers=num_layers
            )
            projector = nn.Sequential(
                nn.Linear(
                    self.experiment_configs["embedding_size"],
                    self.experiment_configs["hidden_size"],
                ),
                nn.ReLU(),
                nn.Linear(self.experiment_configs["hidden_size"], proj_output_size),
            )
            model = SimCLRWrapper(gnn, projector)
            print("SimCLR model built successfully.")

        else:
            raise ValueError(
                f"Invalid model type: {self.experiment_configs['model_type']}"
            )

        model.to(self.device)
        print("Model moved to device:", self.device)
        return model, self.experiment_configs["model_type"]

    def split_dataset(self, X):
        print("\nStarting dataset split...")

        train_size = self.experiment_configs["train_size"]
        val_size = self.experiment_configs["val_size"]
        composite_transforms_size = self.experiment_configs[
            "composite_transforms_size"
        ]

        total_size = len(X)
        print(f"Total dataset size: {total_size}")
        print(f"Composite transformations size: {composite_transforms_size}")

        test_size = 1.0 - train_size - val_size
        test_size = int(test_size * total_size)
        val_size = int(val_size * total_size)
        train_size = total_size - val_size - test_size

        print(
            f"Splitting into {train_size} training, {val_size} validation, and {test_size} test circuits..."
        )

        composite_circuits = X[
            -composite_transforms_size:
        ]  # Last n circuits (composite transformations)
        single_transform_circuits = X[
            :-composite_transforms_size
        ]  # Remaining circuits (single transformations)

        # Shuffle and split
        shuffling_mask = np.random.permutation(len(composite_circuits))
        composite_circuits = [composite_circuits[i] for i in shuffling_mask]

        val_data = composite_circuits[:val_size]
        remaining_composite = composite_circuits[
            min(val_size, composite_transforms_size) :
        ]

        test_data = remaining_composite[: min(test_size, len(remaining_composite))]
        if (
            len(test_data) < test_size
        ):  # If not enough composite circuits, take from single transformation circuits
            print(
                "Not enough composite circuits for test set, adding single transformation circuits..."
            )
            additional_test_data = single_transform_circuits[
                : (test_size - len(test_data))
            ]
            test_data = test_data + additional_test_data

        remaining_composite = remaining_composite[
            min(test_size, len(remaining_composite)) :
        ]
        remaining_single = single_transform_circuits[(test_size - len(test_data)) :]
        train_data = remaining_single + remaining_composite
        shuffling_mask = np.random.permutation(len(train_data))
        train_data = [train_data[i] for i in shuffling_mask]

        if self.n_splits is None:
            print("Using standard train/val split...")
            folds = None
        else:
            print(f"Using {self.n_splits}-fold cross-validation...")
            folds = kfold(train_data+val_data, self.n_splits)
            print(f"Generated {self.n_splits} folds for cross-validation.")
        return train_data, val_data, test_data, folds


    def objective(self, trial):
        print("\n" + "=" * 50)
        print(f"Starting Optuna trial {trial.number}...")
        epochs = get_hyperparameter_value(
            trial, "epochs", self.tuning_configs["epochs"]
        )
        n_layers = get_hyperparameter_value(
            trial, "n_layers", self.tuning_configs["n_layers"]
        )
        patience = get_hyperparameter_value(
            trial, "patience", self.tuning_configs["patience"]
        )
        projection_size = get_hyperparameter_value(
            trial, "projection_size", self.tuning_configs["projection_size"]
        )
        tau = get_hyperparameter_value(
            trial, "tau", self.tuning_configs["tau"]
        )
        batch_size = get_hyperparameter_value(
            trial, "batch_size", self.tuning_configs["batch_size"]
        )
        learning_rate = get_hyperparameter_value(
            trial, "learning_rate", self.tuning_configs["learning_rate"]
        )

        print(
            f"Trial hyperparameters: n_layers={n_layers}, patience={patience}, projection_size={projection_size}, "
            f"tau={tau}, batch_size={batch_size}, learning_rate={learning_rate}\n"
        )

        if self.n_splits is None:
            model, model_type = self.build_model(
                num_layers=n_layers, proj_output_size=projection_size
            )
            # Standard train/val split
            print("Training with standard train/val split...")

            history = train_fn[model_type](
                model,
                X_train,
                X_val,
                epochs=epochs,
                batch_size=batch_size,
                lr=learning_rate,
                tau=tau,
                patience=patience,
                device=self.device,
                **self.experiment_configs["train"],
            )

            min_val_loss = min(history["ema_val_loss"])
            print(
                f"Trial {trial.number} completed with minimum validation loss: {min_val_loss}"
            )
            return min_val_loss

        else:
            # K-Fold cross-validation
            print(f"Training with {self.n_splits}-fold cross-validation...")
            fold_losses = []
            for fold_num, (train_folds, val_fold) in enumerate(folds):
                print(f"\nTraining on fold {fold_num + 1}/{self.n_splits}...")
                model, model_type = self.build_model(
                    num_layers=n_layers, proj_output_size=projection_size
                )

                history = train_fn[model_type](
                    model,
                    train_folds,
                    val_fold,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=learning_rate,
                    tau=tau,
                    patience=patience,
                    device=self.device,
                    **self.experiment_configs["train"],
                )

                # Append the minimum validation loss for this fold
                min_val_loss = min(history["ema_val_loss"])
                fold_losses.append(min_val_loss)

            # Calculate the average validation loss across all folds
            avg_val_loss = np.mean(fold_losses)
            print(
                f"Trial {trial.number} completed with average validation loss: {avg_val_loss}"
            )
            return avg_val_loss

    def run_experiment(self, n_trials=100):
        global X_train, X_val, input_dim, folds

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        print("\nLoading dataset...")
        X, _ = load_graphs()
        print("Dataset loaded successfully.")

        # Split the dataset
        data_train, data_val, data_test, folds = self.split_dataset(X)
        use_pre_paired = self.experiment_configs["use_pre_paired_dataset"]

        print("Data split completed:")
        print(f"Training set: {len(data_train)} samples")
        print(f"Validation set: {len(data_val)} samples")
        print(f"Test set: {len(data_test)} samples")

        X_train = GraphDataset(data_train, pre_paired=use_pre_paired)
        X_val = GraphDataset(data_val, pre_paired=use_pre_paired)
        X_train_val = GraphDataset(data_train + data_val, pre_paired=use_pre_paired)

        if self.n_splits is not None:
            folds = [
                (
                    GraphDataset(train_data, pre_paired=use_pre_paired),
                    GraphDataset(val_data, pre_paired=use_pre_paired),
                )
                for train_data, val_data in folds
            ]

        print(f"\nStarting Optuna study with {n_trials} trials...")
        study = optuna.create_study(direction="minimize")  # Minimize validation loss
        study.optimize(self.objective, n_trials=n_trials)

        # Get the best hyperparameters
        best_params = study.best_params
        print(f"\nHyperparameter tuning complete. Best parameters: {best_params}")

        trials_df = (
            study.trials_dataframe()
            .drop(columns=["datetime_start", "datetime_complete", "duration"])
            .to_dict(orient="records")
        )
        study_results = {
            "best_params": best_params,
            "experiment_configs": self.experiment_configs,
            "trials_dataframe": trials_df,
        }

        # Save the results to a YAML file
        with open("hyperparam_tuning_results_with_configs.yaml", "w") as f:
            yaml.dump(study_results, f, default_flow_style=False)

        print("Results saved to hyperparam_tuning_results_with_configs.yaml")

        print("Retrain the model with the best hyperparameters on both training and validation sets...")
        best_model, model_type = self.build_model(
            num_layers=best_params["n_layers"],
            proj_output_size=best_params["projection_size"],
        )
        history_best = train_fn[model_type](
            best_model,
            X_train_val,
            epochs=best_params["epochs"],
            batch_size=best_params["batch_size"],
            lr=best_params["learning_rate"],
            tau=best_params["tau"],
            #patience=best_params["patience"],
            device=self.device,
            **self.experiment_configs["train"],
        )

        print("Retraining completed. Saving the best model and training history...")
        torch.save(best_model.state_dict(), "best_model.pth")
        plt.plot(history_best["train_loss"], label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("training_best_history.png")

        print("Plot saved as training_best_history.png")
        print("Testing best parameters on the test set with Linear Evaluation...")

        # create labels
        labels = torch.tensor([])
        for i in range(len(data_test)):
            class_labels = torch.ones(len(data_test[i])) * i
            labels = torch.cat((labels, class_labels))
        print(f"In test set there are {len(labels)} samples, distributed in {len(data_test)} classes as follows: {labels}")
        
        # split in train and test
        data = [from_nx_to_geometric(g) for gg in data_test for g in gg]
        train, test, y_train, y_test = train_test_split(data, labels, test_size=0.4, stratify=labels)

        # get embeddings
        gnn = best_model.get_gnn().to(self.device)
        train = torch.cat([gnn(d.to(self.device)) for d in train])
        test = torch.cat([gnn(d.to(self.device)) for d in test])

        # scale embeddings
        scaler = StandardScaler()
        train = scaler.fit_transform(train.cpu().detach().numpy())
        test = scaler.transform(test.cpu().detach().numpy())

        # train a linear classifier (multi-class logistic regression) on top of the embeddings
        print("Training a linear classifier on top of the embeddings...")
        print("Logistic Regression (one-vs-rest) classifier:")
        classifier = LogisticRegression().fit(train, y_train)
        y_pred = classifier.predict(test)
        y_pred_probs = classifier.predict_proba(test)
        print(f"\tProbability of each class:\n\t {y_pred_probs}")
        print(f"\tAccuracy of the classifier: {accuracy_score(y_test, y_pred)}")

        print("Logistic Regression (multinomial) classifier:")
        classifier = LogisticRegression(multi_class='multinomial').fit(train, y_train)
        y_pred = classifier.predict(test)
        y_pred_probs = classifier.predict_proba(test)
        print(f"\tProbability of each class:\n\t {y_pred_probs}")
        print(f"\tAccuracy of the classifier: {accuracy_score(y_test, y_pred)}")
        
        print("Ground truth labels:")   
        print(y_test)

        print("Linear evaluation completed")

        print("Hyperparameter tuning and evaluation completed successfully.")

        return best_params


def get_hyperparameter_value(trial, param_name, param_config):
    if param_config["tune"]:
        if param_config["type"] == "loguniform":
            return trial.suggest_float(param_name, *param_config["range"], log=True)
        elif param_config["type"] == "int":
            return trial.suggest_int(param_name, *param_config["range"])
        elif param_config["type"] == "categorical":
            return trial.suggest_categorical(param_name, param_config["choices"])
    else:
        return param_config["default"]


def kfold(X, n_splits):
    fold_size = len(X) // n_splits
    folds = []

    indices = np.random.permutation(len(X))  # Shuffle the indices for randomness
    X_shuffled = [X[i] for i in indices]

    for i in range(n_splits):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < n_splits - 1 else len(X)
        val_data = X_shuffled[val_start:val_end]
        train_data = X_shuffled[:val_start] + X_shuffled[val_end:]
        folds.append((train_data, val_data))

    return folds
