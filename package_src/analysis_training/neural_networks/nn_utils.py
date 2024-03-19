from time import perf_counter as time
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt

class Training:
    def __init__(self, model, optimizer, loss_fn, device, score_funcs, logging_level = logging.CRITICAL) -> None:
        '''
        Initialize the Training class.

        Args:
            model: The neural network model.
            optimizer: The optimizer for model training.
            loss_fn: The loss function for model training.
            device: The device to perform computations on (e.g., "cpu" or "cuda").
            score_funcs: A dictionary containing scoring functions for evaluation.
            logging_level: The logging level (default is logging.CRITICAL).
        '''
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.score_funcs = score_funcs

        # Initialize logger
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging_level)
        formatter = logging.Formatter("|%(funcName)s():%(lineno)d| - %(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

        # Initialize results and final_predictions
        self.results = {}
        self.final_predictions = None

    def train(self, train_loader, test_loader=None, epochs=10):
        '''
        Train the model.

        Args:
            train_loader: DataLoader for training data.
            test_loader: DataLoader for test data (default is None).
            epochs: Number of epochs for training (default is 10).
        '''
        total_training_time, total_testing_time = 0., 0.

        # Define keys for results
        results_keys = ["train Loss", "training time"]
        if test_loader is not None:
            results_keys.extend(["test Loss", "testing time"])
        for key in self.score_funcs.keys():
            results_keys.extend(["train " + key])
            if test_loader is not None:
                results_keys.extend(["test " + key])

        # Initialize results dictionary
        for key in results_keys:
            self.results[key] = []

        # Loop over epochs
        for epoch in range(epochs):
            self.model.train()  # Set model to training mode

            # Run training epoch
            training_time, _ = self.run_epoch(train_loader, "train")
            total_training_time += training_time

            # Print training metrics
            self._logger.critical(f'Training - Epoch [{epoch + 1}/{epochs}], Loss: {self.results["train Loss"][-1]:.4f}, Accuracy: {self.results["train Accuracy"][-1] * 100:.2f}%')
            self.results["training time"].append(training_time)

            # Run testing epoch if test_loader is provided
            if test_loader is not None:
                self.model.eval() # Set model to evaluation mode
                # Disable gradient calculation during testing
                with torch.no_grad():
                    testing_time, _ = self.run_epoch(test_loader, "test")
                    total_testing_time += testing_time
                    self.results["testing time"].append(testing_time)
                    if epoch == epochs - 1:
                        self.final_predictions = _

            # Print testing metrics
            if test_loader is not None:
                self._logger.critical(f'Testing - Epoch [{epoch + 1}/{epochs}], Loss: {self.results["test Loss"][-1]:.4f}, Accuracy: {self.results["test Accuracy"][-1] * 100:.2f}%')

        # Store total training and testing time
        self.results["total training time"] = total_training_time
        if test_loader is not None:
            self.results["total testing time"] = total_testing_time

    def run_epoch(self, dataloader, prefix = ""):
        '''
        Run one epoch of training or testing.

        Args:
            dataloader: DataLoader for the train/test dataset.
            prefix: Prefix(trains/test) for the result keys (default is "").

        Returns:
            tuple: A tuple containing the time taken for the epoch and the predicted labels.
        '''
        optimizer = self.optimizer
        device = self.device
        criterion = self.loss_fn
        model = self.model

        y_true = []
        y_pred = []
        running_loss = []

        # Record start time of epoch
        t0 = time()

        # Iterate over data batches
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss

            # Backpropagation and optimization if in training mode
            if model.training:
                optimizer.zero_grad()  # Clear gradients
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

            # Record loss
            running_loss.append(loss.item())

            # Calculate metrics if score_funcs are provided
            if len(self.score_funcs) > 0:
                predictions = (outputs.detach().cpu().numpy() > 0.5).astype(int)
                labels = labels.detach().cpu().numpy()

                y_true.extend(labels.tolist())
                y_pred.extend(predictions.tolist())

        # Record end time of epoch
        t1 = time()

        # Compute and store loss and scoring metrics
        self.results[prefix + " " + "Loss"].append(np.mean(running_loss))
        for metric_name, score_func in self.score_funcs.items():
            self.results[prefix + " " + metric_name].append(score_func(y_true, y_pred))

        # Return time taken for the epoch and predicted labels
        return t1 - t0, y_pred



def plot_metrics(train_acc, train_loss, test_acc, test_loss):
    '''
    Plots the training and testing accuracy and loss over epochs.

    Args:
        train_acc (list): List containing training accuracy values for each epoch.
        train_loss (list): List containing training loss values for each epoch.
        test_acc (list): List containing testing accuracy values for each epoch.
        test_loss (list): List containing testing loss values for each epoch.

    Returns:
        None
    '''
    # Create a range of epochs for plotting
    epochs = range(1, len(train_acc) + 1)

    # Create a figure with two subplots
    plt.figure(figsize=(14, 4))

    # Subplot for Accuracy
    plt.subplot(1, 2, 1)
    # Plot training accuracy over epochs
    plt.plot(epochs, train_acc, color='darkorange', label='Train Accuracy')
    # Plot testing accuracy over epochs
    plt.plot(epochs, test_acc, color='forestgreen', label='Test Accuracy')
    plt.title('Train and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    # Subplot for Loss
    plt.subplot(1, 2, 2)
    # Plot training loss over epochs
    plt.plot(epochs, train_loss, color='darkorange', label='Train Loss')
    # Plot testing loss over epochs
    plt.plot(epochs, test_loss, color='forestgreen', label='Test Loss')
    plt.title('Train and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()
    


def plot_folds_train_val_metrics(train_losses, val_losses, train_accuracies, val_accuracies, num_epochs):
    '''
    Plots the training and validation accuracy - loss over epochs for all folds.

    Args:
        train_losses (list of lists): List containing training loss values for each fold.
        val_losses (list of lists): List containing validation loss values for each fold.
        train_accuracies (list of lists): List containing training accuracy values for each fold.
        val_accuracies (list of lists): List containing validation accuracy values for each fold.
        num_epochs (int): Number of epochs.

    Returns:
        None
    '''
    # Create a range of epochs for plotting
    epochs = range(1, num_epochs + 1)

    # Create a figure with two subplots
    plt.figure(figsize=(14, 4))

    # Subplot for Accuracy
    plt.subplot(1, 2, 1)
    # Plot training accuracy for each fold with different colors
    for i, train_acc in enumerate(train_accuracies):
        plt.plot(epochs, train_acc, label=f'Fold {i+1} Train Accuracy', color=plt.cm.viridis(i / len(train_accuracies)))
    
    # Plot validation accuracy for each fold with different colors
    for i, val_acc in enumerate(val_accuracies):
        plt.plot(epochs, val_acc, label=f'Fold {i+1} Validation Accuracy', color=plt.cm.inferno(i / len(val_accuracies)))

    # Set plot title and labels for accuracy plot
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

  
    # Subplot for Loss
    plt.subplot(1, 2, 2)
    
    # Plot training loss for each fold with different colors
    for i, train_loss in enumerate(train_losses):
        plt.plot(epochs, train_loss, label=f'Fold {i+1} Train Loss', color=plt.cm.plasma(i / len(train_losses)))

    # Plot validation loss for each fold with different colors
    for i, val_loss in enumerate(val_losses):
        plt.plot(epochs, val_loss, label=f'Fold {i+1} Validation Loss', color=plt.cm.magma(i / len(val_losses)))

    # Set plot title and labels for loss plot
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Show the loss plot
    plt.show()
