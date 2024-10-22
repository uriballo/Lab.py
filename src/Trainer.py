import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from Plotter import plot_results

class Trainer:
    def __init__(self):
        """
        Initializes the Training class and sets up performance history.
        """
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.precision_history = []
        self.recall_history = []
        self.f1_history = []
        self.confusion_matrix = None  # Initialize confusion matrix

    def test_model(self, val_loader, model, device):
        """
        Tests the model on the validation data and computes loss and accuracy.

        Args:
            val_loader (DataLoader): DataLoader for validation data.
            model (nn.Module): The PyTorch model to be tested.
            device (str): Device on which to perform testing ('cpu' or 'cuda').
        
        Returns:
            val_loss (float): The average loss on the validation data.
            val_acc (float): The accuracy on the validation data.
            precision, recall, f1: Computed metrics for validation.
        """
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        criterion = nn.CrossEntropyLoss()

        start_time = time.time()

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Collect labels and predictions for metrics calculation
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        elapsed_time = time.time() - start_time

        # Compute precision, recall, and f1
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=1)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=1)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=1)

        # Store confusion matrix after the last validation
        self.confusion_matrix = confusion_matrix(all_labels, all_preds)

        # Print validation results
        print(f'\nValidation Results - Epoch:')
        print(f'    Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
        print(f'    Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
        print(f'    Confusion Matrix:\n{self.confusion_matrix}')
        print(f'    Time spent testing: {elapsed_time:.2f} seconds\n')

        return val_loss, val_acc, precision, recall, f1

    def train_model(self, train_loader, val_loader, model, device, optimizer, num_epochs, plot=True, verbose=True, schedulefree=False):
        """
        Trains the model on the training data and evaluates it on the validation data after each epoch.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            model (nn.Module): The PyTorch model to be trained.
            device (str): Device on which to perform training ('cpu' or 'cuda').
            optimizer (optim.Optimizer): Optimizer for training.
            num_epochs (int): Number of epochs to train the model.
            plot (bool): Whether to plot training/validation loss and accuracy after training.
            verbose (bool): Whether to print batch-level outputs for clarity.
            schedulefree (bool): Whether to use schedulefree optimizers.

        Returns:
            None
        """
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            model.train()
            if schedulefree:
                optimizer.train()
            train_loss = 0.0
            correct = 0
            total = 0

            start_time = time.time()

            # Training phase
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                predicted = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if verbose:
                    print(f'Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')

            train_loss /= len(train_loader)
            train_acc = 100 * correct / total
            elapsed_time = time.time() - start_time

            # Print epoch summary
            print(f'\nEpoch [{epoch+1}/{num_epochs}] - Training Summary:')
            print(f'    Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
            print(f'    Time spent training: {elapsed_time:.2f} seconds')

            # Store results for plotting
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)

            # Validation phase
            if schedulefree:
                optimizer.eval()
            val_loss, val_acc, precision, recall, f1 = self.test_model(val_loader, model, device)

            # Store validation and metric results for plotting
            self.val_loss_history.append(val_loss)
            self.val_acc_history.append(val_acc)
            self.precision_history.append(precision)
            self.recall_history.append(recall)
            self.f1_history.append(f1)

        # Plot results if required
        if plot:
            plot_results(self.train_loss_history, self.val_loss_history, self.train_acc_history, self.val_acc_history)

    def plot_confusion_matrix(self):
        """
        Plots the confusion matrix.
        """
        if self.confusion_matrix is not None:
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=(10, 7))
            sns.heatmap(self.confusion_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.show()
        else:
            print("No confusion matrix available. Please run the model first.")
