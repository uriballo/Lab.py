import torch
import time
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from src.Plotter import cf_plot, line_plot

class Trainer:
    def __init__(self, allowed_class_idx=None, save_path="", acc_filename="acc.png", loss_filename="loss.png"):
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.precision_history = []
        self.recall_history = []
        self.f1_history = []
        self.confusion_matrix = None
        self.save_path = save_path
        self.acc_filename = acc_filename
        self.loss_filename = loss_filename
        
        labels = ['Annual\nCrop', 'Forest',
                  'Herbaceous\nVegetation',
                  'Highway', 'Industrial',
                  'Pasture', 'Permanent\nCrop',
                  'Residential', 'River',
                  'SeaLake']
        
        self.labels = labels if allowed_class_idx is None else [labels[i] for i in allowed_class_idx if i < len(labels)]
        
        self.best_val_acc = 0  # Initialize best validation accuracy
        self.best_model_state = None  # To store the best model state

    def test_model(self, val_loader, model, device):
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        criterion = torch.nn.CrossEntropyLoss()

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

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        elapsed_time = time.time() - start_time

        precision = precision_score(all_labels, all_preds, average='macro', zero_division=1)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=1)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=1)

        self.confusion_matrix = confusion_matrix(all_labels, all_preds)

        print(f'\nValidation Results - Epoch:')
        print(f'    Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
        print(f'    Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
        print(f'    Confusion Matrix:\n{self.confusion_matrix}')
        print(f'    Time spent testing: {elapsed_time:.2f} seconds\n')

        return val_loss, val_acc, precision, recall, f1

    def save_best_model(self, model):
        # Save the model state
        torch.save(model.state_dict(), f"{self.save_path}/best_model.pth")
        print("Best model saved.")

    def train_model(self, train_loader, val_loader, model, device, optimizer, num_epochs, plot=True, verbose=True, schedulefree=False):
        model = model.to(device)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            model.train()
            if schedulefree:
                optimizer.train()
            train_loss = 0.0
            correct = 0
            total = 0

            start_time = time.time()

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
                    print(f'Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}')

            train_loss /= len(train_loader)
            train_acc = 100 * correct / total
            elapsed_time = time.time() - start_time

            print(f'\nEpoch [{epoch + 1}/{num_epochs}] - Training Summary:')
            print(f'    Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
            print(f'    Time spent training: {elapsed_time:.2f} seconds')

            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)

            if schedulefree:
                optimizer.eval()
            val_loss, val_acc, precision, recall, f1 = self.test_model(val_loader, model, device)

            self.val_loss_history.append(val_loss)
            self.val_acc_history.append(val_acc)
            self.precision_history.append(precision)
            self.recall_history.append(recall)
            self.f1_history.append(f1)

            # Check if the current validation accuracy is better than the best one
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc  # Update best validation accuracy
                self.best_model_state = model.state_dict()  # Save the current model state

        # Save the best model at the end of training
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)  # Load the best model state
            self.save_best_model(model)

        # Plot results if required
        if plot:
            line_plot("Accuracy", [self.train_acc_history, self.val_acc_history], 
                      ["Train Accuracy", "Test Accuracy"], "Epoch", "(%)", self.save_path, self.acc_filename)
            line_plot("Loss", [self.train_loss_history, self.val_loss_history], 
                      ["Train Loss", "Test Loss"], "Epoch", "", self.save_path, self.loss_filename)
            cf_plot(self.confusion_matrix, self.labels)
