import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def line_plot(title, metrics, labels, xlabel, ylabel, save_path, filename):
    """
    Plots multiple metrics over epochs with different colors and markers.

    Args:
        title (str): The main title of the plot.
        metrics (list of lists): A list containing lists of metric values per epoch.
        labels (list of str): A list of labels corresponding to each metric.
        save_path (str): Directory path to save the plot.
        filename (str): Filename for the saved plot.
    """
    epochs = range(1, len(metrics[0]) + 1)

    plt.figure(figsize=(8, 8))
    
    # Define colors and markers for plotting
    colors = plt.cm.bwr(np.linspace(0, 1, len(metrics)))
    markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'X', 'H', '<', '>']

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        plt.plot(epochs, metric, color=colors[i], marker=markers[i % len(markers)], 
                 label=label, linewidth=2)
    
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(epochs)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path + filename + '.png', bbox_inches='tight')
    
def cf_plot(confusion_matrix, labels, title='Confusion Matrix', save_path='', filename='confusion_matrix'):
    plt.figure(figsize=(8, 6))
    
    # Plotting the confusion matrix with heatmap
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                 xticklabels=labels, yticklabels=labels, cbar=False)

    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path + filename + '.png', bbox_inches='tight')
    #plt.show()

