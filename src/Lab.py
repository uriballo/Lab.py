import json

class ExperimentConfig:
    def __init__(self, path_to_save_plots, path_to_save_raw_data, path_to_dataset, experiments, checkpointing):
        self.path_to_save_plots = path_to_save_plots
        self.path_to_save_raw_data = path_to_save_raw_data
        self.path_to_dataset = path_to_dataset
        self.experiments = experiments
        self.checkpointing = checkpointing

class Experiment:
    def __init__(self, filename, title, subtitle, optimizer, lr, weight_decay, n_classes, image_size, batch_size, allowed_classes, examples_per_class, epochs, confusion_mat):
        self.filename = filename
        self.title = title
        self.subtitle = subtitle
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_classes = n_classes
        self.image_size = image_size
        self.batch_size = batch_size
        self.allowed_classes = allowed_classes
        self.examples_per_class = examples_per_class
        self.epochs = epochs
        self.confusion_mat = confusion_mat

class Checkpointing:
    def __init__(self, save_best, monitor, save_path):
        self.save_best = save_best
        self.monitor = monitor
        self.save_path = save_path
        
def load_config_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        
        # Parse experiments
        experiments = [
            Experiment(
                filename=exp['filename'],
                title=exp['title'],
                subtitle=exp['subtitle'],
                optimizer=exp['optimizer'],
                lr=exp['lr'],
                weight_decay=exp['weight_decay'],
                n_classes=exp['n_classes'],
                image_size=exp['image_size'],
                batch_size=exp['batch_size'],
                allowed_classes=exp['allowed_classes'],
                examples_per_class=exp['examples_per_class'],
                epochs=exp['epochs'],
                confusion_mat=exp['confusion_mat']
            ) for exp in data['experiments']
        ]
        
        # Parse checkpointing
        checkpointing = Checkpointing(
            save_best=data['checkpointing']['save_best'],
            monitor=data['checkpointing']['monitor'],
            save_path=data['checkpointing']['save_path']
        )
        
        # Create and return the ExperimentConfig instance
        return ExperimentConfig(
            path_to_save_plots=data['path_to_save_plots'],
            path_to_save_raw_data=data['path_to_save_raw_data']
            path_to_dataset=data['path_to_dataset'],
            experiments=experiments,
            checkpointing=checkpointing
        )