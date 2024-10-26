import json
import torch.optim as optim
import schedulefree as sf_optim
from src.Trainer import Trainer
from src.EuroSAT import EuroSAT
from src.ModelInstances import choose_model

class ExperimentConfig:
    def __init__(self, path_to_save_plots, path_to_dataset, experiments, checkpointing):
        self.path_to_save_plots = path_to_save_plots
        self.path_to_dataset = path_to_dataset
        self.experiments = experiments
        self.checkpointing = checkpointing

class Experiment:
    def __init__(self, filename, title, model, optimizer, lr, weight_decay, n_classes, image_size, batch_size, allowed_classes, examples_per_class, epochs):
        self.filename = filename
        self.title = title
        self.optimizer = optimizer
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_classes = n_classes
        self.image_size = image_size
        self.batch_size = batch_size
        self.allowed_classes = None # TODO: Convert allowed classes string to list of integers.
        self.examples_per_class = examples_per_class
        self.epochs = epochs

class Checkpointing:
    def __init__(self, save_best, monitor, save_path):
        self.save_best = save_best
        self.monitor = monitor
        self.save_path = save_path
        
class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def initialize_model(self, model_name, num_classes):
        return choose_model(model_name, num_classes)
    
    def initialize_optimizer(self, optimizer_name, model_params, lr, weight_decay):
        if optimizer_name == "Adam":
            return optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "SGD":
            return optim.SGD(model_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "AdamWSF":
            return sf_optim.AdamWScheduleFree(model_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "SGDSF":
            return sf_optim.SGDScheduleFree(model_params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def run_experiment(self, experiment: Experiment):
        # Setup paths
        save_path = f"{self.config.path_to_save_plots}/"
        
        # Initialize model, optimizer, and dataset loaders
        model = self.initialize_model(experiment.model, experiment.n_classes)
        optimizer = self.initialize_optimizer(experiment.optimizer, model.parameters(), experiment.lr, experiment.weight_decay)
        
        # Load dataset and split into train/val sets
        train_loader, val_loader = EuroSAT(
            root=self.config.path_to_dataset,
            batch_size=experiment.batch_size,
            image_size=experiment.image_size,
            allowed_classes=experiment.allowed_classes,
            examples_per_class=experiment.examples_per_class,
            num_classes=experiment.n_classes
        ).get_loaders()
        
        # Initialize Trainer with the experiment's parameters
        trainer = Trainer(
            save_path=save_path,
            acc_filename=f"{experiment.filename}_acc.png",
            loss_filename=f"{experiment.filename}_loss.png",
            cf_filename=f"{experiment.filename}_cf.png" 
        )
        
        # Train model
        trainer.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            device="cpu",
            optimizer=optimizer,
            num_epochs=experiment.epochs,
            plot=True,
            verbose=True
        )

    def run_all_experiments(self):
        for experiment in self.config.experiments:
            print(f"Running experiment: {experiment.title}")
            self.run_experiment(experiment)
        
def load_config_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        
        # Parse experiments
        experiments = [
            Experiment(
                filename=exp['filename'],
                title=exp['title'],
                model=exp['model'],
                optimizer=exp['optimizer'],
                lr=exp['lr'],
                weight_decay=exp['weight_decay'],
                n_classes=exp['n_classes'],
                image_size=exp['image_size'],
                batch_size=exp['batch_size'],
                allowed_classes=exp['allowed_classes'],
                examples_per_class=exp['examples_per_class'],
                epochs=exp['epochs'],
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
            path_to_dataset=data['path_to_dataset'],
            experiments=experiments,
            checkpointing=checkpointing
        )