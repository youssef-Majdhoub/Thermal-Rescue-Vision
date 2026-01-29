import torch
from training_script import resnet50_adapted
import os


class Experimentation:
    def __init__(
        self,
        model_path,
        learning_rates=[0.001],
        optimizers=[],
        batch_sizes=[8],
        epoch_per_experiment=20,
        device="cpu",
    ):
        self.model_path = model_path
        self.dummy = resnet50_adapted(model_path=self.model_path)
        if os.path.exists(
            os.path.join(self.model_path, "experiments", "experiment_state.pt")
        ):
            self.load()
            return
        self.learning_rates = learning_rates
        if optimizers == []:
            self.optimizers = [torch.optim.Adam]
        else:
            self.optimizers = optimizers
        self.batch_sizes = batch_sizes
        self.epoch_per_experiment = epoch_per_experiment
        self.device = device
        self.opti_dex = 0
        self.size_dex = 0
        self.lr_dex = 0

    def run_experiments(self, mode="RTX3060"):
        for a in range(self.size_dex, len(self.batch_sizes)):
            size = self.batch_sizes[a]
            for b in range(self.opti_dex, len(self.optimizers)):
                optim = self.optimizers[b]
                for c in range(self.lr_dex, len(self.learning_rates)):
                    lr = self.learning_rates[c]
                    print(
                        f"Running experiment with batch size {size}, optimizer {optim.__name__}, learning rate {lr}"
                    )
                    optimizer = optim(self.dummy.model.parameters(), lr=lr)
                    self.dummy.train(
                        self.epoch_per_experiment,
                        mode=mode,
                        batch_size=size,
                        optimizer=optimizer,
                    )
                    self.lr_dex += 1
                    self.save()
                self.dummy.choose_models()
                self.lr_dex = 0
                self.opti_dex += 1
            self.opti_dex = 0
            self.size_dex += 1

    def save(self):
        save_path = os.path.join(self.model_path, "experiments")
        if not os.path.exists(os.path.join(self.model_path, "experiments")):
            os.makedirs(os.path.join(self.model_path, "experiments"))
        save_dict = {}
        save_dict["learning_rates"] = self.learning_rates
        save_dict["optimizers"] = self.optimizers
        save_dict["batch_sizes"] = self.batch_sizes
        save_dict["epoch_per_experiment"] = self.epoch_per_experiment
        save_dict["device"] = self.device
        save_dict["opti_dex"] = self.opti_dex
        save_dict["size_dex"] = self.size_dex
        save_dict["lr_dex"] = self.lr_dex
        torch.save(save_dict, os.path.join(save_path, "experiment_state.pt"))

    def load(self):
        self.dummy.load_model()
        save_path = os.path.join(self.model_path, "experiments")
        data = torch.load(os.path.join(save_path, "experiment_state.pt"))
        self.learning_rates = data["learning_rates"]
        self.optimizers = data["optimizers"]
        self.batch_sizes = data["batch_sizes"]
        self.epoch_per_experiment = data["epoch_per_experiment"]
        self.device = data["device"]
        self.opti_dex = data["opti_dex"]
        self.size_dex = data["size_dex"]
        self.lr_dex = data["lr_dex"]
