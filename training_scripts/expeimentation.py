import torch
from training_script import resnet50_adapted
import os
import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt


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
        self.running_time_per_experiment = []

    def run_experiments(self, mode="RTX3060"):
        for a in range(self.size_dex, len(self.batch_sizes)):
            size = self.batch_sizes[a]
            for b in range(self.opti_dex, len(self.optimizers)):
                optim = self.optimizers[b]
                t1 = time.perf_counter()
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
                t2 = time.perf_counter()
                self.running_time_per_experiment.append(t2 - t1)
                self.dummy.choose_models()
                self.lr_dex = 0
                self.opti_dex += 1
            self.opti_dex = 0
            self.size_dex += 1
        print("All experiments completed.")

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
        save_dict["running_time_per_experiment"] = self.running_time_per_experiment
        torch.save(save_dict, os.path.join(save_path, "experiment_state.pt"))

    def load(self):
        self.dummy.load_model()
        save_path = os.path.join(self.model_path, "experiments")
        data = torch.load(
            os.path.join(save_path, "experiment_state.pt"), weights_only=False
        )
        self.learning_rates = data["learning_rates"]
        self.optimizers = data["optimizers"]
        self.batch_sizes = data["batch_sizes"]
        self.epoch_per_experiment = data["epoch_per_experiment"]
        self.device = data["device"]
        self.opti_dex = data["opti_dex"]
        self.size_dex = data["size_dex"]
        self.lr_dex = data["lr_dex"]
        self.running_time_per_experiment = data["running_time_per_experiment"]
        current_epoch = (
            self.size_dex
            * len(self.optimizers)
            * len(self.learning_rates)
            * self.epoch_per_experiment
            + self.opti_dex * len(self.learning_rates) * self.epoch_per_experiment
            + self.lr_dex * self.epoch_per_experiment
        )
        self.dummy.epoch = current_epoch

    def create_df(self):
        eval_path = os.path.join(self.model_path, "evaluation")
        if not os.path.exists(eval_path):
            print("No evaluation directory found.")
            return
        all_files = os.listdir(eval_path)
        for i in range(len(all_files) - 1, -1, -1):
            if "evaluation_data_version" not in all_files[i]:
                all_files.pop(i)
        indexes = np.array(
            [
                int(name.replace("evaluation_data_version", "").replace(".pth", ""))
                for name in all_files
            ],
            dtype=int,
        )
        sorted_indexes = np.argsort(indexes)
        indexes = indexes[sorted_indexes]
        all_files = np.array(all_files)[sorted_indexes]
        df_dict = {"epoch": []}
        for i in range(len(all_files)):
            file = all_files[i]
            data = torch.load(os.path.join(eval_path, file))
            df_dict["epoch"].append(indexes[i])
            for key in data.keys():
                if key not in df_dict:
                    df_dict[key] = []
                df_dict[key].append(data[key])
        return pd.DataFrame(df_dict)

    def create_epoch_description(self):
        c = 0
        re = {}
        for size in self.batch_sizes:
            for optim in self.optimizers:
                for lr in self.learning_rates:
                    for epoch in range(self.epoch_per_experiment):
                        re[c] = [size, optim.__name__, lr]
                        c += 1
        return re

    def show_experiment_order(self):
        description = self.create_epoch_description()
        experiment_order = {
            "experiment_number": [],
            "starting_index": [],
            "experiment_length(in epochs)": [],
            "batch_size": [],
            "optimizer": [],
            "learning_rate": [],
        }
        c = 0
        for i in range(0, len(description), self.epoch_per_experiment):
            experiment_order["experiment_number"].append(c)
            experiment_order["starting_index"].append(i)
            experiment_order["experiment_length(in epochs)"].append(
                self.epoch_per_experiment
            )
            experiment_order["batch_size"].append(description[i][0])
            experiment_order["optimizer"].append(description[i][1])
            experiment_order["learning_rate"].append(description[i][2])
            c += 1
        re = pd.DataFrame(experiment_order)
        print(re)
        return re

    def get_mean(self, df, description, wanted="average_loss_humans_count"):
        mean_accuracies = []
        for i in range(0, len(description), self.epoch_per_experiment):
            mean_accuracy = df[wanted][i : i + self.epoch_per_experiment].mean()
            mean_accuracies.append(mean_accuracy)
        return np.array(mean_accuracies)

    def general_report(self):
        df = self.create_df()
        description = self.create_epoch_description()
        report_path = os.path.join(self.model_path, "report")
        if not os.path.exists(report_path):
            os.makedirs(report_path)
        assets_path = os.path.join(report_path, "assets")
        if not os.path.exists(assets_path):
            os.makedirs(assets_path)
        data_path = os.path.join(report_path, "data")
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        general_report_path = os.path.join(report_path, "general_report.md")
        df.to_csv(os.path.join(data_path, "experiment_data.csv"), index=False)
        with open(general_report_path, "w") as f:
            f.write("# General Report\n\n")
            f.write("## Experiment Order\n\n")
            experiment_order = self.show_experiment_order()
            f.write(experiment_order.to_markdown())
            f.write("\n\n")
            f.write("## report directory structure\n\n")
            f.write(
                f"""{report_path}:includes the general report and assets/data directories\n"""
            )
            f.write(
                f"""{assets_path}:includes all the plots generated needed in the report\n"""
            )
            f.write(
                f"""{data_path}:includes all the data files needed in the report\n"""
            )
            f.write("\n\n")
            f.write("## Experiments Summary\n\n")
            f.write(f"we tested {len(experiment_order)} experiments\n\n")
            f.write(
                f"""testing all the possible combinations of 
                    batch sizes :{self.batch_sizes},
                    optimizers :{[optim.__name__ for optim in self.optimizers]},
                    learning rates :{self.learning_rates}\n\n"""
            )
            f.write("## Running Time per Experiment\n\n")
            time_plot = plt.plot(self.running_time_per_experiment)
            plt.xlabel("Experiment Number")
            plt.ylabel("Time (seconds)")
            plt.title("Global Timeline Graph")
            plt.savefig(os.path.join(assets_path, "running_time_per_experiment.png"))
            f.write(
                "![Global Timeline Graph](assets/running_time_per_experiment.png)\n\n"
            )
            plt.clf()
            f.write(
                "## Detailed Experiments Results: could be found  in data/experiment_data.csv\n\n"
            )
            f.write(
                "we will focus on mean accuracy and mean loss over the experiments\n\n"
            )
            mean_human_count_accuracy = self.get_mean(
                df, description, wanted="True_Positive_Humans"
            ) + self.get_mean(df, description, wanted="True_Negative_Humans")
            mean_human_count_accuracy /= (
                mean_human_count_accuracy
                + self.get_mean(df, description, wanted="False_Positive_Humans")
                + self.get_mean(df, description, wanted="False_Negative_Humans")
            )
            mean_loss_human_count = self.get_mean(
                df, description, wanted="average_loss_humans_count"
            )
            plt.plot(mean_human_count_accuracy)
            plt.xlabel("Experiment Number")
            plt.ylabel("Mean Accuracy")
            plt.title("Mean Human Count Accuracy per Experiment")
            plt.savefig(os.path.join(assets_path, "mean_human_count_accuracy.png"))
            f.write("### Mean Human Count Accuracy per Experiment\n\n")
            f.write(
                "![Mean Human Count Accuracy](assets/mean_human_count_accuracy.png)\n\n"
            )
            plt.clf()
            plt.plot(mean_loss_human_count)
            plt.xlabel("Experiment Number")
            plt.ylabel("Mean Loss")
            plt.title("Mean Human Count Loss per Experiment")
            plt.savefig(os.path.join(assets_path, "mean_human_count_loss.png"))
            f.write("### Mean Human Count Loss per Experiment\n\n")
            f.write("![Mean Human Count Loss](assets/mean_human_count_loss.png)\n\n")
            plt.clf()
            mean_living_count_accuracy = self.get_mean(
                df, description, wanted="True_Positive_Living_Creatures"
            ) + self.get_mean(df, description, wanted="True_Negative_Living_Creatures")
            mean_living_count_accuracy /= (
                mean_living_count_accuracy
                + self.get_mean(
                    df, description, wanted="False_Positive_Living_Creatures"
                )
                + self.get_mean(
                    df, description, wanted="False_Negative_Living_Creatures"
                )
            )
            mean_loss_living_count = self.get_mean(
                df, description, wanted="average_loss_living_creatures_count"
            )
            plt.plot(mean_living_count_accuracy)
            plt.xlabel("Experiment Number")
            plt.ylabel("Mean Accuracy")
            plt.title("Mean Living Creatures Count Accuracy per Experiment")
            plt.savefig(os.path.join(assets_path, "mean_living_count_accuracy.png"))
            f.write("### Mean Living Creatures Count Accuracy per Experiment\n\n")
            f.write(
                "![Mean Living Creatures Count Accuracy](assets/mean_living_count_accuracy.png)\n\n"
            )
            plt.clf()
            plt.plot(mean_loss_living_count)
            plt.xlabel("Experiment Number")
            plt.ylabel("Mean Loss")
            plt.title("Mean Living Creatures Count Loss per Experiment")
            plt.savefig(os.path.join(assets_path, "mean_living_count_loss.png"))
            f.write("### Mean Living Creatures Count Loss per Experiment\n\n")
            f.write(
                "![Mean Living Creatures Count Loss](assets/mean_living_count_loss.png)\n\n"
            )
            plt.clf()
            total_accuracy = (
                mean_human_count_accuracy + mean_living_count_accuracy
            ) / 2
            total_loss = (mean_loss_human_count + mean_loss_living_count) / 2
            plt.plot(total_accuracy)
            plt.xlabel("Experiment Number")
            plt.ylabel("Mean Accuracy")
            plt.title("Total Mean Accuracy per Experiment")
            plt.savefig(os.path.join(assets_path, "total_mean_accuracy.png"))
            f.write("### Total Mean Accuracy per Experiment\n\n")
            f.write("![Total Mean Accuracy](assets/total_mean_accuracy.png)\n\n")
            plt.clf()
            plt.plot(total_loss)
            plt.xlabel("Experiment Number")
            plt.ylabel("Mean Loss")
            plt.title("Total Mean Loss per Experiment")
            plt.savefig(os.path.join(assets_path, "total_mean_loss.png"))
            f.write("### Total Mean Loss per Experiment\n\n")
            f.write("![Total Mean Loss](assets/total_mean_loss.png)\n\n")
            plt.clf()
            f.write("## End of Report\n")


if __name__ == "__main__":
    experiment = Experimentation(
        model_path=os.getcwd(),
        learning_rates=[0.1, 0.001, 0.0001, 0.00001],
        optimizers=[torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW],
        batch_sizes=[6, 8],
        epoch_per_experiment=20,
        device="cuda",
    )
    experiment.run_experiments(mode="RTX3060")
    experiment.general_report()
