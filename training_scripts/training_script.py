import torch
from data_handling import data_set_manager, data_set_server
from torchvision import models
from torchinfo import summary
import os

training_csv_file = "./data_set/human_and_living_creatures_count_data_set.csv"
evaluation_csv_file = (
    "./evaluation_set/human_and_living_creatures_count_evaluation_set.csv"
)
global_train_path = "./archive/FLIR_ADAS_v2/images_thermal_train"
global_eval_path = "./archive/FLIR_ADAS_v2/images_thermal_val"


def RTX2050_data_loader(
    path=global_train_path, csv_file=training_csv_file, batch_size=2, to_GPU=False
):
    dataset = data_set_manager(
        csv_file=csv_file,
        path=path,
        to_GPU=to_GPU,
    )
    data_loader = data_set_server(
        data_set=dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    return dataset, data_loader


def RTX3060_data_loader(
    path=global_train_path, csv_file=training_csv_file, batch_size=8, to_GPU=False
):
    dataset = data_set_manager(
        csv_file=csv_file,
        path=path,
        to_GPU=to_GPU,
    )
    data_loader = data_set_server(data_set=dataset, batch_size=batch_size, shuffle=True)
    return dataset, data_loader


def CPU_data_loader(
    path=global_train_path, csv_file=training_csv_file, batch_size=4, to_GPU=False
):
    dataset = data_set_manager(
        csv_file=csv_file,
        path=path,
        to_GPU=to_GPU,
    )
    data_loader = data_set_server(
        data_set=dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    return dataset, data_loader


class resnet50_adapted:
    def __init__(self, home_path, mode=0, *args, **kwargs):
        # mode 0: training
        # mode 1: inference
        self.home_path = os.path.abspath(home_path)
        self.mode = mode
        self.model = models.resnet50(*args, **kwargs)
        self.model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_features, 2, bias=True)
        if os.path.exists(self.home_path):
            list_of_files = os.listdir(self.home_path)
            if mode == 0 and "training" in list_of_files:
                self.load_it = True
            elif mode == 1 and "deployment" in list_of_files:
                self.load_it = True
            else:
                self.load_it = False
        else:
            self.load_it = False
        if self.load_it:
            self.load_model()
        else:
            # IMPLEMENTATION: Force every single weight in the first layer to be 1.
            # This creates a "Pass-Through" of raw intensity.
            torch.nn.init.constant_(self.model.conv1.weight, 0.0)
            if self.model.conv1.bias is not None:
                torch.nn.init.constant_(self.model.conv1.bias, 0.0)
            with torch.no_grad():
                self.model.conv1.weight[:, 0, 3, 3] = 1.0
            # we default to 0 humans and 0 living creatures
            torch.nn.init.constant_(self.model.fc.weight, 0.0)
            torch.nn.init.constant_(self.model.fc.bias, 0.0)
            self.identity_application()
            self.epoch = 0

    def identity_application(self):
        for name, m in self.model.named_modules():
            if name == "conv1" or name.startswith("fc"):
                continue
            elif isinstance(m, models.resnet.Bottleneck):
                torch.nn.init.constant_(m.bn3.weight, 0.0)

    def human_count_loss_fn(self, output, target, device):
        loss = torch.log(1 + torch.relu(output)) - torch.log(1 + target)
        loss = (100 * loss**2).mean().to(device)
        return loss

    def living_creature_count_loss_fn(self, output, target, device):
        coeff = torch.ones(output.shape).to(device)
        coeff += 9 * ((output <= 0) & (target > 0)).float()
        loss = torch.log(1 + torch.relu(output)) - torch.log(1 + target)
        loss = (coeff * loss**2).mean().to(device)
        return loss

    def the_loss_fn(self, output, target, device):
        human_loss = self.human_count_loss_fn(output[:, 0], target[:, 0], device)
        living_creature_loss = self.living_creature_count_loss_fn(
            output[:, 1], target[:, 1], device
        )
        return human_loss + living_creature_loss

    def evaluate_batch(self, outputs, targets):
        human_negatives = outputs[:, 0] < 0.5
        living_creature_negatives = outputs[:, 1] < 0.5
        false_positives_humans = ((targets[:, 0] == 0) & (outputs[:, 0] >= 0.5)).sum()
        false_positives_living_creatures = (
            (targets[:, 1] == 0) & (outputs[:, 1] >= 0.5)
        ).sum()
        false_negatives_humans = ((targets[:, 0] > 0) & (outputs[:, 0] < 0.5)).sum()
        false_negatives_living_creatures = (
            (targets[:, 1] > 0) & (outputs[:, 1] < 0.5)
        ).sum()
        print(f"False Positives Humans: {false_positives_humans.item()}")
        print(
            f"False Positives Living Creatures: {false_positives_living_creatures.item()}"
        )
        print(f"False Negatives Humans: {false_negatives_humans.item()}")
        print(
            f"False Negatives Living Creatures: {false_negatives_living_creatures.item()}"
        )

    def training_summary(self, device, input_size=(1, 1, 640, 512)):
        report = summary(
            self.model,
            input_size=input_size,
            depth=3,
            device=device,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            verbose=0,
        )
        print(f"report: {report}")

    def load_training_checkpoint(self, device):
        if self.mode != 0:
            print("Model is not in training mode.")
            return
        main_path = os.path.join(self.home_path, "training", "auxiliary_data")
        aux_file = os.path.join(main_path, f"aux_data_version{self.epoch}.pth")
        checkpoint = torch.load(aux_file, map_location=device)
        return checkpoint["optimizer_state_dict"]

    def train(self, epochs=10, mode="RTX3060"):
        if mode == "RTX3060":
            DataSet, DataLoder = RTX3060_data_loader()
            device = torch.device("cuda")
        elif mode == "RTX2050":
            DataSet, DataLoder = RTX2050_data_loader()
            device = torch.device("cuda")
        else:
            DataSet, DataLoder = CPU_data_loader()
            device = torch.device("cpu")
        if self.mode != 0:
            print("Model is not in training mode.")
            return
        self.model.to(device)
        self.device = device
        if self.load_it:
            print("Resuming training from last saved checkpoint.")
            optimizer_state_dict = self.load_training_checkpoint(device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            optimizer.load_state_dict(optimizer_state_dict)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, targets in DataLoder:
                self.model.train()
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.the_loss_fn(outputs, targets, device)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch}, Loss: {running_loss/len(DataLoder)}")
            self.evaluate_batch(outputs, targets)
            self.save_model(self.epoch, optimizer, loss.item())
            self.judge_model(self.epoch, mode=mode)
            self.epoch += 1

    def save_deployed_model(self):
        main_path = os.path.join(self.home_path, "deployment")
        if not os.path.exists(main_path):
            os.makedirs(main_path)
        model_path = os.path.join(main_path, "model_data")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.model.eval()
        torch.save(
            self.model.state_dict(),
            os.path.join(model_path, "deployed_model.pth"),
        )

    def save_model(self, epoch, optimizer, loss):
        if not os.path.exists(self.home_path):
            os.makedirs(self.home_path)
        if self.mode == 0:
            main_path = os.path.join(self.home_path, "training")
            if not os.path.exists(main_path):
                os.makedirs(main_path)
            model_path = os.path.join(main_path, "model_data")
            aux_path = os.path.join(main_path, "auxiliary_data")
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            if not os.path.exists(aux_path):
                os.makedirs(aux_path)
            torch.save(
                self.model.state_dict(),
                os.path.join(model_path, f"model_version{epoch}.pth"),
            )
            torch.save(
                {
                    "epoch": epoch,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                os.path.join(aux_path, f"aux_data_version{epoch}.pth"),
            )
        else:
            self.save_deployed_model()

    def load_model(self):
        if self.mode == 0:
            main_path = os.path.join(self.home_path, "training", "model_data")
            possible_files = os.listdir(main_path)
            latest_file = max(
                possible_files,
                key=lambda x: int(x.replace("model_version", "").replace(".pth", "")),
            )
            self.epoch = int(
                latest_file.replace("model_version", "").replace(".pth", "")
            )
            self.model.load_state_dict(
                torch.load(
                    os.path.join(main_path, latest_file),
                    map_location=torch.device("cpu"),
                )
            )
        else:
            main_path = os.path.join(self.home_path, "deployment", "model_data")
            self.model.load_state_dict(
                torch.load(
                    os.path.join(main_path, "deployed_model.pth"),
                    map_location=torch.device("cpu"),
                )
            )

    def judge_model(self, epoch, mode="RTX3060"):
        if mode == "RTX3060":
            DataSet, DataLoder = RTX3060_data_loader(
                path=global_eval_path, csv_file=evaluation_csv_file
            )
            device = torch.device("cuda")
        elif mode == "RTX2050":
            DataSet, DataLoder = RTX2050_data_loader(
                path=global_eval_path, csv_file=evaluation_csv_file
            )
            device = torch.device("cuda")
        else:
            DataSet, DataLoder = CPU_data_loader(
                path=global_eval_path, csv_file=evaluation_csv_file
            )
            device = torch.device("cpu")
        evaluation_path = os.path.join(self.home_path, "evaluation")
        if not os.path.exists(evaluation_path):
            os.makedirs(evaluation_path)
        self.model.to(device)
        # speed up evaluation by increasing batch size since no autograd is used
        DataLoder.batch_size *= 10
        juging_dict = {
            "True_Positives_Humans": 0,
            "True_Negatives_Humans": 0,
            "False_Positives_Humans": 0,
            "False_Negatives_Humans": 0,
            "True_Positives_Living_Creatures": 0,
            "True_Negatives_Living_Creatures": 0,
            "False_Positives_Living_Creatures": 0,
            "False_Negatives_Living_Creatures": 0,
            "max_loss_humans_count": 0.0,
            "max_loss_living_creatures_count": 0.0,
            "min_loss_humans_count": float("inf"),
            "min_loss_living_creatures_count": float("inf"),
            "average_loss_humans_count": 0.0,
            "average_loss_living_creatures_count": 0.0,
        }
        with torch.no_grad():
            self.model.eval()
            loss_humans_total = 0.0
            loss_living_creatures_total = 0.0
            n = 0
            for inputs, targets in DataLoder:
                n += 1
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                human_loss = self.human_count_loss_fn(
                    outputs[:, 0], targets[:, 0], device
                ).item()
                living_loss = self.living_creature_count_loss_fn(
                    outputs[:, 1], targets[:, 1], device
                ).item()
                loss_humans_total += human_loss
                loss_living_creatures_total += living_loss
                for i in range(outputs.shape[0]):
                    # Humans evaluation
                    if targets[i, 0] > 0 and outputs[i, 0] >= 0.5:
                        juging_dict["True_Positives_Humans"] += 1
                    elif targets[i, 0] == 0 and outputs[i, 0] < 0.5:
                        juging_dict["True_Negatives_Humans"] += 1
                    elif targets[i, 0] == 0 and outputs[i, 0] >= 0.5:
                        juging_dict["False_Positives_Humans"] += 1
                    elif targets[i, 0] > 0 and outputs[i, 0] < 0.5:
                        juging_dict["False_Negatives_Humans"] += 1
                    # Living Creatures evaluation
                    if targets[i, 1] > 0 and outputs[i, 1] >= 0.5:
                        juging_dict["True_Positives_Living_Creatures"] += 1
                    elif targets[i, 1] == 0 and outputs[i, 1] < 0.5:
                        juging_dict["True_Negatives_Living_Creatures"] += 1
                    elif targets[i, 1] == 0 and outputs[i, 1] >= 0.5:
                        juging_dict["False_Positives_Living_Creatures"] += 1
                    elif targets[i, 1] > 0 and outputs[i, 1] < 0.5:
                        juging_dict["False_Negatives_Living_Creatures"] += 1
                    juging_dict["max_loss_humans_count"] = max(
                        juging_dict["max_loss_humans_count"],
                        self.human_count_loss_fn(
                            outputs[i, 0], targets[i, 0], device
                        ).item(),
                    )
                    juging_dict["min_loss_humans_count"] = min(
                        juging_dict["min_loss_humans_count"],
                        self.human_count_loss_fn(
                            outputs[i, 0], targets[i, 0], device
                        ).item(),
                    )
                    juging_dict["max_loss_living_creatures_count"] = max(
                        juging_dict["max_loss_living_creatures_count"],
                        self.living_creature_count_loss_fn(
                            outputs[i, 1], targets[i, 1], device
                        ).item(),
                    )
                    juging_dict["min_loss_living_creatures_count"] = min(
                        juging_dict["min_loss_living_creatures_count"],
                        self.living_creature_count_loss_fn(
                            outputs[i, 1], targets[i, 1], device
                        ).item(),
                    )
            juging_dict["average_loss_humans_count"] = loss_humans_total / n
            juging_dict["average_loss_living_creatures_count"] = (
                loss_living_creatures_total / n
            )
        torch.save(
            juging_dict,
            os.path.join(evaluation_path, f"evaluation_data_version{epoch}.pth"),
        )

    def remove_model(self, epoch):
        training_path = os.path.join(self.home_path, "training", "model_data")
        aux_path = os.path.join(self.home_path, "training", "auxiliary_data")
        model_file = os.path.join(training_path, f"model_version{epoch}.pth")
        aux_file = os.path.join(aux_path, f"aux_data_version{epoch}.pth")
        os.remove(model_file)
        os.remove(aux_file)

    def choose_models(self):
        training_path = os.path.join(self.home_path, "training", "model_data")
        aux_path = os.path.join(self.home_path, "training", "auxiliary_data")
        eval_path = os.path.join(self.home_path, "evaluation")
        possible_files = os.listdir(eval_path)
        indexes = [
            (int(f.replace("evaluation_data_version", "").replace(".pth", "")), f)
            for f in possible_files
        ]
        data = {}
        for index, path in indexes:
            data[index] = torch.load(os.path.join(eval_path, path))
        scores = {}
        for index in data:
            scores[index] = (
                data[index]["True_Positives_Humans"]
                + data[index]["True_Negatives_Humans"]
                - data[index]["False_Positives_Humans"]
                - data[index]["False_Negatives_Humans"]
                + data[index]["True_Positives_Living_Creatures"]
                + data[index]["True_Negatives_Living_Creatures"]
                - data[index]["False_Positives_Living_Creatures"]
                - data[index]["False_Negatives_Living_Creatures"] * 10
                - data[index]["min_loss_humans_count"]
                - data[index]["min_loss_living_creatures_count"]
                - data[index]["average_loss_humans_count"]
                - data[index]["average_loss_living_creatures_count"] * 0.5
                - data[index]["max_loss_humans_count"]
                - data[index]["max_loss_living_creatures_count"] * 0.5
            )
        best_indexes = sorted(list(scores.keys()), key=scores.get, reverse=True)[:10]
        for index in data:
            if index not in best_indexes:
                self.remove_model(index)


if __name__ == "__main__":
    dummy = resnet50_adapted(home_path=os.getcwd(), mode=0)
    dummy.training_summary(device="cpu")
