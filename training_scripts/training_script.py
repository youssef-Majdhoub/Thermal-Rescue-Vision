import torch
from data_handling import data_set_manager, data_set_server
from torchvision import models
from torchinfo import summary
import os

global_csv_file = "./archive/FLIR_ADAS_v2/labels_thermal_train.csv"
global_train_path = "./archive/FLIR_ADAS_v2/images_thermal_train"


def RTX2050_data_loader(path=global_train_path, batch_size=2, to_GPU=False):
    dataset = data_set_manager(
        csv_file=global_csv_file,
        path=path,
        to_GPU=to_GPU,
    )
    data_loader = data_set_server(
        data_set=dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    return dataset, data_loader


def RTX3060_data_loader(path=global_train_path, batch_size=8, to_GPU=False):
    dataset = data_set_manager(
        csv_file=global_csv_file,
        path=path,
        to_GPU=to_GPU,
    )
    data_loader = data_set_server(data_set=dataset, batch_size=batch_size, shuffle=True)
    return dataset, data_loader


def CPU_data_loader(path=global_train_path, batch_size=4, to_GPU=False):
    dataset = data_set_manager(
        csv_file=global_csv_file,
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
            torch.nn.init.constant_(self.model.conv1.weight, 1.0)
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
            self.epoch += 1

    def save_deployed_model(self):
        main_path = os.path.join(self.home_path, "deployment")
        if not os.path.exists(main_path):
            os.makedirs(main_path)
        model_path = os.path.join(main_path, "model_data")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
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


if __name__ == "__main__":
    dummy = resnet50_adapted(home_path=os.getcwd(), mode=0)
