import torch
from data_handling import data_set_manager, data_set_server
from torchvision import models

global_csv_file = "./archive/FLIR_ADAS_v2/labels_thermal_train.csv"
global_train_path = "./archive/FLIR_ADAS_v2/images_thermal_train"


def RTX2050_data_loader(path=global_train_path, batch_size=8, to_GPU=False):
    dataset = data_set_manager(
        csv_file=global_csv_file,
        path=path,
        to_GPU=to_GPU,
    )
    data_loader = data_set_server(
        data_set=dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    return dataset, data_loader


def RTX3060_data_loader(path=global_train_path, batch_size=24, to_GPU=True):
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
    def __init__(self, *args, **kwargs):
        self.model = models.resnet50(*args, **kwargs)
        self.model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        # IMPLEMENTATION: Force every single weight in the first layer to be 1.
        # This creates a "Pass-Through" of raw intensity.
        torch.nn.init.constant_(self.model.conv1.weight, 1.0)
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_features, 2, bias=True)
        # we default to 0 humans and 0 living creatures
        torch.nn.init.constant_(self.model.fc.weight, 0.0)
        torch.nn.init.constant_(self.model.fc.bias, 0.0)
        self.identity_application()

    def identity_application(self):
        for name, m in self.model.named_modules():
            if name == "conv1" or name.startswith("fc"):
                continue
            else:
                if hasattr(m, "weight") and m.weight is not None:
                    torch.nn.init.constant_(m.weight, 0.0)
                if hasattr(m, "bias") and m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)

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

    def train(self, mode="RTX3060"):
        if mode == "RTX3060":
            DataSet, DataLoder = RTX3060_data_loader()
            device = torch.device("cuda")
        elif mode == "RTX2050":
            DataSet, DataLoder = RTX2050_data_loader()
            device = torch.device("cuda")
        else:
            DataSet, DataLoder = CPU_data_loader()
            device = torch.device("cpu")
        self.model.to(device)
        self.device = device
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        for epoch in range(10):
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
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(DataLoder)}")
            self.evaluate_batch(outputs, targets)


training_manager = resnet50_adapted()
