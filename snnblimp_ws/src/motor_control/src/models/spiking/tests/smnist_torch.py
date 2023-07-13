import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

sys.path.append(".")
from spiking.torch.layers.linear import RecurrentLinearLIF
from spiking.torch.utils.surrogates import get_spike_fn


def init_lif_params(*shape):
    params = dict(
        leak_i=torch.ones(shape) * 0.9,
        leak_v=torch.ones(shape) * 0.9,
        thresh=torch.ones(shape),
    )
    return params


class Model(nn.Module):
    """
    Simple recurrent linear spiking model.
    """

    def __init__(self):
        super().__init__()

        self.layer1 = RecurrentLinearLIF(1, 100, {}, init_lif_params(100), get_spike_fn("ArcTan", 1.0, 10.0))
        self.layer2 = nn.Linear(100, 10)

    def forward(self, states, input_):
        b, _ = input_.shape
        out_states = [None] * len(states)

        out_states[0], s = self.layer1(states[0], input_)
        out = self.layer2(s)

        return out_states, out

    @staticmethod
    def reset_state():
        return [None] * 1


def model_step(model, data, target):
    state = model.reset_state()
    for input_ in data.unbind(1):
        state, logits = model(state, input_)
    loss = F.cross_entropy(logits, target)
    accuracy = torch.eq(torch.argmax(logits, dim=1), target).float().mean()
    return loss, accuracy


def train_epoch(model, train_loader, optimizer, device):
    model.train()

    epoch_loss = 0
    epoch_accuracy = 0

    for data, target in train_loader:

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        batch_loss, batch_accuracy = model_step(model, data, target)
        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item()
        epoch_accuracy += batch_accuracy.item()

    return epoch_loss / len(train_loader), epoch_accuracy / len(train_loader)


def eval_model(model, test_loader, device):
    model.eval()

    epoch_loss = 0
    epoch_accuracy = 0

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            batch_loss, batch_accuracy = model_step(model, data, target)

            epoch_loss += batch_loss.item()
            epoch_accuracy += batch_accuracy.item()

    return epoch_loss / len(test_loader), epoch_accuracy / len(test_loader)


def to_flat(img):
    return img.view(-1, 1)


def train_and_evaluate(config: dict, workdir: str):
    # seeding
    torch.manual_seed(0)

    train_ds = MNIST("data", download=True, train=True, transform=Compose([ToTensor(), to_flat]))
    test_ds = MNIST("data", download=True, train=False, transform=Compose([ToTensor(), to_flat]))
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    summary_writer = SummaryWriter(workdir)
    summary_writer.add_hparams(config, {})

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = Model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"])

    for epoch in range(1, config["num_epochs"] + 1):
        loop_time = time.time()

        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, device)
        train_time = time.time()

        test_loss, test_accuracy = eval_model(model, test_loader, device)
        eval_time = time.time()

        print(
            f"epoch:{epoch:3}, time (train/eval): {train_time - loop_time:.2f}s / {eval_time - train_time:.2f}s, train_loss: {train_loss:.4f}, train_accuracy: {train_accuracy * 100:.2f}, test_loss: {test_loss:.4f}, test_accuracy: {test_accuracy * 100:.2f}"
        )

        summary_writer.add_scalar("train_loss", train_loss, epoch)
        summary_writer.add_scalar("train_accuracy", train_accuracy, epoch)
        summary_writer.add_scalar("test_loss", test_loss, epoch)
        summary_writer.add_scalar("test_accuracy", test_accuracy, epoch)

    summary_writer.flush()
    summary_writer.close()


if __name__ == "__main__":
    config = dict(
        learning_rate=0.001,
        momentum=0.9,
        batch_size=1024,
        num_epochs=20,
    )

    train_and_evaluate(config, "./logs")
