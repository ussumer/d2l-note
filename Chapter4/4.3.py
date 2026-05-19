import matplotlib
import torch
from torch import nn

matplotlib.use("Agg")

from d2l import torch as d2l

# Disable inline SVG display helpers when running as a script.
d2l.use_svg_display = lambda *args, **kwargs: None
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def main():
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )

    net.apply(init_weights)
    batch_size, lr, num_epochs = 256, 0.1, 10
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    # Avoid multiprocessing on Windows when running as a script.
    d2l.get_dataloader_workers = lambda: 0
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


if __name__ == "__main__":
    main()