import torch
from tqdm import tqdm, trange
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets.cifar import CIFAR10
from hvit.vision_transformer_block import ViT


def main(mode: str = 'gpu'):
    # Loading data
    transform = transforms.ToTensor()
    device = torch.device('cpu')

    train_set = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=256)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    # Defining training device
    if mode == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('Using GPU')
        else:
            device = torch.device('cpu')
            print('GPU is not available, using CPU')

    if mode == 'cpu':
        device = torch.device('cpu')
        print('Using CPU')

    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    # Defining model and training parameters
    model = ViT(in_channels=3,
                patch_size=4,
                emb_size=512,
                img_size=32,
                depth=4,
                n_classes=10).to(device)

    n_epochs = 100
    lr = 0.005

    # Training loop
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()
    for epoch in trange(n_epochs, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{n_epochs} loss: {train_loss:.2f}")

    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")


if __name__ == '__main__':
    main(mode='gpu')