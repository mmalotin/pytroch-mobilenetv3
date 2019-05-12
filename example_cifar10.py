from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from mn3.nets import MobilenetV3
from mn3.config import SMALL


def get_transforms():
    return transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
                ])


def get_data(bs=16):
    tfs = get_transforms()
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=tfs)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                              shuffle=True, num_workers=1)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=tfs)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs,
                                             shuffle=False, num_workers=1)

    return trainloader, testloader


def run_epoch(loader, model, opt, crit, device):
    r_loss = 0.0
    step = 0
    bar = tqdm(loader, desc=f'Running_loss: -')
    for x, y in bar:
        step += 1
        x = x.to(device)
        y = y.to(device)

        opt.zero_grad()

        out = model(x)

        loss = crit(out, y)
        loss.backward()
        opt.step()

        r_loss += loss.item()

        if step % 100 == 0:
            bar.set_description(f'Running_loss: {r_loss/100}')
            r_loss = 0.0


def train(epochs, loader, model, opt, crit, device, save_path='model.pth'):
    model.train()
    for e in range(epochs):
        print(f'Epoch: {e}\n')
        run_epoch(loader, model, opt, crit, device)

    torch.save(model.state_dict(), save_path)


def test(loader, model, device):
    model.eval()
    with torch.no_grad():
        bar = tqdm(loader, desc=f'Accuracy on test set')
        softmax = torch.nn.Softmax(dim=1)
        model.eval()
        xs = []
        ys = []
        for x, y in bar:
            x = x.to(device)
            y = y.view(-1)
            clf_pred = model(x)
            res = softmax(clf_pred).max(1)[1]
            xs.append(res.cpu())
            ys.append(y)

    preds = torch.cat(xs).view(-1)
    targets = torch.cat(ys).view(-1)
    acc = float((preds == targets).sum())/preds.size(0)
    print(f'Accuracy: {acc}')


if __name__ == '__main__':
    BATCH_SIZE = 16
    EPOCHS = 50
    device = torch.device('cuda')
    train_l, test_l = get_data(BATCH_SIZE)
    small_half_width = SMALL.scale_width(0.5, inplace=False)
    model = MobilenetV3(small_half_width, 10)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    train(EPOCHS, train_l, model, opt, criterion, device, 'model.pth')
    test(test_l, model, device)
