import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleNet(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class ActivationNet(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(ActivationNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1), nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, out_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class BatchNet(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(BatchNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.BatchNorm1d(n_hidden_1), nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.BatchNorm1d(n_hidden_2), nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


batch_size = 64
learning_rate = 1e-2

data_tf = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])


# download data
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# model = SimpleNet(28 * 28, 300, 100, 10)
model = BatchNet(28*28, 300, 100, 10)
if torch.cuda.is_available():
    model = model.cuda()
    print(1)
else:
    print(2)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# train
epoch = 0
for data in train_loader:
    # get img and label
    img, label = data
    img = img.view(img.size(0), -1)
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()
    else:
        # transfor data to tensor
        img = Variable(img)
        label = Variable(label)

    # forward
    out = model(img)
    loss = criterion(out, label)
    print_loss = loss.data.item()

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch += 1
    if epoch % 50 == 0:
        print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))

model.eval()
# The sum of loss,the sum of acc
eval_loss = 0
eval_acc = 0
# use test to test
for data in test_loader:
    # get image
    img, label = data
    img = img.view(img.size(0), -1)
    if torch.cuda.is_available():
        img = Variable(img, volatile=True).cuda()
        label = Variable(label, volatile=True).cuda()
    else:
        image = Variable(img, volatile=True)
        label = Variable(label, volatile=True)

    # forward
    out = model(img)
    loss = criterion(out, label)

    eval_loss += loss.data.item() * label.size(0)
    # get the test resule
    _, pred = torch.max(out, 1)
    # check it
    num_correct = (pred == label).sum()

    # the sum of true result
    eval_acc += num_correct.item()

# print result
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
    eval_loss / (len(test_dataset)),
    eval_acc / (len(test_dataset))
))