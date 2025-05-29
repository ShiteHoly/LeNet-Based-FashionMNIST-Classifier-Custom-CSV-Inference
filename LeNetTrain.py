#%% md
# # Classic LeNet CNN to classify handwritten numbers
#%%
import torch
from torch import nn
from torchvision import datasets, transforms

class Reshape(torch.nn.Module):
    def forward (self, x):
        return x.view(-1,1,28,28)#bactch size unknown here, using -1

net = torch.nn.Sequential(
    Reshape(), nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2,stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*5*5, 120),nn.Sigmoid(),
    nn.Linear(120,84), nn.Sigmoid(),
    nn.Linear(84, 10)
)
#%%
# show what each layer ouputs
X = torch.rand(size = (1,1,28,28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
#%% md
# ![LeNet](lenet.jpg)
#%%
batch_size = 256
train_dataset = datasets.FashionMNIST(
    root='data', train=True, download=True,
    transform=transforms.ToTensor())
test_dataset = datasets.FashionMNIST(
    root='data', train=False, download=True,
    transform=transforms.ToTensor())

train_iter = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)
#%%
def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = torch.zeros(2, device=device)  # metric[0]: correct, metric[1]: total
    for X, y in data_iter:
        if isinstance(X,list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        with torch.no_grad():
            y_hat = net(X)
        metric[0] += (y_hat.argmax(dim=1) == y).sum().float()
        metric[1] += y.numel()
    return metric[0] / metric[1]
#%% md
# # Summary of important variables
# 
# ### Inputs:
# **X**: Input tensor of images, shape ([batch_size, 1, 28, 28]), dtype float32. Each image is a grayscale FashionMNIST image.
# 
# **y**: Target tensor of labels, shape ([batch_size]), dtype int64. Each value is an integer from 0 to 9, representing the class.
# 
# **net**: The neural network (LeNet), which takes X and outputs class scores.
# 
# **data_iter**: DataLoader yielding batches of (X, y).
# 
# **device**: The device (CPU or GPU) on which tensors and the model are located.
# 
# ### Outputs:
# **y_hat**: Output tensor from the network, shape ([batch_size, 10]), dtype float32. Each row contains the (unnormalized) scores for each class.
# 
# **y_hat.argmax(dim=1)**: Tensor of predicted class indices, shape ([batch_size]), dtype int64. Each value is an integer from 0 to 9.
# 
# **metric**: A tensor of shape ([2]), dtype float32. metric[0] is the count of correct predictions, metric[1] is the total number of samples processed.
# 
# **evaluate_accuracy_gpu(...)**: Returns a scalar tensor representing the accuracy (correct predictions / total samples), dtype float32. Value is between 0 and 1.
#%%
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print("Training on", device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        net.train()
        metric = torch.zeros(2, device=device)  # metric[0]: correct, metric[1]: total
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            metric[0] += (y_hat.argmax(dim=1) == y).sum().float()
            metric[1] += y.numel()
        test_acc = evaluate_accuracy_gpu(net, test_iter, device)
        print(f'epoch {epoch + 1}, loss {l:f}, train acc {metric[0] / metric[1]:f}, '
              f'test acc {test_acc:f}')
#%%
num_epochs, lr = 10, 0.9
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
#%%
# Save the model
torch.save(net.state_dict(), 'lenet.pth')