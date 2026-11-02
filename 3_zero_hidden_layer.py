# Broken on purpose: same two-layer network, but every knob starts at 0.
import torch
from torchvision import datasets

train = datasets.MNIST('data', train=True, download=True)
X = train.data.reshape(60000, 784).float() / 255
Y = train.targets

w1 = torch.zeros(784, 100, requires_grad=True)
b1 = torch.zeros(100, requires_grad=True)
w2 = torch.zeros(100, 10, requires_grad=True)
b2 = torch.zeros(10, requires_grad=True)
optimizer = torch.optim.SGD([w1, b1, w2, b2], lr=0.5)

for i in range(201):
    optimizer.zero_grad()
    h = torch.relu(X @ w1 + b1)          # all 0
    scores = h @ w2 + b2
    probs = torch.softmax(scores, dim=1)
    loss = -torch.log(probs[range(60000), Y]).mean()
    loss.backward()
    optimizer.step()
    if i % 50 == 0:
        acc = (scores.argmax(dim=1) == Y).float().mean()
        print(i, round(loss.item(), 3), round(acc.item(), 4))

print('share of digit 1 in training set:', round((Y == 1).float().mean().item(), 4))
