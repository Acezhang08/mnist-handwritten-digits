# One layer: 784 pixels -> 10 scores (one neuron per digit). All knobs start at 0.
import torch
from torchvision import datasets

train = datasets.MNIST('data', train=True, download=True)
X = train.data.reshape(60000, 784).float() / 255   # 60000 images, 784 numbers each (0 to 1)
Y = train.targets                                   # 60000 answers

w = torch.zeros(784, 10, requires_grad=True)        # each column = one digit's neuron
b = torch.zeros(10, requires_grad=True)
optimizer = torch.optim.SGD([w, b], lr=0.5)

for i in range(101):
    optimizer.zero_grad()
    scores = X @ w + b                                       # 60000 x 10
    probs = torch.softmax(scores, dim=1)
    loss = -torch.log(probs[range(60000), Y]).mean()         # cross-entropy
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        acc = (scores.argmax(dim=1) == Y).float().mean()
        print(i, round(loss.item(), 3), round(acc.item(), 3))
