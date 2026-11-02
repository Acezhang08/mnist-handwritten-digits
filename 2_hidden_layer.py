# Two layers: 784 pixels -> 100 hidden neurons (ReLU) -> 10 scores.
# Trained on the 60000 training images, then tested on 10000 images it never saw.
import torch
from torchvision import datasets
torch.manual_seed(0)

train = datasets.MNIST('data', train=True, download=True)
X = train.data.reshape(60000, 784).float() / 255
Y = train.targets
test = datasets.MNIST('data', train=False, download=True)
Xt = test.data.reshape(10000, 784).float() / 255
Yt = test.targets

w1 = (torch.randn(784, 100) * 0.05).requires_grad_()
b1 = torch.zeros(100, requires_grad=True)
w2 = (torch.randn(100, 10) * 0.1).requires_grad_()
b2 = torch.zeros(10, requires_grad=True)
w1_start = w1.detach().clone()                      # remember the starting point (used in 6_look_inside.py)
optimizer = torch.optim.SGD([w1, b1, w2, b2], lr=0.5)

for i in range(201):
    optimizer.zero_grad()
    h = torch.relu(X @ w1 + b1)
    scores = h @ w2 + b2
    probs = torch.softmax(scores, dim=1)
    loss = -torch.log(probs[range(60000), Y]).mean()
    loss.backward()
    optimizer.step()
    if i % 50 == 0:
        acc = (scores.argmax(dim=1) == Y).float().mean()
        # test images: only forward + accuracy. No backward, no step.
        h_t = torch.relu(Xt @ w1 + b1)
        scores_t = h_t @ w2 + b2
        acc_t = (scores_t.argmax(dim=1) == Yt).float().mean()
        print(i, 'train', round(acc.item(), 4), 'test', round(acc_t.item(), 4))

torch.save({'w1': w1.detach(), 'w1_start': w1_start, 'b1': b1.detach(),
            'w2': w2.detach(), 'b2': b2.detach()}, 'net.pt')
