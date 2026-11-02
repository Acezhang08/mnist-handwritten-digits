# Same network, three learning rates.
import torch
from torchvision import datasets
import matplotlib.pyplot as plt

train = datasets.MNIST('data', train=True, download=True)
X = train.data.reshape(60000, 784).float() / 255
Y = train.targets

for lr in [0.005, 0.5, 3]:
    torch.manual_seed(0)
    w1 = (torch.randn(784, 100) * 0.05).requires_grad_()
    b1 = torch.zeros(100, requires_grad=True)
    w2 = (torch.randn(100, 10) * 0.1).requires_grad_()
    b2 = torch.zeros(10, requires_grad=True)
    optimizer = torch.optim.SGD([w1, b1, w2, b2], lr=lr)
    losses = []
    for i in range(201):
        optimizer.zero_grad()
        h = torch.relu(X @ w1 + b1)
        scores = h @ w2 + b2
        probs = torch.softmax(scores, dim=1)
        loss = -torch.log(probs[range(60000), Y]).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    acc = (scores.argmax(dim=1) == Y).float().mean()
    print('lr', lr, 'first 6 losses', [round(l, 2) for l in losses[:6]],
          'last loss', round(losses[-1], 3), 'accuracy', round(acc.item(), 3))
    plt.plot(losses, label=f'lr = {lr}')

plt.ylim(0, 9); plt.xlabel('round'); plt.ylabel('training loss'); plt.legend()
plt.title('Too small: slow. Too big: jumps, then inf / nan')
plt.savefig('learning_rate.png', dpi=110)
