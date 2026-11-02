# Overfitting on purpose: ~80,000 knobs, only 500 training images, 2000 rounds.
import torch
from torchvision import datasets
import matplotlib.pyplot as plt
torch.manual_seed(0)

N = 500
train = datasets.MNIST('data', train=True, download=True)
X = train.data[:N].reshape(N, 784).float() / 255
Y = train.targets[:N]
test = datasets.MNIST('data', train=False, download=True)
Xt = test.data.reshape(10000, 784).float() / 255
Yt = test.targets

w1 = (torch.randn(784, 100) * 0.05).requires_grad_()
b1 = torch.zeros(100, requires_grad=True)
w2 = (torch.randn(100, 10) * 0.1).requires_grad_()
b2 = torch.zeros(10, requires_grad=True)
optimizer = torch.optim.SGD([w1, b1, w2, b2], lr=0.5)

rounds, train_losses, test_losses = [], [], []
for i in range(2001):
    optimizer.zero_grad()
    h = torch.relu(X @ w1 + b1)
    scores = h @ w2 + b2
    probs = torch.softmax(scores, dim=1)
    loss = -torch.log(probs[range(N), Y]).mean()
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        with torch.no_grad():   # test images: no slopes needed
            scores_t = torch.relu(Xt @ w1 + b1) @ w2 + b2
            loss_t = -torch.log(torch.softmax(scores_t, dim=1)[range(10000), Yt]).mean()
        rounds.append(i); train_losses.append(loss.item()); test_losses.append(loss_t.item())
        if i in (0, 50, 100, 500, 1000, 2000):
            acc = (scores.argmax(dim=1) == Y).float().mean()
            acc_t = (scores_t.argmax(dim=1) == Yt).float().mean()
            print(i, 'train', round(loss.item(), 4), round(acc.item(), 3),
                  'test', round(loss_t.item(), 3), round(acc_t.item(), 4))

plt.plot(rounds, train_losses, label='train loss (500 images)')
plt.plot(rounds, test_losses, label='test loss (10000 unseen images)')
plt.xlabel('round'); plt.ylabel('loss'); plt.legend(); plt.title('Overfitting: train loss goes down, test loss goes up')
plt.savefig('overfit.png', dpi=110)
