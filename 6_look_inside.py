# Look inside the network trained by 2_hidden_layer.py (run that first; it saves net.pt).
import torch
from torchvision import datasets
import matplotlib.pyplot as plt

net = torch.load('net.pt')
w1, w1_start = net['w1'], net['w1_start']

# 1) What each hidden neuron "likes": its 784 weights put back into a 28 x 28 picture.
#    red = positive, blue = negative.
for name, M in [('neurons_raw', w1), ('neurons_changed', w1 - w1_start)]:
    fig, ax = plt.subplots(2, 8, figsize=(12, 3.4))
    for k, a in enumerate(ax.flat):
        img = M[:, k].reshape(28, 28)
        m = img.abs().max()
        a.imshow(img, cmap='RdBu_r', vmin=-m, vmax=m)
        a.axis('off'); a.set_title(f'neuron {k}', fontsize=9)
    plt.tight_layout(); plt.savefig(name + '.png', dpi=100); plt.close()

# 2) Test images the network gets wrong.
test = datasets.MNIST('data', train=False, download=True)
Xt = test.data.reshape(10000, 784).float() / 255
Yt = test.targets
probs_t = torch.softmax(torch.relu(Xt @ w1 + net['b1']) @ net['w2'] + net['b2'], dim=1)
pick = probs_t.argmax(dim=1)
wrong = (pick != Yt).nonzero().flatten()
print('wrong on test set:', len(wrong), 'of 10000')

fig, ax = plt.subplots(3, 6, figsize=(9, 5.4))
for a, idx in zip(ax.flat, wrong[:18]):
    a.imshow(test.data[idx], cmap='gray_r'); a.axis('off')
    a.set_title(f'answer {Yt[idx].item()}, picked {pick[idx].item()}\n({probs_t[idx, pick[idx]].item() * 100:.0f}% sure)', fontsize=9)
plt.tight_layout(); plt.savefig('wrong.png', dpi=110)
