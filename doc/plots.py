#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(
    style="white",
    context="paper",
    rc={"text.usetex": True},
)
x = np.linspace(-10, 10, 100)
y = np.square(x)

plt.plot(x,y, linestyle=':')
plt.plot(8, 64, 'o')
for lr in [1.1, 0.2, 0.8]:
    x0 = 8
    y0 = np.square(x0)
    x1 = [x0]
    y1 = [y0]
    for epoch in range(100):
        x0 = x0-2*lr*x0
        if abs(x0)>10:
            break
        y0 = x0**2
        x1.append(x0)
        y1.append(y0)
    plt.plot(x1,y1, label=f"lr={lr}")
plt.legend()
# plt.savefig("../lib/lr.png")
#%%
import torch
x = torch.rand([500,1]) # X 是一个 tensor ，可以把他想象成 500x1 的向量
y_true = 3*x+8
learning_rate = 0.1 # learning rate 是每次梯度下降的“步长”
w = torch.tensor([[0.1]], requires_grad=True)
# w = torch.rand([1,1], requires_grad=True) # w 和 b 我们要 pytorch 自动求导
b = torch.tensor(0, requires_grad=True, dtype=torch.float32)

for i in range(1000):
    y_pred = torch.matmul(x,w)+b # 预测是多少
    loss = (y_true-y_pred).pow(2).mean() # 损失
    if w.grad is not None: # 把上一次的梯度清零
        w.grad.data.zero_()
    if b.grad is not None:
        b.grad.data.zero_()
    loss.backward() # 误差反向传播，得到 w 和 b 的梯度
    w.data = w.data - w.grad*learning_rate # 梯度下降找到新的 w 和 b
    b.data = b.data - b.grad*learning_rate
    if i % 50 == 0:
        print(w.grad, b.grad, loss.grad)

# %%
import torch
from torch import nn,optim

x = torch.rand([50,1])
y = 3*x+8

class Lr(nn.Module):
    def __init__(self):
        super(Lr, self).__init__()
        self.layer = nn.Linear(1,1)
    def forward(self, x):
        return self.layer(x)
model = Lr()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)
for i in range(500):
    out = model(x)
    loss = criterion(y, out)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
list(model.parameters())

# %%
