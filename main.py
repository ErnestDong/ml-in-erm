#%%
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch import nn

# df = pd.read_csv("./corporate_rating.csv", encoding="utf-8")
df = pd.read_csv("/Users/dcy/Code/erm/corporate_rating.csv", encoding="utf-8")
df.info()
#%%

sns.set(
    style="white",
    context="paper",
    rc={"text.usetex": True},
)
df["Rating"].value_counts().plot(kind="bar")
#%%

RANDOM_STATE = 42
Y = df["Rating"]
Y = Y.replace({"CCC": "C", "CC": "C"})
df["Date"] = df["Date"].apply(lambda x: x.split("/")[-1])
dummies = ["Rating Agency Name", "Sector", "Date"]
X = df[[i for i in df.columns if df[i].dtype != "object"]]
for dummy in dummies:
    X = pd.concat([X, pd.get_dummies(df[dummy], drop_first=True, prefix=dummy)], axis=1)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    X, Y, test_size=0.25, random_state=RANDOM_STATE
)
result = {}
X.columns
#%%

def get_score(Xtest, Ytrue, model):
    Ypred = model(Xtest)
    average = "weighted"
    rating_map = {i: ord(i[0]) * 100 - len(i) for i in Y.unique()}
    return {
        "precision": precision_score(Ytrue, Ypred, average=average, zero_division=0),
        "recall": recall_score(Ytrue, Ypred, average=average),
        "f1": f1_score(Ytrue, Ypred, average=average),
        "\(R^2\)": pearsonr(
            [rating_map[i] for i in Ypred], [rating_map[i] for i in Ytest]
        )[0],
    }


random.seed(RANDOM_STATE)
ratings = Y.unique()
tmp = {}
monte_num = 100
for i in range(100):
    Ypredict = Xtest.index.map(lambda x: random.choice(ratings))
    monte = get_score(Xtest, Ytest, lambda _: Ypredict)
    for j in monte:
        if j not in tmp:
            tmp[j] = 0
        tmp[j] += monte[j]
result["random"] = {i: tmp[i] / 100 for i in tmp}
result["random"]
#%%


logit = LogisticRegression(
    multi_class="multinomial", solver="saga", random_state=RANDOM_STATE
)
logit.fit(Xtrain, Ytrain)
result["logit"] = get_score(Xtest, Ytest, logit.predict)
result["logit"]

#%%

dt = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
dt.fit(Xtrain, Ytrain)
result["decision tree"] = get_score(Xtest, Ytest, dt.predict)
result["decision tree"]
#%%


rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=RANDOM_STATE)
rf.fit(Xtrain, Ytrain)
result["random forest"] = get_score(Xtest, Ytest, rf.predict)
result["random forest"]

#%%

gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
gb.fit(Xtrain, Ytrain)
result["gradient boosting"] = get_score(Xtest, Ytest, gb.predict)
result["gradient boosting"]
#%%


svm = SVC(kernel="rbf", gamma="auto", random_state=RANDOM_STATE)
svm.fit(Xtrain, Ytrain)
result["svm"] = get_score(Xtest, Ytest, svm.predict)
result["svm"]

#%%

KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(Xtrain, Ytrain)
result["KNN"] = get_score(Xtest, Ytest, KNN.predict)
result["KNN"]
#%%

torch.manual_seed(42)
x = torch.rand([500, 1])  # X 是一个 tensor ，可以把他想象成 500x1 的向量
y_true = 3 * x + 8
learning_rate = 0.05  # learning rate 是每次梯度下降的“步长”
w = torch.rand([1, 1], requires_grad=True)  # w 和 b 我们要 pytorch 自动求导
b = torch.tensor(0, requires_grad=True, dtype=torch.float32)
for i in range(500):
    y_pred = torch.matmul(x, w) + b  # 预测是多少
    loss = (y_true - y_pred).pow(2).mean()  # 损失
    if w.grad is not None:  # 把上一次的梯度清零
        w.grad.data.zero_()
    if b.grad is not None:
        b.grad.data.zero_()
    loss.backward()  # 误差反向传播，得到 w 和 b 的梯度
    w.data = w.data - w.grad * learning_rate  # 梯度下降找到新的 w 和 b
    b.data = b.data - b.grad * learning_rate
    if i % 50 == 0:
        print(w.item(), b.item(), loss.item())
#%%


Ytrain_nn = pd.get_dummies(Ytrain)
encode = Ytrain_nn.columns
Ytrain_nn = torch.tensor(Ytrain_nn.values, dtype=torch.float32)
Xtrain_nn = torch.tensor(Xtrain.values, dtype=torch.float32)

hidden_layer = 40
net = nn.Sequential(
    nn.Linear(Xtrain_nn.shape[1], hidden_layer),
    nn.ReLU(),
    nn.Linear(hidden_layer, len(encode)),
    nn.Softmax(dim=1),
)
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()

for t in range(10000):
    prediction = net(Xtrain_nn)
    loss = loss_func(Ytrain_nn, prediction)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
Xtest_nn = torch.tensor(Xtest.values, dtype=torch.float32)
prediction = pd.DataFrame(net(Xtest_nn).detach().numpy())
Ypredict = prediction.idxmax(axis=1).map(lambda x: encode[x])
result["bp neural network"] = get_score(Xtest, Ytest, lambda _: Ypredict)
result["bp neural network"]
#%%

class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(Xtrain_nn.shape[1], 20, 3, padding=3),
            nn.Tanh(),
            nn.AvgPool1d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(40, len(encode)),
            nn.ReLU(),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


Xtrain_cnn = Xtrain_nn.unsqueeze(2)
Xtest_cnn = Xtest_nn.unsqueeze(2)
net = CNN()
optimizer = torch.optim.Adamax(net.parameters())
loss_func = torch.nn.L1Loss()
epochnum = 10000
for epoch in range(epochnum):
    prediction = net(Xtrain_cnn)
    loss = loss_func(Ytrain_nn, prediction)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % (epochnum / 10) == 0:
        print("epoch:", epoch, "loss:", loss.item())
prediction = pd.DataFrame(net(Xtest_cnn).detach().numpy())
Ypredict = prediction.idxmax(axis=1).map(lambda x: encode[x])
result["CNN"] = get_score(Xtest, Ytest, lambda _: Ypredict)
result["CNN"]

#%%

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(32 * 2, num_classes)

    def forward(self, x):
        # x, _ = x
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


input_size = 1
hidden_size = 32
num_layers = 1
num_classes = 7
net = LSTM()
optimizer = torch.optim.Adamax(net.parameters())
loss_func = nn.MSELoss()
epochnum = 3000
for epoch in range(epochnum):
    out = net(Xtrain_nn.unsqueeze(2))
    loss = loss_func(out, Ytrain_nn)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % (epochnum / 10) == 0:
        print("epoch:", epoch, "loss:", loss.item())

prediction = pd.DataFrame(net(Xtest_nn.unsqueeze(2)).detach().numpy())
Ypredict = prediction.idxmax(axis=1).map(lambda x: encode[x])
result["RNN"] = get_score(Xtest, Ytest, lambda _: Ypredict)
result["RNN"]
#%%

feature = ["precision", "recall", "f1", "\(R^2\)"]
[["model"] + feature] + list(
    [i[0]] + [round(j, 4) for j in i[1].values()] for i in result.items()
)

N = len(feature)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
angles = np.concatenate((angles, [angles[0]]))
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
for model in result:
    values = [i for i in result[model].values()] + [result[model]["precision"]]
    ax.plot(angles, values, label=model)
    ax.fill(angles, values, alpha=0.1)
ax.set_thetagrids(angles[:-1] * 180 / np.pi, feature)
ax.grid(True)
plt.legend(bbox_to_anchor=(1.2, -0.1), ncol=3)
plt.show()
