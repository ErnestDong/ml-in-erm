## 安装方式：

pytorch 占的空间挺大，建议采用虚拟环境安装。虚拟环境会把 python 库安装在当前文件夹下，不会污染你安装的python

方式1.
cmd(windows)/terminal(mac) 中运行以下命令：
```
pip3 install poetry # 安装虚拟环境管理命令
poetry config virtualenvs.in-project true # 虚拟环境建立在 project 目录下
cd /path/to/erm # 转到本文件夹所在位置
poetry install # 安装依赖
```

方式2.
cmd(windows)/terminal(mac) 中运行以下命令：
```
cd /path/to/erm # 转到本文件夹所在位置
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple # 使用清华镜像加快下载速度
python3 -m venv .venv # 建立虚拟环境
source .venv/bin/activate #mac
\Scripts\activate.bat # windows
pip install -r requirements.txt
```

## 运行方式：

方式1：打开 notebook 逐块运行
方式2：打开 main.py 按 code cell 逐个运行(每个“#%%”隔开的，vscode中会显示“运行单元”)
方式3：kaggle 注册后 https://www.kaggle.com/code/ernestdong/notebook0e25bfe343，记得选取使用 GPU 加速
