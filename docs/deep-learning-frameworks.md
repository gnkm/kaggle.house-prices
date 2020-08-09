# Deep Learning Frameworks

[google trends](https://trends.google.co.jp/trends/explore?date=today 3-m&geo=JP&q=%2Fg%2F11bwp1s2k3,%2Fg%2F11gd3905v1,keras,%2Fg%2F11g6ym8nbt,%2Fg%2F11g9fn390t)

- Pytorch
- TensorFlow

が多いか．

とりあえずこの 2 つを比較する．

[まとめて解説！機械学習・深層学習で使われるフレームワーク7選](https://ainow.ai/2020/08/07/224868/)

## PyTorch

[PyTorch](https://pytorch.org/)

- デプロイが容易？
- 並列処理を行うので速い
- 画像処理や自然言語処理のためのライブラリが豊富
- クラウド環境で使いやすい
- 動的な計算グラフを生成できる？

### サンプルコード

- [pytorch超入門 - Qiita](https://qiita.com/miyamotok0105/items/1fd1d5c3532b174720cd)

```
import torch
import torch.nn as nn
# import ...

# データを取得する
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

class Net(nn.Module):
    """モデルを定義する．

    1. 勾配を初期化する
    2. 順伝播する
    3. lossを計算する
    4. 逆誤差伝播する
    5. パラメータを更新する
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)

# 以下のような方法も可能
net = nn.Sequential()
net.add_module('Conv1',nn.Conv2d(1, 6, 3))
net.add_moduke('Conv2', nn.Conv2d(6, 16, 3))
net.add_module('fc1', nn.Linear(16 * 6 * 6, 120))
net.add_module('fc2', nn.Linear(120, 84))
net.add_module('fc3', nn.Linear(84, 10))

# モデルを生成する
model = Net()
if args.cuda:
    model.cuda()

# 最適化関数を設定する
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# 学習の方法を定義する
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

# 学習の実行
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)

# 予測
y_pred = model(test_loader)
```

その他

- [PyTorch入門 - ＠IT](https://www.atmarkit.co.jp/ait/series/17748/)
- [PyTorch入門  [公式Tutorial:DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZを読む] - Qiita](https://qiita.com/north_redwing/items/30f9619f0ee727875250)
- [PyTorch超入門！ - Smile Engineering Blog](https://smile-jsp.hateblo.jp/entry/2019/12/15/224408)

### 雑感

- deep learning についての本質的な理解が必要
- Chainer から fork されたもので Chainer と似ているということだが，似ていると感じられない
- 内部を細かくチューニングできる

## TensorFlow

[TensorFlow](https://www.tensorflow.org/)

- Keras を使うことでモデル構築が容易
- デプロイが容易: デプロイ先のプラットフォームを選ばない
- 柔軟性が高い: モデルをカスタマイズできる

### サンプルコード

写真を分類するタスク．
`fashion_mnist` データを用いる．

```
import tensorflow as tf
from tensorflow import keras
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# test_images.shape # => (60000, 28, 28) 28 x 28 ピクセルの画像が 60,000 枚含まれている

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
predictions = model.predict(test_images)
```

### 雑感

- Scikit-learn と同じような API でわかりやすい
- 直感的にわかりやすい


## Keras
## Caffe
## Microsoft Cognitive Toolkit
## MxNet
## Chainer

- [Chainer: A flexible framework for neural networks](https://chainer.org/)
- [Chainer – A flexible framework of neural networks — Chainer 7.7.0 documentation](https://docs.chainer.org/en/stable/)

- Powerful: GPU を使える(他のフレームワークも使える)
- flexible: ネットワークを柔軟に定義できる
- intuitive: deep learning の各要素を定義する．またデバッグしやすい．

### サンプルコード

省略．

### 雑感

予測を行うために，下記構造を持つ `Trainer` を準備する．

- Trainer
  - Updator
    - Iterator
      - Dataset
    - Optimizer
      - Model
  - Extensions

```
trainer.run()
```

で学習を行う(らしい)のだが，予測は

```
model.predictor()
```

で行う．
