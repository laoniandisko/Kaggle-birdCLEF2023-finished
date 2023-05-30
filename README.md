# Kaggle-birdCLEF2023-finished
kaggle竞赛的birdclef2023已完结，小记首次参加kaggle，成绩为9%，bronze medal

数据集可以从官网下载，但是由于我对数据集做了一些修改这里附上数据集链接
百度网盘的文件大小限制，本穷逼只能将数据集拆成两份，解压后将train_audio.rar中的文件放到kaggle-input（对应根目录下kaggle中的input）中的对应位置，kaggle-input在本项目的kaggle文件夹下。

train_audio.rar
链接：https://pan.baidu.com/s/1Gt-ybpkdFLpYviv_OZKAgw?pwd=1111

kaggle-input.rar
链接：https://pan.baidu.com/s/1lY5g_SAVM97YbrUyg5EJPQ?pwd=1111 

噪声数据(对应根目录下input)：
input.rar
链接：https://pan.baidu.com/s/1DAh7KveExq5iMLfALw8OuA?pwd=1111 

训练好的模型：
upsample40.rar
链接：https://pan.baidu.com/s/1mSGM9rMtznuCAF-JiidvEw?pwd=1111 

2060还是不行，做迁移训练比较困难，前面的预训练有六十多G的音频，预训练就直接用的去年birdCLEF2022的模型的参数作为预训练模型，将卷积层全部冻结，训练线性层和attention层（万不得已，由于这一层也有鸟类种类），
勉强拿了9%，如果时间和算力足够的情况下我也会试一试不冻结卷积层，当然预期效果估计不行，
就这样，明年比赛加油




