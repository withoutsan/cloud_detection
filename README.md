## 关于文件的简要说明

datasets文件夹存放的是处理数据用的代码

networks存放的是模型的实现代码，包括了十个对比模型（含自己的方法CDCTFM）以及消融实验的模型

utils存放了一些tools，如数据装载和dice损失函数代码，都是别人造好的轮子直接拿来用的

dark.py是生成图像暗通道的代码

test.py是别人的代码，我没用到

test1.py是数据集38-Cloud的测试代码

test2.py是数据集MODIS的测试代码

train1.py是38-Cloud的训练代码

train2.py是MODIS的训练代码

util.py也是别人的轮子

注：38-Cloud的测试代码test1.py只负责生成mask，不给出评价指标（如F1-socre等），计算评价指标的代码使用的是官方给出的matlab代码，具体可见[38-Cloud](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset)