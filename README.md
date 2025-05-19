## 项目名称
Fine-tuning-Pretrained-CNN-for-Caltech101-Classification

## 项目描述
本项目使用在ImageNet上预训练的卷积神经网络(CNN)架构(如AlexNet或ResNet-18)，通过微调(fine-tuning)将其应用于Caltech-101数据集的分类任务。项目包含完整的训练流程、超参数实验以及与从零开始训练的对比实验。

## 项目结构
```
Fine-tuning-Pretrained-CNN-for-Caltech101-Classification/
├── data/                    # 数据集目录(需自行下载)
├── models/                  # 模型定义代码
├── utils/                   # 工具函数
├── train.py                 # 训练脚本
├── test.py                  # 测试脚本
├── requirements.txt         # 依赖库
└── README.md                # 项目说明文件
```

## 快速开始

### 1. 环境配置
```bash
git clone https://github.com/[your-username]/Fine-tuning-Pretrained-CNN-for-Caltech101-Classification.git
cd Fine-tuning-Pretrained-CNN-for-Caltech101-Classification
pip install -r requirements.txt
```

### 2. 数据集准备
1. 从[Caltech-101官网](https://data.caltech.edu/records/mzrjq-6wc02)下载数据集
2. 解压到项目目录下的`data/`文件夹
3. 数据集将自动按照标准划分训练集和测试集

### 3. 训练模型
```bash
python train.py \
    --model resnet18 \          # 选择模型架构(resnet18或alexnet)
    --epochs 50 \               # 训练轮数
    --batch_size 32 \           # 批次大小
    --lr 0.001 \                # 初始学习率
    --fine_tune True \          # 是否微调预训练模型
    --log_dir ./logs            # TensorBoard日志目录
```

### 4. 测试模型
```bash
python test.py \
    --model resnet18 \          # 模型架构
    --weights path/to/model.pth # 训练好的模型权重
    --batch_size 32             # 批次大小
```

### 5. 可视化训练过程
```bash
tensorboard --logdir ./logs
```

## 预训练模型下载
训练好的模型权重已上传至[Google Drive](https://drive.google.com/drive/folders/[your-folder-id])。

## 实验报告
实验报告PDF包含以下内容：
1. 模型架构和数据集介绍
2. 训练过程中的loss曲线和accuracy变化(TensorBoard截图)
3. 不同超参数组合的实验结果对比
4. 微调模型与从零训练模型的性能对比
5. 结论与分析

报告链接：[实验报告PDF](https://drive.google.com/file/d/[your-file-id]/view)

## 贡献
欢迎提交issue或pull request。对于重大更改，请先开issue讨论您想做的更改。

## 许可证
[MIT License](LICENSE)