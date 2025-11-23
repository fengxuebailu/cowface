# 牛脸识别 (Cow Face Verification) 项目

基于 ResNet50 + ArcFace 的牛脸识别系统，用于区分不同个体的牛脸特征。

## 项目结构

```
.
├── cow_face_verification.py    # 主程序
├── requirements.txt             # 依赖列表
├── README.md                    # 本文件
└── data/                        # 数据目录（需要自己准备）
    ├── train/train/            # 训练数据（牛ID文件夹 -> 图片）
    ├── test-new/test-new/      # 测试图片
    ├── test-1118.csv           # 待预测的图片对列表
    └── sample-submission.csv   # 提交样例
```

## 环境要求

- **操作系统**: Ubuntu Linux
- **硬件**: 4x NVIDIA RTX 4090 (建议)
- **Python**: 3.8+

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行流程

### 1. 数据准备

确保数据目录结构如下：

```
./data/
├── train/train/
│   ├── KQ25009664/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── KQ25009665/
│   │   └── ...
│   └── ...
├── test-new/test-new/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
├── test-1118.csv
└── sample-submission.csv
```

### 2. 运行主程序

```bash
python cow_face_verification.py
```

程序会自动执行以下三个阶段：

#### 阶段 1：模型训练
- 从 `./data/train/train/` 加载带标签的训练数据
- 自动划分 90% 训练集 + 10% 验证集
- 使用 ResNet50 + ArcFace 进行训练，30 个 Epoch
- 保存最佳模型到 `./models/best_model.pth`

#### 阶段 2：阈值搜索
- 在验证集上计算所有样本对的余弦相似度
- 遍历阈值范围 [0.1, 0.95]，找到能使验证集准确率最高的阈值
- 保存最佳阈值到 `./models/best_threshold.npy`

#### 阶段 3：生成提交文件
- 读取 `./data/test-1118.csv` 中的待预测图片对
- 使用 TTA (Test Time Augmentation) 提取特征
- 计算每对图片的余弦相似度
- 根据最佳阈值生成预测结果
- 保存提交文件到 `./output/submission.csv`

## 核心算法说明

### 模型架构
- **骨干网络**: ResNet50 (ImageNet 预训练)
- **特征提取**: 512 维特征向量（L2 归一化）
- **分类头**: ArcFace (Margin=0.5, Scale=64)

### 损失函数
- **训练**: CrossEntropyLoss (基于牛的 ID 分类)
- **推理**: Cosine Similarity (余弦相似度)

### 数据增强
- **训练**: Resize(256) → RandomCrop(224) → RandomHorizontalFlip → ColorJitter → Normalization
- **测试**: Resize(224) → Normalization

### TTA (Test Time Augmentation)
- 对每张图片计算原图和水平翻转图的特征
- 两个特征向量进行 L2 归一化后相加
- 有助于提升推理的稳定性和准确率

## 输出文件

- `./models/best_model.pth`: 训练好的模型权重
- `./models/best_threshold.npy`: 最佳余弦相似度阈值
- `./output/submission.csv`: 最终提交文件

## 配置参数

在 `cow_face_verification.py` 中的 `Config` 类可以修改以下参数：

```python
BATCH_SIZE = 128              # Batch 大小
NUM_EPOCHS = 30               # 训练轮数
LEARNING_RATE = 0.01          # 初始学习率
VAL_SPLIT = 0.1               # 验证集比例
INPUT_SIZE = 224              # 输入图像大小
ARCFACE_MARGIN = 0.5          # ArcFace 角度间隔
ARCFACE_SCALE = 64            # ArcFace 缩放因子
USE_AMP = True                # 是否使用混合精度训练
```

## 性能指标

- 目标准确率: 0.99+
- 预期验证集准确率: 0.95+
- 推理速度: 受硬件限制，单对约 50-100ms (含 TTA)

## 故障排除

### 显存不足
- 减少 `BATCH_SIZE`
- 降低 `INPUT_SIZE`
- 关闭混合精度 (`USE_AMP = False`)

### 模型不收敛
- 检查数据是否正确加载
- 增加 `NUM_EPOCHS`
- 调整 `LEARNING_RATE`

### 文件找不到
- 检查数据目录结构是否正确
- 确保路径使用了 `pathlib.Path` (自动处理路径分隔符)

## 参考文献

- [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- [ResNet: Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

## 许可证

MIT License
