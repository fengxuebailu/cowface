"""
牛脸识别 (Cow Face Verification) 项目
基于 ResNet50 + ArcFace，用于识别不同牛的脸部特征

运行环境: Ubuntu Linux with 4x NVIDIA RTX 4090
"""

import os
import sys
import csv
import warnings
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image

warnings.filterwarnings('ignore')

# ============================================================================
# 配置参数
# ============================================================================

class Config:
    """项目配置"""
    # 路径配置
    DATA_ROOT = Path('./.claude/data')
    TRAIN_DIR = DATA_ROOT / 'train' / 'train'
    TEST_DIR = DATA_ROOT / 'test-new' / 'test-new'
    TEST_CSV = DATA_ROOT / 'test-1118.csv'
    SAMPLE_SUBMISSION = DATA_ROOT / 'sample-submission.csv'

    # 模型保存路径
    MODEL_SAVE_PATH = Path('./models')
    CHECKPOINT_PATH = MODEL_SAVE_PATH / 'best_model.pth'
    BEST_THRESHOLD_PATH = MODEL_SAVE_PATH / 'best_threshold.npy'

    # 输出路径
    OUTPUT_DIR = Path('./output')
    SUBMISSION_PATH = OUTPUT_DIR / 'submission.csv'

    # 训练参数
    BATCH_SIZE = 128  # 使用充足的显存
    NUM_EPOCHS = 30   # 训练轮数
    LEARNING_RATE = 0.01  # 学习率
    WEIGHT_DECAY = 5e-4
    VAL_SPLIT = 0.1  # 验证集比例

    # K-Fold 参数
    USE_KFOLD = True  # 是否使用 K-Fold 交叉验证
    NUM_FOLDS = 5     # K-Fold 数量

    # 模型参数
    INPUT_SIZE = 320  # 从 224 增大到 320，保留更多细节（牛脸纹理）
    NUM_CLASSES = None  # 将在数据加载时动态设置

    # ArcFace 参数
    ARCFACE_MARGIN = 0.5
    ARCFACE_SCALE = 64

    # 硬件参数
    USE_AMP = True  # 混合精度训练
    NUM_WORKERS = 4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    USE_MULTI_GPU = torch.cuda.device_count() > 1

    # 阈值搜索参数：从 -1 到 1（余弦相似度范围）
    THRESHOLD_RANGE = np.arange(-1.0, 1.0, 0.02)

    # 随机种子
    SEED = 42


# ============================================================================
# 数据集定义
# ============================================================================

class CowTrainDataset(Dataset):
    """牛脸识别训练数据集

    自动解析双层目录结构: ./data/train/train/[牛ID]/[图片.jpg]
    """

    def __init__(self,
                 root_dir: Path,
                 transform=None,
                 indices: List[int] = None):
        """
        Args:
            root_dir: 训练数据根目录 (./data/train/train/)
            transform: 数据增强变换
            indices: 用于划分训练/验证集的索引列表
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        # 读取所有牛ID和对应的图片
        self.samples = []  # 列表：(图片路径, 标签ID, 牛ID字符串)
        self.id_to_label = {}  # 字典：牛ID -> 数字标签

        label_counter = 0
        for cow_id_dir in sorted(self.root_dir.iterdir()):
            if not cow_id_dir.is_dir():
                continue

            cow_id = cow_id_dir.name
            self.id_to_label[cow_id] = label_counter

            # 获取该牛ID下的所有图片
            for img_file in sorted(cow_id_dir.glob('*.jpg')):
                self.samples.append((img_file, label_counter, cow_id))

            label_counter += 1

        # 如果指定了索引，则只保留对应的样本
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

        print(f"[CowTrainDataset] 加载了 {len(self.samples)} 张图片, 共 {label_counter} 个牛ID")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, cow_id = self.samples[idx]

        # 读取图片
        img = Image.open(img_path).convert('RGB')

        # 应用数据增强
        if self.transform:
            img = self.transform(img)

        return {
            'image': img,
            'label': torch.tensor(label, dtype=torch.long),
            'cow_id': cow_id,
            'path': str(img_path)
        }


class CowTestDataset(Dataset):
    """牛脸识别测试数据集

    从 test-new/test-new/ 目录加载图片，文件名为 [ID].jpg
    """

    def __init__(self,
                 root_dir: Path,
                 image_ids: List[str],
                 transform=None):
        """
        Args:
            root_dir: 测试图片目录 (./data/test-new/test-new/)
            image_ids: 要加载的图片ID列表 (例如 ['0001', '0002', ...])
            transform: 数据增强变换
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_ids = image_ids

        # 验证所有图片存在
        missing_files = []
        for img_id in self.image_ids:
            img_path = self.root_dir / f'{img_id}.jpg'
            if not img_path.exists():
                missing_files.append(img_id)

        if missing_files:
            print(f"[警告] 以下图片不存在: {missing_files}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = self.root_dir / f'{img_id}.jpg'

        # 读取图片
        img = Image.open(img_path).convert('RGB')

        # 应用数据增强
        if self.transform:
            img = self.transform(img)

        return {
            'image': img,
            'image_id': img_id,
            'path': str(img_path)
        }


# ============================================================================
# ArcFace 模块
# ============================================================================

class ArcFace(nn.Module):
    """ArcFace 损失函数模块

    ArcFace 是一种角度化的人脸识别损失函数，通过在角度空间中增加边际来提高类间差异性。
    参考: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    """

    def __init__(self,
                 feature_dim: int = 512,
                 num_classes: int = None,
                 margin: float = 0.5,
                 scale: float = 64):
        """
        Args:
            feature_dim: 特征向量维度
            num_classes: 分类数（牛的数量）
            margin: 角度间隔 (弧度)
            scale: 缩放因子
        """
        super(ArcFace, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # 权重矩阵，每行代表一个类别的特征原型
        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, feature_dim).uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)
        )

    def forward(self, feature: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature: 特征向量，形状 (batch_size, feature_dim)，需要已归一化
            label: 类别标签，形状 (batch_size,)

        Returns:
            logits: 缩放后的角度 logits，形状 (batch_size, num_classes)
        """
        batch_size = feature.size(0)

        # 对权重进行归一化
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # 计算余弦相似度：(batch_size, num_classes)
        cos_theta = F.linear(feature, weight_norm)
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

        # 计算角度：(batch_size, num_classes)
        theta = torch.acos(cos_theta)

        # 对正样本（目标类）添加边际：使用 mask 向量化操作
        # 创建 label 的 one-hot 编码
        one_hot = F.one_hot(label, num_classes=self.num_classes).float()

        # 计算添加边际后的角度（仅对正样本添加）
        theta_m = theta + one_hot * self.margin

        # 计算输出
        cos_theta_m = torch.cos(theta_m)

        # 缩放
        output = cos_theta_m * self.scale

        return output


class CowFaceModel(nn.Module):
    """牛脸识别模型：ResNet50 + ArcFace"""

    def __init__(self,
                 num_classes: int,
                 feature_dim: int = 512,
                 pretrained: bool = True):
        """
        Args:
            num_classes: 牛的总数（分类数）
            feature_dim: 特征向量维度
            pretrained: 是否使用 ImageNet 预训练权重
        """
        super(CowFaceModel, self).__init__()

        # ResNet50 骨干网络
        self.backbone = resnet50(pretrained=pretrained)

        # 删除分类层，只保留特征提取器
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # 特征归一化层
        self.feature_fc = nn.Linear(in_features, feature_dim)
        self.feature_bn = nn.BatchNorm1d(feature_dim)

        # ArcFace 头
        self.arcface = ArcFace(
            feature_dim=feature_dim,
            num_classes=num_classes,
            margin=Config.ARCFACE_MARGIN,
            scale=Config.ARCFACE_SCALE
        )

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取归一化的特征向量

        Args:
            x: 输入图片，形状 (batch_size, 3, H, W)

        Returns:
            features: 归一化特征向量，形状 (batch_size, feature_dim)
        """
        # 通过骨干网络
        x = self.backbone(x)

        # 映射到特征空间
        features = self.feature_fc(x)
        features = self.feature_bn(features)

        # 归一化
        features = F.normalize(features, p=2, dim=1)

        return features

    def forward(self, x: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入图片，形状 (batch_size, 3, H, W)
            label: 类别标签（训练时需要），形状 (batch_size,)

        Returns:
            output: 如果提供了 label，返回 ArcFace logits；否则返回特征向量
        """
        features = self.get_features(x)

        if label is not None:
            # 训练模式：返回 ArcFace logits
            return self.arcface(features, label)
        else:
            # 推理模式：返回特征向量
            return features


# ============================================================================
# 数据增强
# ============================================================================

def get_train_transforms(input_size: int = 320) -> transforms.Compose:
    """训练集数据增强 - 强化版

    包含：
    - Resize + RandomResizedCrop（保持宽高比的随机裁剪）
    - RandomRotation（几何变换）
    - RandomHorizontalFlip（水平翻转）
    - ColorJitter（颜色抖动，强化版）
    - RandomAffine（随机仿射变换）
    - GaussianBlur（高斯模糊）
    - Normalization
    """
    return transforms.Compose([
        transforms.Resize(int(input_size * 1.15)),  # 320 -> 368
        transforms.RandomRotation(degrees=15),  # 新增：旋转 [-15, 15] 度
        transforms.RandomResizedCrop(
            input_size,
            scale=(0.85, 1.0),  # 裁剪比例
            ratio=(0.9, 1.1)    # 宽高比范围
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.3,   # 增强从 0.2 -> 0.3
            contrast=0.3,     # 增强从 0.2 -> 0.3
            saturation=0.3,   # 增强从 0.2 -> 0.3
            hue=0.1           # 新增：色调调整
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),  # 新增：随机平移 ±10%
            scale=(0.95, 1.05)     # 新增：随机缩放
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.2)),  # 新增：高斯模糊
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_test_transforms(input_size: int = 320) -> transforms.Compose:
    """测试集数据增强

    只包含：resize -> normalization
    """
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ============================================================================
# 训练相关函数
# ============================================================================

def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                scaler: GradScaler,
                device: str) -> float:
    """训练一个 epoch

    Args:
        model: 模型
        dataloader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        scaler: 混合精度梯度缩放器
        device: 计算设备

    Returns:
        avg_loss: 平均损失
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc='训练中', leave=False)
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # 混合精度训练
        with autocast(enabled=Config.USE_AMP):
            logits = model(images, labels)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(model: nn.Module,
             dataloader: DataLoader,
             device: str) -> float:
    """验证模型

    Args:
        model: 模型
        dataloader: 验证数据加载器
        device: 计算设备

    Returns:
        avg_loss: 平均损失
    """
    model.eval()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc='Validating', leave=False)
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        logits = model(images, labels)
        loss = F.cross_entropy(logits, labels)

        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train_model(train_loader: DataLoader,
                val_loader: DataLoader,
                model: nn.Module,
                num_epochs: int = 30,
                device: str = 'cuda'):
    """完整的模型训练流程

    Args:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        model: 模型
        num_epochs: 训练轮数
        device: 计算设备
    """
    print(f"\n[训练阶段] 开始训练...")

    # 多卡并行
    if Config.USE_MULTI_GPU:
        model = nn.DataParallel(model)

    model = model.to(device)

    # 优化器：使用更激进的学习率
    optimizer = SGD(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        momentum=0.9,
        weight_decay=Config.WEIGHT_DECAY
    )

    # 学习率调度器：余弦退火
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 混合精度训练
    scaler = GradScaler(enabled=Config.USE_AMP)

    # 训练循环
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\n[Epoch {epoch+1}/{num_epochs}]")

        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        print(f"  平均损失: {train_loss:.4f}")

        # 验证
        val_loss = validate(model, val_loader, device)
        print(f"  验证损失: {val_loss:.4f}")

        # 学习率调度
        scheduler.step()
        print(f"  学习率:   {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  ✓ 保存最佳模型")

            # 创建模型保存目录
            Config.MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

            # 保存模型（处理 DataParallel）
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), Config.CHECKPOINT_PATH)

    print(f"\n[训练完成] 最佳模型已保存: {Config.CHECKPOINT_PATH}")
    return model


# ============================================================================
# 特征提取
# ============================================================================

@torch.no_grad()
def extract_features(model: nn.Module,
                     dataloader: DataLoader,
                     device: str) -> Tuple[np.ndarray, List[str]]:
    """从数据集中提取特征向量

    Args:
        model: 模型（可能已经 eval() 了）
        dataloader: 数据加载器
        device: 计算设备

    Returns:
        features: 特征矩阵，形状 (N, feature_dim)
        identifiers: 样本标识符列表（图片ID或牛ID）
    """
    model.eval()

    all_features = []
    all_identifiers = []

    pbar = tqdm(dataloader, desc='提取特征', leave=False)
    for batch in pbar:
        images = batch['image'].to(device)

        # 提取特征
        # 处理 DataParallel 的情况
        if hasattr(model, 'module'):
            features = model.module.get_features(images)
        else:
            features = model.get_features(images)

        all_features.append(features.cpu().numpy())

        # 保存标识符（cow_id 或 image_id）
        if 'cow_id' in batch:
            all_identifiers.extend(batch['cow_id'])
        else:
            all_identifiers.extend(batch['image_id'])

    # 拼接所有特征
    features = np.vstack(all_features)

    return features, all_identifiers


# ============================================================================
# 阈值搜索
# ============================================================================

def compute_pair_similarity(features: np.ndarray,
                           idx1: int,
                           idx2: int) -> float:
    """计算两个特征向量的余弦相似度

    Args:
        features: 特征矩阵，形状 (N, feature_dim)，已归一化
        idx1: 第一个样本的索引
        idx2: 第二个样本的索引

    Returns:
        similarity: 余弦相似度，范围 [0, 1]
    """
    feat1 = features[idx1]
    feat2 = features[idx2]

    # 余弦相似度 = 点积（因为特征已归一化）
    similarity = np.dot(feat1, feat2)

    return similarity


def search_best_threshold(val_dataset: CowTrainDataset,
                         val_features: np.ndarray,
                         threshold_range: np.ndarray = None) -> Tuple[float, float]:
    """在验证集上搜索最佳余弦相似度阈值

    Args:
        val_dataset: 验证数据集
        val_features: 验证集特征矩阵，形状 (N, feature_dim)
        threshold_range: 阈值搜索范围

    Returns:
        best_threshold: 最佳阈值
        best_accuracy: 最佳准确率
    """
    print(f"\n[阈值搜索阶段] 开始搜索最佳阈值...")

    if threshold_range is None:
        threshold_range = Config.THRESHOLD_RANGE

    # 构造验证集中的正负样本对
    cow_ids = [item[2] for item in val_dataset.samples]  # 获取牛ID

    # 生成正样本对（相同牛ID）
    positive_pairs = []
    for i in range(len(cow_ids)):
        for j in range(i + 1, len(cow_ids)):
            if cow_ids[i] == cow_ids[j]:
                positive_pairs.append((i, j, 1))

    # 生成负样本对（不同牛ID）
    negative_pairs = []
    for i in range(len(cow_ids)):
        for j in range(i + 1, len(cow_ids)):
            if cow_ids[i] != cow_ids[j]:
                negative_pairs.append((i, j, 0))

    # 限制负样本对数量，以提高计算效率（至少保留和正样本一样多的负样本）
    if len(negative_pairs) > len(positive_pairs) * 10:
        indices = np.random.choice(len(negative_pairs),
                                   size=min(len(negative_pairs), len(positive_pairs) * 10),
                                   replace=False)
        negative_pairs = [negative_pairs[i] for i in indices]

    all_pairs = positive_pairs + negative_pairs
    print(f"  正样本对: {len(positive_pairs)}, 负样本对: {len(negative_pairs)}")

    # 计算所有对的相似度和真实标签
    similarities = []
    labels = []

    print(f"  计算样本对的相似度...")
    for idx1, idx2, label in tqdm(all_pairs, leave=False):
        similarity = compute_pair_similarity(val_features, idx1, idx2)
        similarities.append(similarity)
        labels.append(label)

    similarities = np.array(similarities)
    labels = np.array(labels)

    # 打印相似度分布，帮助诊断
    pos_sims = similarities[labels == 1]
    neg_sims = similarities[labels == 0]
    print(f"  正样本相似度: min={pos_sims.min():.4f}, max={pos_sims.max():.4f}, mean={pos_sims.mean():.4f}, std={pos_sims.std():.4f}")
    print(f"  负样本相似度: min={neg_sims.min():.4f}, max={neg_sims.max():.4f}, mean={neg_sims.mean():.4f}, std={neg_sims.std():.4f}")

    # 遍历阈值范围，找到最佳阈值
    print(f"  遍历阈值范围...")
    best_threshold = 0.5
    best_accuracy = 0.0

    for threshold in tqdm(threshold_range, leave=False):
        predictions = (similarities > threshold).astype(int)
        accuracy = np.mean(predictions == labels)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    print(f"  最佳阈值: {best_threshold:.3f}")
    print(f"  最佳准确率: {best_accuracy:.4f}")
    print(f"  阈值处的正样本判对率: {np.mean((similarities[labels==1] > best_threshold).astype(int)):.4f}")
    print(f"  阈值处的负样本判对率: {np.mean((similarities[labels==0] <= best_threshold).astype(int)):.4f}")

    # 如果准确率太低，使用中位数阈值作为备选
    if best_accuracy < 0.6:
        print(f"\n  ⚠️  警告：阈值搜索准确率较低（{best_accuracy:.4f}），可能存在特征分布问题")
        print(f"      尝试使用中位数阈值...")
        median_threshold = np.median(similarities)
        median_accuracy = np.mean((similarities > median_threshold) == labels)
        print(f"      中位数阈值: {median_threshold:.3f}, 准确率: {median_accuracy:.4f}")

        if median_accuracy > best_accuracy:
            best_threshold = median_threshold
            best_accuracy = median_accuracy
            print(f"  ✓ 使用中位数阈值")

    # 保存最佳阈值
    Config.MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    np.save(Config.BEST_THRESHOLD_PATH, best_threshold)

    return best_threshold, best_accuracy


# ============================================================================
# 推理和提交
# ============================================================================

def load_test_csv(csv_path: Path) -> List[str]:
    """读取测试 CSV 文件

    Args:
        csv_path: CSV 文件路径

    Returns:
        pairs: 待预测的图片对列表，例如 ['0001_0002', '0003_0004', ...]
    """
    print(f"\n[推理阶段] 读取测试数据: {csv_path}")

    pairs = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)

        # 检查是否有表头
        first_row = next(reader, None)
        if first_row is None:
            return []

        # 判断第一行是否是表头
        try:
            # 如果能解析为数字，说明不是表头
            int(first_row[0].split('_')[0])
            # 重新处理第一行
            pairs.append(first_row[0])
        except:
            # 是表头，继续读取
            pass

        # 读取后续行
        for row in reader:
            if len(row) > 0:
                pairs.append(row[0])

    print(f"  加载了 {len(pairs)} 个待预测对")
    return pairs


def generate_submission(model: nn.Module,
                       test_dir: Path,
                       test_pairs: List[str],
                       best_threshold: float,
                       device: str,
                       use_tta: bool = True) -> np.ndarray:
    """生成提交文件

    Args:
        model: 训练好的模型（可能已经 eval() 了）
        test_dir: 测试图片目录
        test_pairs: 待预测的图片对列表
        best_threshold: 最佳余弦相似度阈值
        device: 计算设备
        use_tta: 是否使用 TTA (Test Time Augmentation)

    Returns:
        predictions: 预测结果数组
    """
    print(f"\n[推理阶段] 开始生成预测...")

    model.eval()

    # 特征缓存，避免重复计算
    feature_cache = {}

    test_transform = get_test_transforms(Config.INPUT_SIZE)

    predictions = []
    all_similarities = []  # 调试：记录所有相似度

    pbar = tqdm(test_pairs, desc='生成预测')
    for pair_str in pbar:
        # 解析配对
        img_id_1, img_id_2 = pair_str.split('_')

        # 加载特征（从缓存或计算）
        if img_id_1 not in feature_cache:
            img_path_1 = test_dir / f'{img_id_1}.jpg'
            img_1 = Image.open(img_path_1).convert('RGB')
            img_1 = test_transform(img_1).unsqueeze(0).to(device)

            with torch.no_grad():
                # 处理 DataParallel
                if hasattr(model, 'module'):
                    feat_1 = model.module.get_features(img_1).cpu().numpy()
                else:
                    feat_1 = model.get_features(img_1).cpu().numpy()

            # 如果使用 TTA，还要提取水平翻转版本
            if use_tta:
                img_1_flipped = transforms.functional.hflip(img_1)
                with torch.no_grad():
                    if hasattr(model, 'module'):
                        feat_1_flipped = model.module.get_features(img_1_flipped).cpu().numpy()
                    else:
                        feat_1_flipped = model.get_features(img_1_flipped).cpu().numpy()
                # 归一化后相加
                feat_1 = (feat_1 + feat_1_flipped) / 2
                feat_1 = feat_1 / (np.linalg.norm(feat_1) + 1e-8)

            feature_cache[img_id_1] = feat_1

        if img_id_2 not in feature_cache:
            img_path_2 = test_dir / f'{img_id_2}.jpg'
            img_2 = Image.open(img_path_2).convert('RGB')
            img_2 = test_transform(img_2).unsqueeze(0).to(device)

            with torch.no_grad():
                if hasattr(model, 'module'):
                    feat_2 = model.module.get_features(img_2).cpu().numpy()
                else:
                    feat_2 = model.get_features(img_2).cpu().numpy()

            # 如果使用 TTA
            if use_tta:
                img_2_flipped = transforms.functional.hflip(img_2)
                with torch.no_grad():
                    if hasattr(model, 'module'):
                        feat_2_flipped = model.module.get_features(img_2_flipped).cpu().numpy()
                    else:
                        feat_2_flipped = model.get_features(img_2_flipped).cpu().numpy()
                feat_2 = (feat_2 + feat_2_flipped) / 2
                feat_2 = feat_2 / (np.linalg.norm(feat_2) + 1e-8)

            feature_cache[img_id_2] = feat_2

        # 计算相似度
        feat_1 = feature_cache[img_id_1].flatten()
        feat_2 = feature_cache[img_id_2].flatten()
        similarity = np.dot(feat_1, feat_2)

        # 调试：记录相似度
        all_similarities.append(similarity)

        # 判断
        prediction = 1 if similarity > best_threshold else 0
        predictions.append(prediction)

    # 调试：打印推理集的相似度分布
    all_similarities = np.array(all_similarities)
    print(f"\n  推理集相似度分布:")
    print(f"    Min: {all_similarities.min():.4f}, Max: {all_similarities.max():.4f}")
    print(f"    Mean: {all_similarities.mean():.4f}, Std: {all_similarities.std():.4f}")
    print(f"    Median: {np.median(all_similarities):.4f}")
    print(f"    预测为1的比例: {np.mean(predictions):.4f}")

    return np.array(predictions)


def save_submission(predictions: np.ndarray,
                   test_pairs: List[str],
                   output_path: Path):
    """保存提交文件

    Args:
        predictions: 预测结果数组
        test_pairs: 待预测的图片对列表
        output_path: 输出文件路径
    """
    print(f"\n[推理阶段] 保存提交文件: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID_ID', 'prediction'])  # 表头

        for pair, pred in zip(test_pairs, predictions):
            writer.writerow([pair, pred])

    print(f"  提交文件已保存，共 {len(predictions)} 个预测")


# ============================================================================
# K-Fold 交叉验证
# ============================================================================

def train_with_kfold(train_dataset: CowTrainDataset,
                     num_folds: int = 5) -> Tuple[List[nn.Module], List[float], float]:
    """使用 K-Fold 交叉验证训练多个模型

    Args:
        train_dataset: 训练数据集
        num_folds: 折数

    Returns:
        models: K 个训练好的模型列表
        best_thresholds: 每个 Fold 的最佳阈值
        ensemble_accuracy: 集成模型的最终准确率
    """
    print(f"\n[K-Fold 交叉验证] 使用 {num_folds}-Fold 训练...")

    num_samples = len(train_dataset)
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=Config.SEED)

    models = []
    best_thresholds = []
    all_val_accuracies = []

    # 获取所有样本的索引
    all_indices = np.arange(num_samples)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_indices)):
        print(f"\n[Fold {fold + 1}/{num_folds}]")
        print(f"  训练样本: {len(train_idx)}, 验证样本: {len(val_idx)}")

        # 创建当前 Fold 的训练和验证数据集
        train_fold_dataset = CowTrainDataset(
            root_dir=Config.TRAIN_DIR,
            transform=get_train_transforms(Config.INPUT_SIZE),
            indices=train_idx.tolist()
        )

        val_fold_dataset = CowTrainDataset(
            root_dir=Config.TRAIN_DIR,
            transform=get_test_transforms(Config.INPUT_SIZE),
            indices=val_idx.tolist()
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_fold_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_fold_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )

        # 创建新模型
        model = CowFaceModel(
            num_classes=Config.NUM_CLASSES,
            feature_dim=512,
            pretrained=True
        )

        # 训练模型
        model = train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            num_epochs=Config.NUM_EPOCHS,
            device=Config.DEVICE
        )

        # 保存模型
        model_path = Config.MODEL_SAVE_PATH / f'fold_{fold + 1}_model.pth'
        Config.MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), model_path)
        print(f"  Fold {fold + 1} 模型已保存: {model_path}")

        # 在验证集上提取特征并搜索阈值
        model.eval()
        val_features, _ = extract_features(model, val_loader, Config.DEVICE)
        val_features = val_features / (np.linalg.norm(val_features, axis=1, keepdims=True) + 1e-8)

        best_threshold, best_accuracy = search_best_threshold(val_fold_dataset, val_features)
        best_thresholds.append(best_threshold)
        all_val_accuracies.append(best_accuracy)

        print(f"  Fold {fold + 1} - 最佳阈值: {best_threshold:.4f}, 验证准确率: {best_accuracy:.4f}")

        models.append(model)

    # 计算平均准确率
    ensemble_accuracy = np.mean(all_val_accuracies)
    print(f"\n[K-Fold 结果] 平均验证准确率: {ensemble_accuracy:.4f}")
    print(f"  各 Fold 准确率: {[f'{acc:.4f}' for acc in all_val_accuracies]}")
    print(f"  各 Fold 最佳阈值: {[f'{th:.4f}' for th in best_thresholds]}")

    return models, best_thresholds, ensemble_accuracy


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序入口"""

    print("="*70)
    print("牛脸识别 (Cow Face Verification) 项目")
    print("="*70)

    # 设置随机种子
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    print(f"\n[系统信息]")
    print(f"  设备: {Config.DEVICE}")
    print(f"  多卡并行: {Config.USE_MULTI_GPU}")
    print(f"  混合精度: {Config.USE_AMP}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  K-Fold: {'是' if Config.USE_KFOLD else '否'}")

    # ========== 阶段 1：数据加载和训练 ==========

    # 加载训练数据集
    print(f"\n[阶段 1：训练阶段]")
    print(f"  加载训练数据集...")

    train_dataset = CowTrainDataset(
        root_dir=Config.TRAIN_DIR,
        transform=None  # 先不应用增强，后续根据索引划分
    )

    # 动态设置类别数
    Config.NUM_CLASSES = len(train_dataset.id_to_label)
    print(f"  牛的总数: {Config.NUM_CLASSES}")
    print(f"  总图片数: {len(train_dataset)}")

    # 使用 K-Fold 或传统的单次训练
    if Config.USE_KFOLD:
        # K-Fold 交叉验证训练
        models, best_thresholds, ensemble_accuracy = train_with_kfold(
            train_dataset=train_dataset,
            num_folds=Config.NUM_FOLDS
        )
        # 使用平均阈值
        best_threshold = np.mean(best_thresholds)
        best_accuracy = ensemble_accuracy
    else:
        # 传统的单次训练方式
        num_samples = len(train_dataset)
        num_val = int(num_samples * Config.VAL_SPLIT)
        num_train = num_samples - num_val

        indices = np.random.permutation(num_samples)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]

        # 创建具有数据增强的训练/验证数据集
        train_dataset_augmented = CowTrainDataset(
            root_dir=Config.TRAIN_DIR,
            transform=get_train_transforms(Config.INPUT_SIZE),
            indices=train_indices.tolist()
        )

        val_dataset = CowTrainDataset(
            root_dir=Config.TRAIN_DIR,
            transform=get_test_transforms(Config.INPUT_SIZE),
            indices=val_indices.tolist()
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset_augmented,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )

        print(f"  训练集: {num_train} 张图片")
        print(f"  验证集: {num_val} 张图片")

        # 创建模型
        print(f"\n  创建模型...")
        model = CowFaceModel(
            num_classes=Config.NUM_CLASSES,
            feature_dim=512,
            pretrained=True
        )
        print(f"  模型参数数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

        # 训练模型
        print(f"\n[阶段 1：训练模型]")
        model = train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            num_epochs=Config.NUM_EPOCHS,
            device=Config.DEVICE
        )

        # ========== 阶段 2：阈值搜索 ==========

        print(f"\n[阶段 2：阈值搜索]")

        # 加载最佳模型
        model_to_load = model.module if hasattr(model, 'module') else model
        saved_state = torch.load(Config.CHECKPOINT_PATH, map_location=Config.DEVICE)
        model_to_load.load_state_dict(saved_state)
        model_to_load.eval()

        # 提取验证集特征
        print(f"  提取验证集特征...")
        val_features, _ = extract_features(model_to_load, val_loader, Config.DEVICE)

        # 搜索最佳阈值
        best_threshold, best_accuracy = search_best_threshold(
            val_dataset,
            val_features,
            threshold_range=Config.THRESHOLD_RANGE
        )

        # 保存用于推理的最终模型
        models = [model_to_load]

    # ========== 阶段 3：生成提交 ==========

    print(f"\n[阶段 3：生成提交]")

    # 读取测试数据
    test_pairs = load_test_csv(Config.TEST_CSV)

    # 对于 K-Fold，使用集成模型投票
    if Config.USE_KFOLD:
        print(f"  使用 {len(models)} 个模型进行集成预测...")
        all_predictions = []
        for fold_idx, model in enumerate(models):
            model_to_load = model.module if hasattr(model, 'module') else model
            model_to_load.eval()
            predictions = generate_submission(
                model=model_to_load,
                test_dir=Config.TEST_DIR,
                test_pairs=test_pairs,
                best_threshold=best_thresholds[fold_idx],
                device=Config.DEVICE,
                use_tta=True
            )
            all_predictions.append(predictions)

        # 集成投票：多数投票
        ensemble_predictions = []
        for pred_idx in range(len(test_pairs)):
            votes = [all_predictions[fold][pred_idx] for fold in range(len(models))]
            ensemble_pred = 1 if sum(votes) > len(models) / 2 else 0
            ensemble_predictions.append(ensemble_pred)
        predictions = ensemble_predictions
    else:
        # 单模型预测
        model_to_load = models[0]
        predictions = generate_submission(
            model=model_to_load,
            test_dir=Config.TEST_DIR,
            test_pairs=test_pairs,
            best_threshold=best_threshold,
            device=Config.DEVICE,
            use_tta=True
        )

    # 保存提交文件
    save_submission(
        predictions=predictions,
        test_pairs=test_pairs,
        output_path=Config.SUBMISSION_PATH
    )

    print("\n" + "="*70)
    print("项目完成！")
    print("="*70)
    print(f"提交文件: {Config.SUBMISSION_PATH}")
    print(f"最佳阈值: {best_threshold:.3f}")
    print(f"验证集准确率: {best_accuracy:.4f}")
    if Config.USE_KFOLD:
        print(f"训练模式: K-Fold ({Config.NUM_FOLDS}-Fold) 集成")


if __name__ == '__main__':
    main()
