#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TextCNN 商品评论情感分类
适用于中文和英文评论的二分类任务
"""

import re
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import jieba
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# ==================== 配置参数 ====================
class Config:
    # 中文训练数据路径
    train_cn_pos_path = "sample.positive.txt"
    train_cn_neg_path = "sample.negative.txt"

    # 英文训练数据路径
    train_en_pos_path = "sample.positive.txt"
    train_en_neg_path = "sample.negative.txt"

    # 测试数据路径
    test_cn_path = "test.cn.txt"
    test_en_path = "test.en.txt"

    # 测试标签路径（用于评估）
    test_label_cn_path = "test_label_cn.txt"
    test_label_en_path = "test_label_en.txt"

    # 模型参数
    embed_dim = 128
    num_filters = 100
    filter_sizes = [2, 3, 4, 5]
    dropout = 0.5
    max_len = 200
    vocab_size = 50000

    # 训练参数
    batch_size = 64
    epochs = 10
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 输出配置
    team_name = "MyTeam"
    run_tag = 1


config = Config()


# ==================== 数据解析（修复版）====================
def read_file_content(file_path):
    """读取文件内容，尝试多种编码"""
    encodings = ['utf-8', 'utf-8-sig', 'utf-16', 'utf-16-le', 'utf-16-be', 'gbk', 'gb2312', 'gb18030', 'latin-1']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
                # 检查是否读取成功（包含review标签）
                if '<review' in content:
                    print(f"  使用编码: {encoding}")
                    return content
        except Exception as e:
            continue

    # 最后尝试二进制读取
    try:
        with open(file_path, 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')
            return content
    except:
        pass

    return None


def clean_content(content):
    """清理文件内容中的特殊字符"""
    if content is None:
        return None
    # 统一换行符 - 处理各种换行符组合
    content = content.replace('\r\r\n', '\n')
    content = content.replace('\r\n', '\n')
    content = content.replace('\r', '\n')
    # 移除BOM
    content = content.lstrip('\ufeff')
    return content


def parse_reviews_from_content(content, has_label=False):
    """从内容中解析评论"""
    reviews = []

    if content is None:
        return reviews

    content = clean_content(content)

    if has_label:
        # 带标签格式: <review id="0" label="1"> 或 <review id="0"  label="1">
        pattern = r'<review\s+id="(\d+)"\s+label="(\d+)">\s*(.*?)\s*</review>'
    else:
        # 不带标签格式: <review id="0">
        pattern = r'<review\s+id="(\d+)">\s*(.*?)\s*</review>'

    matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)

    for match in matches:
        if has_label:
            review_id, label, text = match
            text = text.strip()
            # 清理文本中的多余空白
            text = re.sub(r'\s+', ' ', text).strip()
            if text:
                reviews.append({
                    'id': int(review_id),
                    'text': text,
                    'label': int(label)
                })
        else:
            review_id, text = match
            text = text.strip()
            text = re.sub(r'\s+', ' ', text).strip()
            if text:
                reviews.append({
                    'id': int(review_id),
                    'text': text
                })

    return reviews


def parse_train_data(pos_path, neg_path):
    """解析训练数据"""
    reviews = []

    print(f"  正样本文件: {pos_path}")
    print(f"  负样本文件: {neg_path}")

    # 检查文件是否存在
    if not os.path.exists(pos_path):
        print(f"  错误: 正样本文件不存在!")
        return reviews
    if not os.path.exists(neg_path):
        print(f"  错误: 负样本文件不存在!")
        return reviews

    # 读取正面样本
    print("  读取正面样本...")
    content = read_file_content(pos_path)
    if content:
        pos_reviews = parse_reviews_from_content(content, has_label=False)
        for r in pos_reviews:
            r['label'] = 1  # positive
        reviews.extend(pos_reviews)
        print(f"  正面样本数: {len(pos_reviews)}")
    else:
        print("  警告: 无法读取正面样本文件")

    # 读取负面样本
    print("  读取负面样本...")
    content = read_file_content(neg_path)
    if content:
        neg_reviews = parse_reviews_from_content(content, has_label=False)
        for r in neg_reviews:
            r['label'] = 0  # negative
        reviews.extend(neg_reviews)
        print(f"  负面样本数: {len(neg_reviews)}")
    else:
        print("  警告: 无法读取负面样本文件")

    return reviews


def parse_test_data(file_path, has_label=False):
    """解析测试数据"""
    print(f"  测试文件: {file_path}")

    if not os.path.exists(file_path):
        print(f"  错误: 测试文件不存在!")
        return []

    content = read_file_content(file_path)
    if content:
        reviews = parse_reviews_from_content(content, has_label=has_label)
        print(f"  解析到 {len(reviews)} 条评论")
        return reviews
    else:
        print("  警告: 无法读取测试文件")
        return []


# ==================== 分词与词表 ====================
def tokenize_chinese(text):
    """中文分词"""
    text = str(text).lower()
    tokens = list(jieba.cut(text))
    tokens = [t for t in tokens if t.strip()]
    return tokens


def tokenize_english(text):
    """英文分词"""
    text = str(text).lower()
    tokens = re.findall(r'\b[a-zA-Z]+\b', text)
    return tokens


def build_vocab(reviews, is_chinese=True, max_vocab_size=50000):
    """构建词表"""
    counter = Counter()
    tokenize_func = tokenize_chinese if is_chinese else tokenize_english

    for review in reviews:
        tokens = tokenize_func(review['text'])
        counter.update(tokens)

    most_common = counter.most_common(max_vocab_size - 2)

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in most_common:
        vocab[word] = len(vocab)

    return vocab


def text_to_ids(text, vocab, max_len, is_chinese=True):
    """将文本转换为ID序列"""
    tokenize_func = tokenize_chinese if is_chinese else tokenize_english
    tokens = tokenize_func(text)
    ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]

    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [vocab['<PAD>']] * (max_len - len(ids))

    return ids


# ==================== 数据集类 ====================
class ReviewDataset(Dataset):
    def __init__(self, reviews, vocab, max_len, is_chinese=True, has_label=True):
        self.reviews = reviews
        self.vocab = vocab
        self.max_len = max_len
        self.is_chinese = is_chinese
        self.has_label = has_label

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        ids = text_to_ids(review['text'], self.vocab, self.max_len, self.is_chinese)

        if self.has_label:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'label': torch.tensor(review['label'], dtype=torch.long),
                'review_id': review['id']
            }
        else:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'review_id': review['id']
            }


# ==================== TextCNN 模型 ====================
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, num_classes=2, dropout=0.5):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)

        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


# ==================== 训练函数 ====================
def train_model(model, train_loader, val_loader, config, model_name="model"):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_f1 = 0
    best_model_state = None

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            ids = batch['ids'].to(config.device)
            labels = batch['label'].to(config.device)

            optimizer.zero_grad()
            outputs = model(ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        val_acc, val_f1_pos, val_f1_neg = evaluate(model, val_loader, config)
        avg_f1 = (val_f1_pos + val_f1_neg) / 2

        print(f"[{model_name}] Epoch {epoch + 1}/{config.epochs} | "
              f"Loss: {total_loss / len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"F1: {avg_f1:.4f}")

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_model_state:
        model.load_state_dict(best_model_state)

    print(f"[{model_name}] 最佳验证F1: {best_f1:.4f}")
    return model


def evaluate(model, data_loader, config):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            ids = batch['ids'].to(config.device)
            labels = batch['label']

            outputs = model(ids)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1_pos = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    f1_neg = f1_score(all_labels, all_preds, pos_label=0, zero_division=0)

    return acc, f1_pos, f1_neg


def predict(model, data_loader, config):
    """预测"""
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            ids = batch['ids'].to(config.device)
            review_ids = batch['review_id']

            outputs = model(ids)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            for review_id, pred in zip(review_ids, preds):
                label = 'positive' if pred == 1 else 'negative'
                predictions.append((int(review_id), label))

    return predictions


# ==================== 生成提交文件 ====================
def save_submission(predictions, output_path, team_name, run_tag):
    """保存提交文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for review_id, label in sorted(predictions, key=lambda x: x[0]):
            f.write(f"{team_name} {run_tag} {review_id} {label}\n")
    print(f"提交文件已保存: {output_path}")


def evaluate_on_labeled_test(predictions, label_path):
    """在带标签的测试集上评估"""
    content = read_file_content(label_path)
    if not content:
        return None

    test_labeled = parse_reviews_from_content(content, has_label=True)
    if not test_labeled:
        return None

    label_dict = {r['id']: r['label'] for r in test_labeled}

    y_true = []
    y_pred = []

    for review_id, pred_label in predictions:
        if review_id in label_dict:
            y_true.append(label_dict[review_id])
            y_pred.append(1 if pred_label == 'positive' else 0)

    if not y_true:
        return None

    acc = accuracy_score(y_true, y_pred)
    f1_pos = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_neg = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    precision_pos = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_pos = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    precision_neg = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall_neg = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

    return {
        'accuracy': acc,
        'f1_pos': f1_pos,
        'f1_neg': f1_neg,
        'avg_f1': (f1_pos + f1_neg) / 2,
        'precision_pos': precision_pos,
        'recall_pos': recall_pos,
        'precision_neg': precision_neg,
        'recall_neg': recall_neg
    }


# ==================== 主函数 ====================
def train_and_predict(train_pos_path, train_neg_path, test_path, test_label_path,
                      is_chinese, lang_name, config):
    """训练模型并预测"""
    print(f"\n{'=' * 60}")
    print(f"处理{lang_name}数据")
    print(f"{'=' * 60}")

    # 1. 加载训练数据
    print(f"\n[1] 加载{lang_name}训练数据...")
    train_reviews = parse_train_data(train_pos_path, train_neg_path)

    if not train_reviews:
        print(f"错误: 无法加载{lang_name}训练数据!")
        return None

    print(f"\n训练样本总数: {len(train_reviews)}")
    pos_count = sum(1 for r in train_reviews if r['label'] == 1)
    neg_count = sum(1 for r in train_reviews if r['label'] == 0)
    print(f"正样本: {pos_count}, 负样本: {neg_count}")

    # 显示样本示例
    if train_reviews:
        print(f"\n样本示例:")
        print(f"  正样本: {train_reviews[0]['text'][:50]}...")
        for r in train_reviews:
            if r['label'] == 0:
                print(f"  负样本: {r['text'][:50]}...")
                break

    # 2. 构建词表
    print(f"\n[2] 构建{lang_name}词表...")
    vocab = build_vocab(train_reviews, is_chinese, config.vocab_size)
    print(f"词表大小: {len(vocab)}")

    # 3. 划分训练集和验证集
    print(f"\n[3] 划分训练集和验证集...")
    np.random.seed(42)
    indices = np.random.permutation(len(train_reviews))
    val_size = int(len(train_reviews) * 0.1)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_data = [train_reviews[i] for i in train_indices]
    val_data = [train_reviews[i] for i in val_indices]

    print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}")

    # 4. 创建数据加载器
    train_dataset = ReviewDataset(train_data, vocab, config.max_len, is_chinese)
    val_dataset = ReviewDataset(val_data, vocab, config.max_len, is_chinese)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # 5. 创建模型
    print(f"\n[4] 创建TextCNN模型...")
    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=config.embed_dim,
        num_filters=config.num_filters,
        filter_sizes=config.filter_sizes,
        num_classes=2,
        dropout=config.dropout
    ).to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 6. 训练模型
    print(f"\n[5] 开始训练{lang_name}模型...")
    model = train_model(model, train_loader, val_loader, config, model_name=lang_name)

    # 7. 在测试集上预测
    print(f"\n[6] {lang_name}测试集预测...")

    test_reviews = parse_test_data(test_path, has_label=False)

    if test_reviews:
        test_dataset = ReviewDataset(test_reviews, vocab, config.max_len, is_chinese, has_label=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

        predictions = predict(model, test_loader, config)

        # 保存提交文件
        lang_suffix = "CN" if is_chinese else "EN"
        output_path = f"{config.team_name}_{config.run_tag}_{lang_suffix}.txt"
        save_submission(predictions, output_path, config.team_name, config.run_tag)

        # 如果有标签文件，评估结果
        if os.path.exists(test_label_path):
            print(f"\n[7] 在标注测试集上评估...")
            results = evaluate_on_labeled_test(predictions, test_label_path)

            if results:
                print(f"\n{lang_name}测试集评估结果:")
                print(f"  准确率: {results['accuracy']:.4f}")
                print(
                    f"  正类 - P: {results['precision_pos']:.4f}, R: {results['recall_pos']:.4f}, F1: {results['f1_pos']:.4f}")
                print(
                    f"  负类 - P: {results['precision_neg']:.4f}, R: {results['recall_neg']:.4f}, F1: {results['f1_neg']:.4f}")
                print(f"  平均F1: {results['avg_f1']:.4f}")
                return results
    else:
        print(f"警告: 无法解析{lang_name}测试文件!")

    return None


def main():
    print("=" * 60)
    print("TextCNN 商品评论情感分类")
    print("=" * 60)
    print(f"运行设备: {config.device}")
    print(f"队名: {config.team_name}")
    print(f"Run Tag: {config.run_tag}")

    # 首先检查文件路径
    print("\n检查文件路径...")
    paths_to_check = [
        ("中文训练正样本", config.train_cn_pos_path),
        ("中文训练负样本", config.train_cn_neg_path),
        ("英文训练正样本", config.train_en_pos_path),
        ("英文训练负样本", config.train_en_neg_path),
        ("中文测试集", config.test_cn_path),
        ("英文测试集", config.test_en_path),
    ]

    all_exist = True
    for name, path in paths_to_check:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")
        if not exists:
            all_exist = False

    if not all_exist:
        print("\n警告: 部分文件不存在，请检查路径是否正确!")
        print("提示: 请确认文件夹中是否有sample_positive.txt和sample_negative.txt文件")

    results = {}

    # 训练并预测中文模型
    cn_results = train_and_predict(
        train_pos_path=config.train_cn_pos_path,
        train_neg_path=config.train_cn_neg_path,
        test_path=config.test_cn_path,
        test_label_path=config.test_label_cn_path,
        is_chinese=True,
        lang_name="中文",
        config=config
    )
    if cn_results:
        results['中文'] = cn_results

    # 训练并预测英文模型
    en_results = train_and_predict(
        train_pos_path=config.train_en_pos_path,
        train_neg_path=config.train_en_neg_path,
        test_path=config.test_en_path,
        test_label_path=config.test_label_en_path,
        is_chinese=False,
        lang_name="英文",
        config=config
    )
    if en_results:
        results['英文'] = en_results

    # 打印最终结果汇总
    print("\n" + "=" * 60)
    print("最终结果汇总")
    print("=" * 60)

    if results:
        for lang, res in results.items():
            print(f"\n{lang}:")
            print(f"  准确率: {res['accuracy']:.4f}")
            print(f"  F1_pos: {res['f1_pos']:.4f}, F1_neg: {res['f1_neg']:.4f}")
            print(f"  平均F1: {res['avg_f1']:.4f}")
    else:
        print("\n没有成功训练任何模型，请检查数据文件路径!")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()