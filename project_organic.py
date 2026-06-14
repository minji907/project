# ==============================================
# 쓰레기 분류 프로젝트 (PyTorch + MobileNetV2)
# ==============================================

# Step 1 — 라이브러리 임포트
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights
from sklearn.metrics import classification_report, confusion_matrix

print('모든 라이브러리 임포트 성공!')

# Step 2 — GPU 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'사용 장치: {device}')
if torch.cuda.is_available():
    print(f'GPU 이름: {torch.cuda.get_device_name(0)}')

# Step 3 — 설정
DATASET_PATH = r'C:\Users\user\.cache\kagglehub\datasets\techsash\waste-classification-data\versions\1\DATASET\DATASET'
TRAIN_DIR = os.path.join(DATASET_PATH, 'TRAIN')
TEST_DIR = os.path.join(DATASET_PATH, 'TEST')
OUTPUT_DIR = r'C:\Users\user\Desktop\프로젝트\output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_P1 = 15
EPOCHS_P2 = 10
LEARNING_RATE = 1e-4

print('설정 완료!')

# 데이터셋 구조 확인
for split in ['TRAIN', 'TEST']:
    for cls in os.listdir(os.path.join(DATASET_PATH, split)):
        count = len(os.listdir(os.path.join(DATASET_PATH, split, cls)))
        label = 'Organic' if cls == 'O' else 'Inorganic'
        print(f'{split}/{cls} ({label}): {count}장')

# Step 4 — 데이터 전처리 & 증강
# 훈련용 증강
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomRotation(30),  # 30도 회전
    transforms.RandomHorizontalFlip(),  # 좌우 반전
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 확대/축소
    transforms.ColorJitter(brightness=0.2),  # 밝기 조절
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet 평균
                         [0.229, 0.224, 0.225])  # ImageNet 표준편차
])

# 검증/테스트용 (정규화만)
val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 데이터셋 로드
full_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_transform)

# 훈련/검증 분할 (85% / 15%)
val_size = int(0.15 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# 검증셋은 증강 없이 (transform 교체)
val_dataset.dataset = datasets.ImageFolder(TRAIN_DIR, transform=val_transform)

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f'\n클래스 인덱스: {full_train_dataset.class_to_idx}')
print(f'훈련 샘플 수 : {train_size}')
print(f'검증 샘플 수 : {val_size}')
print(f'테스트 샘플 수: {len(test_dataset)}')

# Step 5 — 샘플 이미지 시각화
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Sample Dataset', fontsize=16, fontweight='bold')

# 원본 이미지용 transform (정규화 없이)
raw_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
]))
raw_loader = DataLoader(raw_dataset, batch_size=10, shuffle=True)
batch_imgs, batch_labels = next(iter(raw_loader))
class_map = {v: k for k, v in raw_dataset.class_to_idx.items()}

for i, ax in enumerate(axes.flat):
    if i < len(batch_imgs):
        img = batch_imgs[i].permute(1, 2, 0).numpy()  # (C,H,W) → (H,W,C)
        ax.imshow(img)
        label = class_map[batch_labels[i].item()]
        color = '#4CAF50' if label == 'O' else '#F44336'
        ax.set_title('Organic' if label == 'O' else 'Inorganic', color=color, fontweight='bold')
        ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sample_dataset.png'), dpi=150, bbox_inches='tight')
plt.show()
print('샘플 이미지 저장 완료!')

# Step 6 — 모델 생성 (Transfer Learning MobileNetV2)
# 사전 학습된 MobileNetV2 불러오기
base_model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

# 기본 모델 가중치 고정 (Phase 1)
for param in base_model.parameters():
    param.requires_grad = False

# 분류기 교체 (이진 분류용)
base_model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(base_model.last_channel, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.2),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)

model = base_model.to(device)
print('모델 생성 완료!')
print(f'총 파라미터 수: {sum(p.numel() for p in model.parameters()):,}')
print(f'학습 가능한 파라미터: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

# Step 7 — 학습 함수 정의
criterion = nn.BCELoss()  # 이진 분류 손실함수


def train_epoch(model, loader, optimizer):
    """한 에폭 학습"""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader):
    """검증/테스트 평가"""
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

    return total_loss / total, correct / total


# Step 8 — 1차 학습 (Feature Extraction: 분류기만 학습)
print('\n 1차 학습 시작: 특징 추출 (분류기만 학습)...')

optimizer_p1 = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)
scheduler_p1 = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_p1, factor=0.5, patience=3
)

best_val_acc = 0
patience_counter = 0
history1 = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(EPOCHS_P1):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer_p1)
    val_loss, val_acc = evaluate(model, val_loader)
    scheduler_p1.step(val_loss)

    history1['train_loss'].append(train_loss)
    history1['train_acc'].append(train_acc)
    history1['val_loss'].append(val_loss)
    history1['val_acc'].append(val_acc)

    print(f'Epoch {epoch + 1:02d}/{EPOCHS_P1} | '
          f'Train Loss: {train_loss:.4f} Acc: {train_acc * 100:.2f}% | '
          f'Val Loss: {val_loss:.4f} Acc: {val_acc * 100:.2f}%')

    # 최고 모델 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model_phase1.pth'))
        print(f'  최고 모델 저장! (val_acc: {val_acc * 100:.2f}%)')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 5:
            print('조기 종료!')
            break

print(f'\n 1차 학습 완료! 최고 검증 정확도: {best_val_acc * 100:.2f}%')

# Step 9 — 2차 학습 (Fine Tuning: 전체 모델 학습)
print('\n 2차 학습 시작: Fine Tuning (전체 모델 학습)...')

# 전체 모델 학습 가능하게 설정
for param in model.parameters():
    param.requires_grad = True

# 최고 모델 불러오기
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'model_phase1.pth')))

optimizer_p2 = optim.Adam(model.parameters(), lr=LEARNING_RATE / 10)  # 더 낮은 학습률
scheduler_p2 = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_p2, factor=0.5, patience=3
)

best_val_acc2 = 0
patience_counter2 = 0
history2 = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(EPOCHS_P2):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer_p2)
    val_loss, val_acc = evaluate(model, val_loader)
    scheduler_p2.step(val_loss)

    history2['train_loss'].append(train_loss)
    history2['train_acc'].append(train_acc)
    history2['val_loss'].append(val_loss)
    history2['val_acc'].append(val_acc)

    print(f'Epoch {epoch + 1:02d}/{EPOCHS_P2} | '
          f'Train Loss: {train_loss:.4f} Acc: {train_acc * 100:.2f}% | '
          f'Val Loss: {val_loss:.4f} Acc: {val_acc * 100:.2f}%')

    if val_acc > best_val_acc2:
        best_val_acc2 = val_acc
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model_final.pth'))
        print(f'최고 모델 저장! (val_acc: {val_acc * 100:.2f}%)')
        patience_counter2 = 0
    else:
        patience_counter2 += 1
        if patience_counter2 >= 5:
            print('조기 종료!')
            break

print(f'\n 2차 학습 완료! 최고 검증 정확도: {best_val_acc2 * 100:.2f}%')

# Step 10 — 학습 곡선 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Training History', fontsize=16, fontweight='bold')

# 전체 history 합치기
all_train_acc = history1['train_acc'] + history2['train_acc']
all_val_acc = history1['val_acc'] + history2['val_acc']
all_train_loss = history1['train_loss'] + history2['train_loss']
all_val_loss = history1['val_loss'] + history2['val_loss']
p1_end = len(history1['train_acc'])

# 정확도 그래프
axes[0].plot(all_train_acc, label='Train Accuracy', color='#4CAF50')
axes[0].plot(all_val_acc, label='Val Accuracy', color='#2196F3')
axes[0].axvline(x=p1_end, color='gray', linestyle='--', label='Phase 1→2')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 손실 그래프
axes[1].plot(all_train_loss, label='Train Loss', color='#F44336')
axes[1].plot(all_val_loss, label='Val Loss', color='#FF9800')
axes[1].axvline(x=p1_end, color='gray', linestyle='--', label='Phase 1→2')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
plt.show()
print(' 학습 곡선 저장 완료!')

# Step 11 — 최종 테스트 평가
print('\n 최종 테스트 평가...')
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'model_final.pth')))
test_loss, test_acc = evaluate(model, test_loader)
print(f'테스트 정확도: {test_acc * 100:.2f}%')
print(f'테스트 손실  : {test_loss:.4f}')

# 예측값 수집 (confusion matrix용)
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = (outputs.squeeze() > 0.5).long().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# 분류 리포트
print('\n분류 리포트:')
print(classification_report(all_labels, all_preds, target_names=['Organic', 'Inorganic']))

# Confusion Matrix 시각화
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Organic', 'Inorganic'],
            yticklabels=['Organic', 'Inorganic'])
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()
print('모든 완료!')
