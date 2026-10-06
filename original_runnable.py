# -*- coding: utf-8 -*-
"""
學姊原版程式碼（可執行版，用來和修正版比較結果）

★ 保留原本的「方法問題」，例如：
    - 測試集加雜訊後混進訓練集（資料洩漏）
    - 用測試集做 scheduler / 早停 / 選最佳模型
    - SHAP 使用未經 QuantileTransformer 轉換的特徵、背景取前 100 筆
    - 擴增雜訊為固定 0.01、預測時 StandardScaler 在「擴增前」資料上 fit
    - 沒有固定亂數種子（每次跑結果會略有不同）

★ 只修改了「不修就跑不動」的部分：
    1. 貼上時被切斷的字串、數字，以及消失的縮排
    2. 路徑改成相對於本檔案的資料夾，結果全部存到 output_old/（不會蓋掉修正版的 output/）
    3. imputena 與新版 pandas 不相容 -> 改用 sklearn IterativeImputer（與修正版相同）
    4. ReduceLROnPlateau 的 verbose 參數在新版 PyTorch 已移除
    5. 原本手動產生的 4.PM2.5_train.csv / 4.PM2.5_test.csv 改成程式自動產生
    6. 原本靠手動改名的 3.201901.csv，改成直接讀第 4 區輸出的標準化檔
    7. 論文不用的 SHAP 方向性圖包 try/except，失敗時不會中斷後面的預測

資料夾結構（全部放同一個資料夾）：
    2.all.xlsx, 3.train.csv, 3.test.csv, 2.2019_01_MICE.csv, 3.teds.csv
    output_old/   <- 自動建立
"""

#%% ========== 0. 共用設定 ==========
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

OUT_DIR = BASE_DIR / "output_old"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_MICE = True   # 第 1 區的結果後面用不到（後面直接讀 3.train.csv），想省時間可改 False


#%% ========== 1. MICE 補值 ==========
if RUN_MICE:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    data = pd.read_excel(BASE_DIR / "2.all.xlsx")
    impu_data_pre = data.iloc[:, 6:].apply(pd.to_numeric, errors="coerce")

    # 檢查超出 float64 範圍的數值
    data_array = impu_data_pre.to_numpy(dtype=float)
    exceed_mask = np.isinf(data_array)
    if np.any(exceed_mask):
        print("發現超出 float64 範圍的數值：")
        for row_idx, col_idx in np.argwhere(exceed_mask)[:20]:
            print(f"位置 (列: {row_idx}, 欄: '{impu_data_pre.columns[col_idx]}') "
                  f"的數值為 {data_array[row_idx, col_idx]}")
        impu_data_pre = impu_data_pre.mask(exceed_mask)
    else:
        print("沒有超出 float64 範圍的數值")

    feat_names = impu_data_pre.columns
    imputed_list = []
    for i in range(4):                              # 生成 4 組插補結果
        imp = IterativeImputer(max_iter=10, sample_posterior=True,
                               keep_empty_features=True, random_state=i)
        imputed_list.append(imp.fit_transform(impu_data_pre.to_numpy(dtype=float)))
        print(f"第 {i + 1}/4 組補值完成")
    new_data = np.mean(imputed_list, axis=0)        # 4 次插補結果平均

    data[feat_names] = pd.DataFrame(new_data, columns=feat_names, index=data.index)
    data.to_csv(OUT_DIR / "MICE_all.csv", index=False)
    print("補值後缺失比例：", np.sum(data.isna().values) / (data.shape[0] * data.shape[1]))


#%% ========== 2. 資料擴增 + 標準化 ==========
train_df = pd.read_csv(BASE_DIR / "3.train.csv")
test_df = pd.read_csv(BASE_DIR / "3.test.csv")

# 目標欄位為多欄位
target_cols = ["PM2.5", "K+", "Na+", "NH4+", "NO3-", "SO42-"]
# 特徵欄位（排除目標欄位）
feature_cols = [col for col in train_df.columns if col not in target_cols]


# 資料擴增函式（隨機為部分特徵加上噪音，目標欄位不變）
def selective_augment(df, target_cols, n_aug=9, noise_level=0.01):
    augmented = []
    for _ in range(n_aug):
        augmented_batch = []
        for i in range(len(df)):
            row = df.iloc[i].copy()
            # 取出非目標欄位特徵
            features = row.drop(labels=target_cols)
            # 隨機選擇 1 到 9 個特徵加噪音
            selected_cols = np.random.choice(
                features.index,
                size=np.random.randint(1, min(10, len(features))),
                replace=False)
            noise = np.random.normal(0, noise_level, size=len(selected_cols))
            features[selected_cols] += noise
            # 建立新資料列，保留目標欄位原值
            new_row = features.to_dict()
            for tcol in target_cols:
                new_row[tcol] = row[tcol]
            augmented_batch.append(pd.Series(new_row))
        augmented.append(pd.DataFrame(augmented_batch))
    # 回傳原資料與擴增資料合併
    return pd.concat([df] + augmented, axis=0).reset_index(drop=True)


# 執行訓練資料擴增（保留原始資料）—— 約需數十秒
train_augmented = selective_augment(train_df, target_cols, n_aug=9)

# 標準化：只針對特徵欄位做 fit_transform
features_only = train_augmented[feature_cols]
scaler = StandardScaler()
features_scaled = pd.DataFrame(scaler.fit_transform(features_only),
                               columns=features_only.columns)
# 合併標準化後特徵與目標欄位
train_final_output = pd.concat([train_augmented[target_cols], features_scaled], axis=1)
train_final_output.to_csv(OUT_DIR / "train_aug.csv", index=False)

# 對測試資料做相同標準化（不做擴增）
test_features_only = test_df[feature_cols]
test_features_scaled = pd.DataFrame(scaler.transform(test_features_only),
                                    columns=test_features_only.columns)
test_final_output = pd.concat([test_df[target_cols], test_features_scaled], axis=1)
test_final_output.to_csv(OUT_DIR / "test_aug.csv", index=False)

# 原本是手動從 train_aug / test_aug 挑出 PM2.5 + 特徵存成 4.PM2.5_*.csv，這裡自動產生
train_final_output[["PM2.5"] + feature_cols].to_csv(OUT_DIR / "4.PM2.5_train.csv", index=False)
test_final_output[["PM2.5"] + feature_cols].to_csv(OUT_DIR / "4.PM2.5_test.csv", index=False)
print("擴增後訓練資料筆數：", len(train_augmented), "測試資料筆數：", len(test_df))


#%% ========== 3. MLP 建模（PM2.5）==========
def mean_absolute_percentage_error(y_pred, y_true):
    epsilon = 1e-8
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def r2_score_torch(y_pred, y_true):
    y_true_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_true_mean) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return r2


df_train = pd.read_csv(OUT_DIR / "4.PM2.5_train.csv")
df_extest = pd.read_csv(OUT_DIR / "4.PM2.5_test.csv")
label = df_train.loc[:, "PM2.5"].copy().values
feature = df_train.iloc[:, 1:].copy().values
extest_label = df_extest.loc[:, "PM2.5"].copy().values
ex_test_feature = df_extest.iloc[:, 1:].copy().values

mm = QuantileTransformer()
feature_scaled = mm.fit_transform(feature)
ex_test_feature_scaled = mm.transform(ex_test_feature)

# （原版做法）把測試集加雜訊後，連同真實標籤一起併入訓練資料
noise_std = 0.01
noise = np.random.normal(loc=0.0, scale=noise_std, size=ex_test_feature_scaled.shape)
augmented_features = ex_test_feature_scaled + noise
augmented_labels = extest_label.copy()
feature_all = np.vstack((feature_scaled, augmented_features))
label_all = np.hstack((label, augmented_labels))

# 轉換成 Tensor
X_train_A_tensor = torch.from_numpy(feature_all).float()
y_train_A_tensor = torch.from_numpy(label_all).float().view(-1, 1)
X_test_A_tensor = torch.from_numpy(ex_test_feature_scaled).float()
y_test_A_tensor = torch.from_numpy(extest_label).float().view(-1, 1)

batch_size = 64
train_dataset = TensorDataset(X_train_A_tensor, y_train_A_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim=feature_all.shape[1], output_dim=1):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.dropout3 = nn.Dropout(0.05)
        self.fc4 = nn.Linear(16, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x


model = MLPRegressor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-6)  # 已移除 verbose

# 早停設定
best_r2 = -float('inf')
patience = 40
no_improve_counter = 0
best_model_path = OUT_DIR / "PM2.5_best_mlp_model.pth"
mae_loss = lambda outputs, targets: torch.mean(torch.abs(outputs - targets))

performance = {
    "train_loss": [], "test_loss": [],
    "train_mae": [], "test_mae": [],
    "train_R2": [], "test_R2": [],
    "train_mape": [], "test_mape": [],
}

epochs = 500
for epoch in range(epochs):
    model.train()
    train_losses = []
    train_maes = []
    train_mapes = []
    train_r2s = []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        train_maes.append(mae_loss(outputs, yb).item())
        train_mapes.append(mean_absolute_percentage_error(outputs, yb).item())
        train_r2s.append(r2_score_torch(outputs, yb).item())
    train_loss = sum(train_losses) / len(train_losses)
    train_mae = sum(train_maes) / len(train_maes)
    train_mape = sum(train_mapes) / len(train_mapes)
    train_r2 = sum(train_r2s) / len(train_r2s)

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_A_tensor)
        test_loss = criterion(test_outputs, y_test_A_tensor).item()
        test_mae = mae_loss(test_outputs, y_test_A_tensor).item()
        test_mape = mean_absolute_percentage_error(test_outputs, y_test_A_tensor).item()
        test_r2 = r2_score_torch(test_outputs, y_test_A_tensor).item()

    scheduler.step(test_loss)

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
          f"MAE: {train_mae:.4f}/{test_mae:.4f} | "
          f"MAPE: {train_mape:.2f}/{test_mape:.2f} | "
          f"R²: {train_r2:.4f}/{test_r2:.4f}")

    if test_r2 > best_r2:
        best_r2 = test_r2
        torch.save(model.state_dict(), best_model_path)
        no_improve_counter = 0
    else:
        no_improve_counter += 1
        if no_improve_counter >= patience:
            print(f"\n[Early Stopping] Test R² 未提升超過 {patience} epochs，停止訓練。")
            break

    performance["train_loss"].append(train_loss)
    performance["test_loss"].append(test_loss)
    performance["train_mae"].append(train_mae)
    performance["test_mae"].append(test_mae)
    performance["train_mape"].append(train_mape)
    performance["test_mape"].append(test_mape)
    performance["train_R2"].append(train_r2)
    performance["test_R2"].append(test_r2)

sns.set(font_scale=1.8)
sns.set_style("whitegrid")
sns.set_palette("cool")
length = range(1, len(performance["train_loss"]) + 1)

# 繪製損失曲線
plt.figure(figsize=(20, 16))
ax = plt.subplot(2, 2, 1)
ax.plot(length, performance["train_loss"], label='Train', marker='o', markersize=3)
ax.plot(length, performance["test_loss"], label='Test', marker='o', markersize=3)
ax.set_xlabel('Epoch', weight="bold")
ax.set_ylabel('Loss', weight="bold")
ax.set_title('PM$_{2.5}$ Loss Curve', weight="bold", fontsize=24)
ax.legend()
ax.grid(True, linestyle="--")

ax1 = plt.subplot(2, 2, 2)
ax1.plot(length, performance["train_mae"], label='Train', marker='o', markersize=3)
ax1.plot(length, performance["test_mae"], label='Test', marker='o', markersize=3)
ax1.set_xlabel('Epoch', weight="bold")
ax1.set_ylabel('Loss', weight="bold")
ax1.set_title('PM$_{2.5}$ MAE Curve', weight="bold", fontsize=24)
ax1.legend()
ax1.grid(True, linestyle="--")

ax2 = plt.subplot(2, 2, 3)
ax2.plot(length, performance["train_mape"], label='Train', marker='o', markersize=3)
ax2.plot(length, performance["test_mape"], label='Test', marker='o', markersize=3)
ax2.set_xlabel('Epoch', weight="bold")
ax2.set_ylabel('Loss', weight="bold")
ax2.set_title('PM$_{2.5}$ MAPE Curve', weight="bold", fontsize=24)
ax2.legend()
ax2.grid(True, linestyle="--")

ax3 = plt.subplot(2, 2, 4)
ax3.plot(length, performance["train_R2"], label='Train', marker='o', markersize=3)
ax3.plot(length, performance["test_R2"], label='Test', marker='o', markersize=3)
ax3.set_xlabel('Epoch', weight="bold")
ax3.set_ylabel('Loss', weight="bold")
ax3.set_title('PM$_{2.5}$ R$^2$ Curve', weight="bold", fontsize=24)
ax3.legend()
ax3.grid(True, linestyle="--")
plt.savefig(OUT_DIR / "PM2.5_MLP_loss_curve.png")
plt.show()

# 載入最佳模型
model.load_state_dict(torch.load(best_model_path))
model.eval()

# 計算訓練集的預測及 R²
with torch.no_grad():
    train_preds = model(X_train_A_tensor)
    train_r2 = r2_score_torch(train_preds, y_train_A_tensor).item()

# 計算測試集的預測及 R²
with torch.no_grad():
    test_preds = model(X_test_A_tensor)
    test_r2 = r2_score_torch(test_preds, y_test_A_tensor).item()

print(f"Best Model Train R²: {train_r2:.4f}")
print(f"Best Model Test R²: {test_r2:.4f}")

# 補充：和修正版相同格式的測試集指標，方便比較
_te_mae = torch.mean(torch.abs(test_preds - y_test_A_tensor)).item()
_te_rmse = torch.sqrt(torch.mean((test_preds - y_test_A_tensor) ** 2)).item()
_te_mape = mean_absolute_percentage_error(test_preds, y_test_A_tensor).item()
print(f"[比較用] Test | R2 {test_r2:.4f} | MAE {_te_mae:.3f} | "
      f"RMSE {_te_rmse:.3f} | MAPE {_te_mape:.2f}%")

# 將 tensor 轉成 numpy 陣列
y_true_np = y_test_A_tensor.numpy().flatten()   # 真實值
y_pred_np = test_preds.numpy().flatten()        # 預測值

sns.set(font_scale=2)
sns.set_style("whitegrid")
plt.figure(figsize=(8, 8))
plt.scatter(y_pred_np, y_true_np, alpha=0.6)
plt.plot([np.min(y_true_np), np.max(y_true_np)],
         [np.min(y_true_np), np.max(y_true_np)], 'k-',
         label=r"R$^2$={:.2f}".format(test_r2))
plt.xlabel("Predicted Values", weight="bold", fontsize=20)
plt.ylabel("True Values", weight="bold", fontsize=20)
plt.title("PM$_{2.5}$ Results", weight="bold", fontsize=25)
plt.legend(loc="lower right")
plt.xticks(weight="bold", fontsize=15)
plt.yticks(weight="bold", fontsize=15)
plt.savefig(OUT_DIR / "PM2.5_put.png", dpi=600, bbox_inches="tight")
plt.show()

# ---------- SHAP（原版：用未轉換的特徵、背景取前 100 筆）----------
X_train_np = feature
X_test_np = ex_test_feature

# 若模型在 GPU，需搬回 CPU
if torch.cuda.is_available():
    model = model.cpu()
model.eval()

# 取部分樣本作為 SHAP 背景與解釋樣本
background = torch.from_numpy(X_train_np[:100]).float()
test_samples = torch.from_numpy(X_test_np[:50]).float()

# 建立 DeepExplainer
explainer = shap.DeepExplainer(model, background)
# 計算測試集 SHAP 值
shap_values = explainer.shap_values(test_samples, check_additivity=False)

# 整理 SHAP 結果
shap_values = np.array(shap_values)
shap_values = np.squeeze(shap_values)   # (樣本數, 特徵數)
feature_names = df_train.columns[1:62].tolist()   # 調整特徵欄位數
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
shap_summary_df = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap_value': mean_abs_shap,
})
shap_summary_df = shap_summary_df.sort_values(
    by='mean_abs_shap_value', ascending=False).reset_index(drop=True)
print(shap_summary_df)
shap_summary_df.to_csv(OUT_DIR / "PM2.5_shap.csv", index=False)

plt.figure(figsize=(12, 12))
plt.barh(shap_summary_df['feature'], shap_summary_df['mean_abs_shap_value'],
         color='red', height=0.5)
plt.xlabel('Mean Absolute SHAP Value')
plt.ylabel('Feature')
plt.title('PM$_{2.5}$ Feature Importance', fontsize=20, fontweight='bold')
plt.gca().invert_yaxis()   # 反轉 y 軸，讓重要性最高的特徵在上方
plt.yticks(fontsize=10)
plt.xticks(fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / "PM2.5_shap.png", dpi=300, bbox_inches='tight')
plt.show()

# ---------- 論文中不使用（失敗也不影響後面的預測）----------
try:
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    shap.summary_plot(
        shap_values,
        test_samples.numpy(),
        feature_names=feature_names,
        show=False,
        color_bar=False,
        cmap="bwr",
    )
    fig = plt.gcf()
    ax = plt.gca()
    cmap = plt.get_cmap("bwr")
    norm = mcolors.Normalize(vmin=test_samples.numpy().min(),
                             vmax=test_samples.numpy().max())
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    start, end = cbar.ax.get_ylim()
    cbar.ax.set_yticks([start, end])
    cbar.ax.set_yticklabels(['Low', 'High'])
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('Feature value', fontsize=18, labelpad=0.5, rotation=270)
    fig.set_size_inches(10, 10)
    ax.set_xlabel('PM$_{2.5}$ SHAP value (impact on model output)', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Directionality_shap_PM2.5.png", dpi=300)
    plt.show()
except Exception as e:
    print("方向性 SHAP 圖略過：", repr(e))


#%% ========== 4. 預測資料的標準化（原版：scaler 在「擴增前」的訓練資料上 fit）==========
# 跟建模時不同，不進行增強，只針對新資料做標準化
train_df = pd.read_csv(BASE_DIR / "3.train.csv")           # 還沒增強前
test_df = pd.read_csv(BASE_DIR / "2.2019_01_MICE.csv")
train_cols = train_df.columns.tolist()
test_cols = test_df.columns.tolist()

# train 去掉前 6 欄，剩下的為特徵；test 去掉前 3 欄，剩下的為特徵
train_feature_cols = train_cols[6:]
test_feature_cols = test_cols[3:]

scaler = StandardScaler()
train_features_scaled = pd.DataFrame(
    scaler.fit_transform(train_df[train_feature_cols]), columns=train_feature_cols)

# 對 test_df 用相同 scaler 轉換，欄位名稱直接用 train_feature_cols
test_features_scaled = pd.DataFrame(
    scaler.transform(test_df[test_feature_cols]), columns=train_feature_cols)

# 輸出 test 資料（標準化後）
std_output_path = OUT_DIR / "201901_std.csv"
test_features_scaled.to_csv(std_output_path, index=False)
print("標準化完成：", std_output_path)


#%% ========== 5. 預測（原版：重新 fit QuantileTransformer）==========
class MLPRegressor(nn.Module):        # 需與訓練時相同
    def __init__(self, input_dim=61, output_dim=1):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.dropout3 = nn.Dropout(0.05)
        self.fc4 = nn.Linear(16, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x


# 讀取訓練資料（用來 fit QuantileTransformer）
df_train = pd.read_csv(OUT_DIR / "4.PM2.5_train.csv")      # 建模資料集（增強後）
feature_train = df_train.iloc[:, 1:].copy().values
# 讀取測試資料（第 4 區輸出的標準化檔）
df_test = pd.read_csv(std_output_path)
feature_test = df_test.values.copy()

# 重新 fit QuantileTransformer（因為訓練時 mm 沒儲存）
mm = QuantileTransformer()
feature_train_scaled = mm.fit_transform(feature_train)
feature_test_scaled = mm.transform(feature_test)

X_test_tensor = torch.from_numpy(feature_test_scaled).float()

# 建立模型並載入權重
model = MLPRegressor(input_dim=feature_test.shape[1], output_dim=1)
model.load_state_dict(torch.load(best_model_path))
model.eval()   # 設為評估模式，關閉 dropout 和 batchnorm 的訓練行為

with torch.no_grad():
    predictions = model(X_test_tensor)
pred_np = predictions.numpy()

# 加入座標
df_csv = pd.read_csv(BASE_DIR / "3.teds.csv")
df_pred = pd.DataFrame(pred_np, columns=["Prediction"])
df_merged = pd.concat([df_csv, df_pred], axis=1)   # 橫向合併（列數需相同）
pred_path = OUT_DIR / "PM2.5_201901_pred.csv"
df_merged.to_csv(pred_path, index=False)
print("預測完成，結果已儲存：", pred_path)


#%% ========== 6. 和修正版的預測結果比較（需先跑過修正版）==========
new_path = BASE_DIR / "output" / "PM2.5_201901_pred.csv"
if new_path.exists():
    old_p = pd.read_csv(pred_path)["Prediction"].to_numpy()
    new_p = pd.read_csv(new_path)["Prediction"].to_numpy()
    if len(old_p) == len(new_p):
        print(f"預測筆數：{len(old_p)}")
        print(f"舊版預測平均 {old_p.mean():.2f}，修正版預測平均 {new_p.mean():.2f}")
        print(f"兩版預測的相關係數：{np.corrcoef(old_p, new_p)[0, 1]:.4f}")
        print(f"兩版預測的平均絕對差：{np.mean(np.abs(old_p - new_p)):.3f}")
    else:
        print("兩份預測檔列數不同，無法比較")
else:
    print("找不到修正版的預測檔（output/PM2.5_201901_pred.csv），略過比較")
