# -*- coding: utf-8 -*-
# MICE 補值 -> 資料切分 -> 資料擴增/標準化 -> MLP 建模 -> SHAP
# 資料夾結構（所有檔案都放在同一個資料夾 "MLP pychome test"）：
#   MLP_test/
#   pm25_model_training.py     ← 建模程式
#   2.all.xlsx                 ← 要放的
#   預測資料（例如 2019_02.xlsx）不需要放在建模資料夾
#   預測程式會自己產生 2019_02_MICE.csv
# output/                         ← 程式自動建立
# MICE_train.csv
# MICE_test.csv
# output_modeling/
#     train_aug.csv       ← Training，有擴增
#     val.csv             ← Validation，無擴增
#     test.csv            ← Test，無擴增
#     models/
#         mice_imputers.pkl
#         std_scaler.pkl
#         (target)_quantile.pkl
#         (target)_best_mlp_model.pth
#     (target)/
#         (target)_MLP_loss_curve.png
#         (target)_scatter.png
#         (target)_shap.csv
#         (target)_shap.png
#         Directionality_shap_(target).png

#%% ========== 0. 共用設定（每次執行前先跑這一格）==========
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# 設定 Matplotlib 優先讀取 Windows 內建微軟正黑體，並修復負號顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 正常顯示負號

# ---- 路徑：以「本檔案所在資料夾」為基準，不依賴工作目錄 ----
try:
    BASE_DIR = Path(__file__).resolve().parent  # 若是以 python 指令執行腳本，取得該腳本所在的母資料夾路徑
except NameError:                               # 若在 Jupyter 或 Spyder 等互動式介面執行（沒有 __file__ 變數）
    BASE_DIR = Path.cwd()                      # 則退回使用目前的工作目錄（Current Working Directory）

OUT_DIR = BASE_DIR / "output"                  # 設定所有輸出檔案（圖表、預測 CSV）存放的根目錄
MODELING_DIR = OUT_DIR / "output_modeling"
MODEL_DIR = MODELING_DIR / "models"
for d in (OUT_DIR, MODEL_DIR):                 # 逐一檢查輸出目錄與模型目錄
    d.mkdir(parents=True, exist_ok=True)       # 若資料夾不存在則自動建立，已存在則略過不報錯

SEED = 42                                      # 可改，定義隨機種子，確保所有隨機抽樣、資料切分與訓練結果可再現
TARGET_COLS = ["PM2.5", "K+", "Na+", "NH4+", "NO3-", "SO42-"]  # 資料集裡所有的預測目標候選欄位清單
TARGET = "NO3-"                                  # 可改，本次要建模的目標欄位名稱，更換此處即可更換目標
LABEL = "NO3$^-$" if TARGET == "NO3-" else TARGET
## 圖表標籤名稱，支援 LaTeX 上下標

TARGET_DIR = MODELING_DIR / TARGET             # 為當前目標建立專屬輸出目錄（例如 output/output_modeling/K+/）
TARGET_DIR.mkdir(parents=True, exist_ok=True)  # 自動建立該目標專屬的儲存資料夾

np.random.seed(SEED)                           # 設定 NumPy 隨機種子
torch.manual_seed(SEED)                        # 設定 PyTorch 隨機種子，確保權重初始化一致


class MLPRegressor(nn.Module):                 # 繼承 PyTorch 的 nn.Module 類別
    """訓練與預測共用同一個定義，避免兩邊不一致。"""

    def __init__(self, input_dim, output_dim=1):
        super().__init__()                     # 呼叫父類別的初始化函式
        self.fc1 = nn.Linear(input_dim, 64)    # 第 1 個全連接層：將輸入維度特徵映射到 64 個神經元
        self.bn1 = nn.BatchNorm1d(64)          # 第 1 層批次正規化（Batch Normalization），穩定數值分佈加速收斂
        self.dropout1 = nn.Dropout(0.1)        # 第 1 層 Dropout，訓練時隨機將 20% 神經元設為 0 以防止過擬合
        self.fc2 = nn.Linear(64, 32)           # 第 2 個全連接層：64 維降至 32 維
        self.bn2 = nn.BatchNorm1d(32)          # 第 2 層批次正規化
        self.dropout2 = nn.Dropout(0.0)        # 第 2 層 Dropout，隨機捨棄 10% 神經元
        self.fc3 = nn.Linear(32, 16)           # 第 3 個全連接層：32 維降至 16 維
        self.bn3 = nn.BatchNorm1d(16)          # 第 3 層批次正規化
        self.fc4 = nn.Linear(16, output_dim)   ## 輸出層：將 16 維縮減為 1 維數值（即回歸預測值）

    def forward(self, x):                      # 定義前向傳播（Forward Pass）流程
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))  # 線性轉換 -> 批次標準化 -> ReLU 激活函數 -> 20% Dropout
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))  # 線性轉換 -> 批次標準化 -> ReLU 激活函數 -> 10% Dropout
        x = F.relu(self.bn3(self.fc3(x)))                 # 線性轉換 -> 批次標準化 -> ReLU 激活函數（此層無 Dropout）
        return self.fc4(x)                                # 輸出層直接線性輸出預測數值（回歸問題不加激活函數）


#%% ========== 1. 切分 train/test -> 分別 MICE 補值 ==========
# 原則和 scaler、QuantileTransformer 一致：補值模型只用訓練集 fit，
# 測試集只能「套用」已經學好的補值模型，不能參與 fit，避免資訊滲漏。
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (必須先 import 才能用)
from sklearn.impute import IterativeImputer

N_IMPUTATIONS = 4     # 可改：多重插補組數
TEST_SIZE = 0.2       # 可改：測試集比例

data = pd.read_excel(BASE_DIR / "2.all.xlsx")       # 檔名前面不要多一個「.」

# 前 6 欄為各個污染物變項，我們目前是一個污染物一個污染物分析，不補值；第 6 欄之後才是要用來分析的數值欄位
meta_cols = data.columns[:6]
feat_names = data.columns[6:]
data[feat_names] = data[feat_names].apply(pd.to_numeric, errors="coerce")  # 非數字 -> NaN

# 超出 float64 範圍（讀進來會變成 inf）的值改成 NaN
arr = data[feat_names].to_numpy(dtype=float)
exceed_mask = np.isinf(arr)
if exceed_mask.any():
    print("發現超出 float64 範圍的數值（最多顯示 20 筆）：")
    for r, c in np.argwhere(exceed_mask)[:20]:
        print(f"  列 {r}, 欄 '{feat_names[c]}' = {arr[r, c]}")
    data[feat_names] = data[feat_names].mask(exceed_mask)
else:
    print("沒有超出 float64 範圍的數值")

# ---- 先切分：原始資料（還帶著缺失值）就先切開，測試集不參與任何補值模型的訓練 ----
# 若資料是時間序列，請改成 shuffle=False（用時間順序切）
data_train_raw, data_test_raw = train_test_split(
    data, test_size=TEST_SIZE, random_state=SEED, shuffle=True
)

all_nan_cols = [c for c in feat_names if data_train_raw[c].isna().all()]
if all_nan_cols:
    print("警告：以下欄位在訓練集中全部是缺失值，無法補值（會被填 0）：", all_nan_cols)

# ---- MICE：每次用不同亂數種子、從後驗分布抽樣補值，共 N 組最後取平均 ----
# 補值模型（imp）只用訓練集 fit；測試集用「同一個已 fit 好」的 imp 做 transform，不重新 fit
train_imputed_list = []
test_imputed_list = []
imputers = []
for i in range(N_IMPUTATIONS):
    imp = IterativeImputer(
        max_iter=10,
        sample_posterior=True,        # 每組補值結果不同（多重插補）
        keep_empty_features=True,     # 全空欄位不要被丟掉，避免欄數對不上
        random_state=SEED + i,
    )
    imputers.append(imp)
    train_imputed_list.append(imp.fit_transform(data_train_raw[feat_names].to_numpy(dtype=float)))
    test_imputed_list.append(imp.transform(data_test_raw[feat_names].to_numpy(dtype=float)))  # 只 transform
    print(f"第 {i + 1}/{N_IMPUTATIONS} 組補值完成")

train_avg = np.mean(train_imputed_list, axis=0)
test_avg = np.mean(test_imputed_list, axis=0)

# 儲存已在訓練集 fit 完成的 4 組 MICE imputer
# 預測未知資料時只能使用 transform()，不能重新 fit
joblib.dump(
    {
        "imputers": imputers,
        "feature_cols": list(feat_names),
    },
    MODEL_DIR / "mice_imputers.pkl"
)
print(f"MICE imputer 已儲存：{MODEL_DIR / 'mice_imputers.pkl'}")

# 整欄覆蓋（轉成 float），避免 Month 這類整數欄位被塞入小數而報錯
train_df = data_train_raw.reset_index(drop=True).copy()
test_df = data_test_raw.reset_index(drop=True).copy()
train_df[feat_names] = pd.DataFrame(train_avg, columns=feat_names)
test_df[feat_names] = pd.DataFrame(test_avg, columns=feat_names)

train_df.to_csv(OUT_DIR / "MICE_train.csv", index=False)
test_df.to_csv(OUT_DIR / "MICE_test.csv", index=False)
print(f"切分後：訓練集 {len(train_df)} 筆，測試集 {len(test_df)} 筆")
print("訓練集補值後缺失比例：", train_df[feat_names].isna().values.mean())
print("測試集補值後缺失比例：", test_df[feat_names].isna().values.mean())

# 看目標 TARGET 有無缺值
before_tr, before_te = len(train_df), len(test_df)
train_df = train_df.dropna(subset=[TARGET]).reset_index(drop=True)
test_df = test_df.dropna(subset=[TARGET]).reset_index(drop=True)
print(f"目標「{TARGET}」缺失：訓練集移除 {before_tr - len(train_df)} 筆，"
      f"測試集移除 {before_te - len(test_df)} 筆")

#%% ========== 2. 切驗證集 -> 擴增 -> 標準化 ==========
# train_df / test_df 已經在第 1 區切好、補完值，這裡直接延用

feature_cols = [c for c in train_df.columns if c not in TARGET_COLS]
missing = [c for c in feature_cols + TARGET_COLS if c not in test_df.columns]
assert not missing, f"測試集缺少欄位：{missing}"

# 先切出驗證集，「再」擴增；否則同一筆資料的擴增複本會同時出現在訓練集與驗證集
# 若資料是時間序列，請改成 shuffle=False（用時間順序切）
train_part, val_part = train_test_split(
    train_df, test_size=0.15, random_state=SEED, shuffle=True
)


def selective_augment(df,
                      feature_cols, n_aug=9,            # n_aug 數字可改
                      noise_level=0.01,            # noise_level 數字可改，建議範圍 0.005 ~ 0.03
                      max_feat=9, seed=SEED):
    """對每筆資料隨機挑 1~max_feat 個特徵加雜訊，目標欄位不變。
    雜訊大小 = noise_level * 該特徵的標準差（不同單位的特徵才公平）。
    保留原始資料，回傳 原始 + n_aug 份擴增資料。"""
    rng = np.random.default_rng(seed)
    X = df[feature_cols].to_numpy(dtype=float)
    n, p = X.shape
    std = np.nanstd(X, axis=0)
    std = np.where(std == 0, 1e-6, std)                # 避免常數特徵標準差為 0 造成無效運算
    k_max = min(max_feat, p)

    parts = [df.reset_index(drop=True)]
    for _ in range(n_aug):
        k = rng.integers(1, k_max + 1, size=n)                         # 每列要加噪的特徵數
        rank = rng.random((n, p)).argsort(axis=1).argsort(axis=1)      # 每列的隨機排名
        mask = rank < k[:, None]                                       # 排名前 k 名的特徵加噪
        noise = rng.normal(0.0, 1.0, size=(n, p)) * noise_level * std
        new = df.reset_index(drop=True).copy()
        new[feature_cols] = X + noise * mask
        parts.append(new)
    return pd.concat(parts, axis=0, ignore_index=True)


train_aug = selective_augment(train_part, feature_cols, n_aug=9, noise_level=0.005)

# 只用（擴增後的）訓練集 fit 標準化，驗證/測試只 transform
scaler = StandardScaler().fit(train_aug[feature_cols])


def build_output(df):
    df = df.reset_index(drop=True)
    feats = pd.DataFrame(scaler.transform(df[feature_cols]), columns=feature_cols)
    return pd.concat([df[TARGET_COLS], feats], axis=1)


build_output(train_aug).to_csv(MODELING_DIR / "train_aug.csv", index=False)  # 已擴增
build_output(val_part).to_csv(MODELING_DIR / "val.csv", index=False)        # 不擴增
build_output(test_df).to_csv(MODELING_DIR / "test.csv", index=False)        # 不擴增

# 存起來，預測時直接載入，不必重新 fit
joblib.dump({"scaler": scaler, "feature_cols": feature_cols}, MODEL_DIR / "std_scaler.pkl")
print("訓練:", len(train_aug), "驗證:", len(val_part), "測試:", len(test_df))

#%% ========== 3. MLP 建模（目標由 TARGET 決定）==========
# 使用 PyTorch 計算 R-squared（決定係數）指標。
def r2_torch(y_pred, y_true):
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)  # 總離均差平方和
    ss_res = torch.sum((y_true - y_pred) ** 2)          # 殘差平方和
    return 1 - ss_res / (ss_tot + 1e-8)                 # 1 - 殘差比率（加 1e-8 防止除以零）

# 使用 PyTorch 計算 MAPE（平均絕對百分比誤差，%）。
def mape_torch(y_pred, y_true, eps=1e-3):
    return torch.mean(torch.abs((y_true - y_pred) / y_true.abs().clamp(min=eps))) * 100  # clamp 防止真值為 0


def metrics(y_pred, y_true):
    return {
        "loss": F.mse_loss(y_pred, y_true).item(),
        "mae": torch.mean(torch.abs(y_pred - y_true)).item(),
        "mape": mape_torch(y_pred, y_true).item(),
        "r2": r2_torch(y_pred, y_true).item(),
    }


tr_df = pd.read_csv(MODELING_DIR / "train_aug.csv")
va_df = pd.read_csv(MODELING_DIR / "val.csv")
te_df = pd.read_csv(MODELING_DIR / "test.csv")
feat_cols = [c for c in tr_df.columns if c not in TARGET_COLS]   # 用欄位名稱，不靠位置
feature_names = feat_cols

#----原數值------
#def get_xy(df):
    #return df[feat_cols].to_numpy(dtype=float), df[TARGET].to_numpy(dtype=float)

#----將數值取log------
def get_xy(df):
    y = df[TARGET].to_numpy(dtype=float)
    # 取 log 壓縮極端值，加 1e-4 避免 log(0)
    y_log = np.log(np.clip(y, 1e-4, None))
    return df[feat_cols].to_numpy(dtype=float), y_log

X_tr, y_tr = get_xy(tr_df)
X_va, y_va = get_xy(va_df)
X_te, y_te = get_xy(te_df)

# QuantileTransformer 只用訓練集 fit，並存檔給預測用
qt = QuantileTransformer(n_quantiles=min(1000, len(X_tr)), random_state=SEED)
X_tr_s = qt.fit_transform(X_tr)
X_va_s = qt.transform(X_va)
X_te_s = qt.transform(X_te)
joblib.dump(qt, MODEL_DIR / f"{TARGET}_quantile.pkl")


def to_x(a):
    return torch.from_numpy(np.ascontiguousarray(a).copy()).float()


def to_y(a):
    return torch.from_numpy(np.ascontiguousarray(a).copy()).float().view(-1, 1)


X_tr_t, y_tr_t = to_x(X_tr_s), to_y(y_tr)
X_va_t, y_va_t = to_x(X_va_s), to_y(y_va)
X_te_t, y_te_t = to_x(X_te_s), to_y(y_te)

# drop_last=True：避免最後一批只有 1 筆，BatchNorm 會報錯
train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=64,
                          shuffle=True, drop_last=True)

model = MLPRegressor(input_dim=X_tr_t.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)  # 可改：lr 學習率（建議 1e-4 ~ 5e-3）、weight_decay（建議 1e-4 ~ 1e-1）
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=30, min_lr=1e-6)       # 可改：factor（建議 0.1 ~ 0.5）、patience（建議 15 ~ 35）

EPOCHS = 500           # 可改，可以設大於 20 的任何正整數
PATIENCE = 60          # 可改，可設大於 10 的正整數（需大於 scheduler 的 patience）
best_val_loss = float("inf")
no_improve = 0
best_model_path = MODEL_DIR / f"{TARGET}_best_mlp_model.pth"
perf = {f"{s}_{m}": [] for s in ("train", "val") for m in ("loss", "mae", "mape", "r2")}

for epoch in range(EPOCHS):
    model.train()
    preds, trues = [], []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        preds.append(out.detach())
        trues.append(yb)
    tr = metrics(torch.cat(preds), torch.cat(trues))   # 整個 epoch 一起算，不是逐批平均

    model.eval()
    with torch.no_grad():
        va = metrics(model(X_va_t), y_va_t)

    scheduler.step(va["loss"])                         # 用驗證集，不用測試集

    for k, v in tr.items():
        perf[f"train_{k}"].append(v)
    for k, v in va.items():
        perf[f"val_{k}"].append(v)                     # 先記錄再判斷早停，最後一輪不會漏

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss {tr['loss']:.4f}/{va['loss']:.4f} | "
          f"MAE {tr['mae']:.3f}/{va['mae']:.3f} | MAPE {tr['mape']:.2f}/{va['mape']:.2f} | "
          f"R2 {tr['r2']:.4f}/{va['r2']:.4f}")

    if va["loss"] < best_val_loss - 1e-6:
        best_val_loss = va["loss"]
        torch.save(model.state_dict(), best_model_path)
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\n[Early Stopping] 驗證 Loss 已 {PATIENCE} 個 epoch 沒有改善，停止訓練。")
            break

#%% ========== 4. 載入最佳模型 -> 最終評估 -> 繪圖 ==========
# 載入最佳模型，測試集只在這裡評估一次
model.load_state_dict(torch.load(best_model_path, weights_only=True, map_location="cpu"))
model.eval()
with torch.no_grad():
    final = {
        "Train": metrics(model(X_tr_t), y_tr_t),
        "Val": metrics(model(X_va_t), y_va_t),
        "Test": metrics(test_preds := model(X_te_t), y_te_t),
    }
for name, m in final.items():
    print(f"Best model {name:5s} | R2 {m['r2']:.4f} | MAE {m['mae']:.3f} | "
          f"RMSE {m['loss'] ** 0.5:.3f} | MAPE {m['mape']:.2f}%")
test_r2 = final["Test"]["r2"]

# ---- 訓練曲線 ----
sns.set_theme(style="whitegrid", font_scale=1.8, palette="cool")
x_axis = range(1, len(perf["train_loss"]) + 1)
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
for ax, (key, ylabel, title) in zip(
        axes.ravel(),
        [("loss", "MSE Loss", "Loss"), ("mae", "MAE", "MAE"),
         ("mape", "MAPE (%)", "MAPE"), ("r2", "R$^2$", "R$^2$")]):
    ax.plot(x_axis, perf[f"train_{key}"], label="Train", marker="o", markersize=3)
    ax.plot(x_axis, perf[f"val_{key}"], label="Validation", marker="o", markersize=3)
    ax.set_xlabel("Epoch", weight="bold")
    ax.set_ylabel(ylabel, weight="bold")
    ax.set_title(f"{LABEL} {title} Curve", weight="bold", fontsize=24)
    ax.legend()
    ax.grid(True, linestyle="--")
plt.tight_layout()
plt.savefig(TARGET_DIR / f"{TARGET}_MLP_loss_curve.png")
plt.show()

# ---- 測試集 預測 vs 真實 ----
y_true_np = y_te_t.numpy().ravel()
y_pred_np = test_preds.numpy().ravel()
lims = [min(y_true_np.min(), y_pred_np.min()), max(y_true_np.max(), y_pred_np.max())]
plt.figure(figsize=(8, 8))
plt.scatter(y_pred_np, y_true_np, alpha=0.6)
plt.plot(lims, lims, "k-", label=rf"R$^2$={test_r2:.2f}")
plt.xlabel("Predicted Values", weight="bold", fontsize=20)
plt.ylabel("True Values", weight="bold", fontsize=20)
plt.title(f"{LABEL} Results", weight="bold", fontsize=25)
plt.legend(loc="lower right")
plt.savefig(TARGET_DIR / f"{TARGET}_scatter.png", dpi=600, bbox_inches="tight")
plt.show()

# ---- SHAP：分別解釋「訓練集」與「測試集」，比較是否過擬合 ----
# 背景值依官方建議，從訓練集（擴增後）隨機抽 1000 筆，不要整份丟入
# 解釋訓練集時，只用「未擴增的原始訓練資料」，不解釋人工加噪的複本
# 解釋測試集時，維持全部測試集
model = model.cpu().eval()
rng = np.random.default_rng(SEED)

bg_idx = rng.choice(len(X_tr_s), size=min(1000, len(X_tr_s)), replace=False)
background = to_x(X_tr_s[bg_idx])

n_orig_train = len(train_part)              # 原始訓練筆數
train_explain = to_x(X_tr_s[:n_orig_train])  # 只解釋真實觀測，不解釋加噪複本
test_explain = to_x(X_te_s)                  # 全部測試集

explainer = shap.DeepExplainer(model, background)


def compute_shap_summary(samples, name):
    """計算 SHAP 值、整理成特徵重要性表，並存檔。"""
    values = explainer.shap_values(samples, check_additivity=False)
    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values).reshape(len(samples), -1)
    assert values.shape[1] == len(feature_names), f"{name}：SHAP 特徵數與欄位數不符"

    mean_abs = np.abs(values).mean(axis=0)
    summary = (pd.DataFrame({"feature": feature_names, "mean_abs_shap_value": mean_abs})
               .sort_values("mean_abs_shap_value", ascending=False)
               .reset_index(drop=True))
    summary.to_csv(TARGET_DIR / f"{TARGET}_shap_{name}.csv", index=False)
    print(f"\n=== {name} 的 SHAP 特徵重要性（前 10）===")
    print(summary.head(10))
    return values, summary


def plot_shap_bar(summary, name):
    plt.figure(figsize=(12, 12))
    plt.barh(summary["feature"], summary["mean_abs_shap_value"],
             color="red", height=0.5)
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.title(f"{LABEL} Feature Importance ({name})", fontsize=20, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=11)
    plt.tight_layout()
    plt.savefig(TARGET_DIR / f"{TARGET}_shap_{name}.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_shap_summary(values, samples, name):
    shap.summary_plot(values, samples.numpy(), feature_names=feature_names, show=False)
    plt.gcf().set_size_inches(10, 10)
    plt.xlabel(f"{LABEL} SHAP value (impact on model output)", fontsize=20)
    plt.title(name, fontsize=16)
    plt.tight_layout()
    plt.savefig(TARGET_DIR / f"{TARGET}_SHAP_summary_{name}.png", dpi=300, bbox_inches="tight")
    plt.show()


print("計算訓練集 SHAP（檢查模型學到的決策行為）...")
shap_values_train, shap_summary_train = compute_shap_summary(train_explain, "train")
plot_shap_bar(shap_summary_train, "train")
plot_shap_summary(shap_values_train, train_explain, "train")

print("計算測試集 SHAP（檢查模型對未知資料的預測依據）...")
shap_values_test, shap_summary_test = compute_shap_summary(test_explain, "test")
plot_shap_bar(shap_summary_test, "test")
plot_shap_summary(shap_values_test, test_explain, "test")

# ---- 比較兩者排序是否一致，作為過擬合的輔助判斷 ----
compare_df = (shap_summary_train.rename(columns={"mean_abs_shap_value": "train_shap"})
             .merge(shap_summary_test.rename(columns={"mean_abs_shap_value": "test_shap"}),
                    on="feature"))
compare_df["train_rank"] = compare_df["train_shap"].rank(ascending=False)
compare_df["test_rank"] = compare_df["test_shap"].rank(ascending=False)
compare_df["rank_diff"] = (compare_df["train_rank"] - compare_df["test_rank"]).abs()
compare_df = compare_df.sort_values("rank_diff", ascending=False).reset_index(drop=True)
compare_df.to_csv(TARGET_DIR / f"{TARGET}_shap_train_vs_test.csv", index=False)

rho, p_value = spearmanr(compare_df["train_shap"], compare_df["test_shap"])
print(f"\n訓練集 vs 測試集 SHAP 排序的 Spearman 相關係數：{rho:.3f}（p={p_value:.4f}）")
print("排名差距最大的前 5 個特徵：")
print(compare_df.head(5)[["feature", "train_rank", "test_rank", "rank_diff"]])

plt.figure(figsize=(8, 8))
plt.scatter(compare_df["train_rank"], compare_df["test_rank"], alpha=0.7)
max_rank = len(compare_df)
plt.plot([1, max_rank], [1, max_rank], "k--", alpha=0.5, label="Identical Rank")
for _, row in compare_df.head(5).iterrows():
    plt.annotate(row["feature"], (row["train_rank"], row["test_rank"]), fontsize=9)
plt.xlabel("Train Rank", fontsize=14)
plt.ylabel("Test Rank", fontsize=14)
plt.title(f"{LABEL} SHAP Feature Rank: Train vs Test", fontsize=16, fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig(TARGET_DIR / f"{TARGET}_shap_rank_comparison.png", dpi=300, bbox_inches="tight")
plt.show()