# -*- coding: utf-8 -*-
#MICE 補值 -> 資料擴增/標準化 -> MLP 建模 -> SHAP -> 預測（修正版）
#資料夾結構（所有檔案都放在同一個資料夾 "MLP pychome test"）：
    #MLP_test/
    #pm25_pipeline_fixed.py
    #2.all.xlsx                 ← 要放的
    #3.teds.csv                 ← 要放的
    #output/                         ← 程式自動建立
       # MICE_train.csv
       # MICE_test.csv
       # train_aug.csv
       # val.csv
       # test_aug.csv
       # PM2.5_201901_pred.csv
       # models/
       #     std_scaler.pkl
       #     PM2.5_quantile.pkl
       #     PM2.5_best_mlp_model.pth
       # PM2.5/
       #     PM2.5_MLP_loss_curve.png
       #     PM2.5_scatter.png
       #     PM2.5_shap.csv
       #     PM2.5_shap.png
       #     Directionality_shap_PM2.5.png

#%% ========== 0. 共用設定（每次執行前先跑這一格）==========
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ---- 路徑：以「本檔案所在資料夾」為基準，不依賴工作目錄 ----
try:
    BASE_DIR = Path(__file__).resolve().parent #若是以 python 指令執行腳本，取得該腳本所在的母資料夾路徑
except NameError:          # 若在 Jupyter 或 Spyder 等互動式介面執行（沒有 __file__ 變數）
    BASE_DIR = Path.cwd() # 則退回使用目前的工作目錄（Current Working Directory）

PRED_DIR = BASE_DIR    # 設定預測輸入檔案所在的資料夾路徑（與主程式同層）                         # 預測用檔案也在同一個資料夾
OUT_DIR = BASE_DIR / "output" # 設定所有輸出檔案（圖表、預測 CSV）存放的根目錄
MODEL_DIR = OUT_DIR / "models" # 設定模型權重與 Scaler 存檔專用的子目錄
for d in (OUT_DIR, MODEL_DIR): # 逐一檢查輸出目錄與模型目錄
    d.mkdir(parents=True, exist_ok=True)  # 若資料夾不存在則自動建立，已存在則略過不報錯

SEED = 42 # 定義隨機種子，確保所有隨機抽樣、資料切分與訓練結果可再現
TARGET_COLS = ["PM2.5", "K+", "Na+", "NH4+", "NO3-", "SO42-"] # 資料集裡所有的預測目標候選欄位清單
TARGET = "PM2.5"      #本次要建模的目標欄位名稱，更換此處即可更換目標
LABEL = "PM$_{2.5}$" if TARGET == "PM2.5" else TARGET ## 圖表標籤名稱，若為 PM2.5 則轉為 LaTeX 下標格式
TARGET_DIR = OUT_DIR / TARGET # 為當前目標建立專屬輸出目錄（例如 output/PM2.5/）
TARGET_DIR.mkdir(parents=True, exist_ok=True) # 自動建立該目標專屬的儲存資料夾

np.random.seed(SEED) # 設定 NumPy 隨機種子
torch.manual_seed(SEED) # 設定 PyTorch 隨機種子，確保權重初始化一致


class MLPRegressor(nn.Module):# 繼承 PyTorch 的 nn.Module 類別
    """訓練與預測共用同一個定義，避免兩邊不一致。"""

    def __init__(self, input_dim, output_dim=1):
        super().__init__() # 呼叫父類別的初始化函式
        self.fc1 = nn.Linear(input_dim, 64) # 第 1 個全連接層：將輸入維度特徵映射到 64 個神經元
        self.bn1 = nn.BatchNorm1d(64) # 第 1 層批次正規化（Batch Normalization），穩定數值分佈加速收斂
        self.dropout1 = nn.Dropout(0.2) # 第 1 層 Dropout，訓練時隨機將 20% 神經元設為 0 以防止過擬合
        self.fc2 = nn.Linear(64, 32) # 第 2 個全連接層：64 維降至 32 維
        self.bn2 = nn.BatchNorm1d(32) # 第 2 層批次正規化
        self.dropout2 = nn.Dropout(0.1) # 第 2 層 Dropout，隨機捨棄 10% 神經元
        self.fc3 = nn.Linear(32, 16) # 第 3 個全連接層：32 維降至 16 維
        self.bn3 = nn.BatchNorm1d(16) # 第 3 層批次正規化
        self.fc4 = nn.Linear(16, output_dim)   ## 輸出層：將 16 維縮減為 1 維數值（即回歸預測值）

    def forward(self, x): # 定義前向傳播（Forward Pass）流程
        x = self.dropout1(F.relu(self.bn1(self.fc1(x)))) # 線性轉換 -> 批次標準化 -> ReLU 激活函數 -> 20% Dropout
        x = self.dropout2(F.relu(self.bn2(self.fc2(x)))) # 線性轉換 -> 批次標準化 -> ReLU 激活函數 -> 10% Dropout
        x = F.relu(self.bn3(self.fc3(x))) # 線性轉換 -> 批次標準化 -> ReLU 激活函數（此層無 Dropout）
        return self.fc4(x) # 輸出層直接線性輸出預測數值（回歸問題不加激活函數）


#%% ========== 1. 切分 train/test -> 分別 MICE 補值 ==========
# 原則和 scaler、QuantileTransformer 一致：補值模型只用訓練集 fit，
# 測試集只能「套用」已經學好的補值模型，不能參與 fit，避免資訊滲漏。
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (必須先 import 才能用)
from sklearn.impute import IterativeImputer

N_IMPUTATIONS = 4                                   # 可改：多重插補組數
TEST_SIZE = 0.2                                     # 可改：測試集比例

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
for i in range(N_IMPUTATIONS):
    imp = IterativeImputer(
        max_iter=10,
        sample_posterior=True,        # 每組補值結果不同（多重插補）
        keep_empty_features=True,     # 全空欄位不要被丟掉，避免欄數對不上
        random_state=SEED + i,
    )
    train_imputed_list.append(imp.fit_transform(data_train_raw[feat_names].to_numpy(dtype=float)))
    test_imputed_list.append(imp.transform(data_test_raw[feat_names].to_numpy(dtype=float)))  # 只 transform
    print(f"第 {i + 1}/{N_IMPUTATIONS} 組補值完成")

train_avg = np.mean(train_imputed_list, axis=0)
test_avg = np.mean(test_imputed_list, axis=0)

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

#看目標TARGET有無缺值
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


def selective_augment(df, feature_cols, n_aug=9, noise_level=0.01, max_feat=9, seed=SEED):
    """對每筆資料隨機挑 1~max_feat 個特徵加雜訊，目標欄位不變。
    雜訊大小 = noise_level * 該特徵的標準差（不同單位的特徵才公平）。
    保留原始資料，回傳 原始 + n_aug 份擴增資料。"""
    rng = np.random.default_rng(seed)
    X = df[feature_cols].to_numpy(dtype=float)
    n, p = X.shape
    std = np.nanstd(X, axis=0)
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


train_aug = selective_augment(train_part, feature_cols, n_aug=9)

# 只用（擴增後的）訓練集 fit 標準化，驗證/測試只 transform
scaler = StandardScaler().fit(train_aug[feature_cols])


def build_output(df):
    df = df.reset_index(drop=True)
    feats = pd.DataFrame(scaler.transform(df[feature_cols]), columns=feature_cols)
    return pd.concat([df[TARGET_COLS], feats], axis=1)


build_output(train_aug).to_csv(OUT_DIR / "train_aug.csv", index=False)
build_output(val_part).to_csv(OUT_DIR / "val.csv", index=False)      # 不擴增
build_output(test_df).to_csv(OUT_DIR / "test_aug.csv", index=False)  # 不擴增

# 存起來，預測時直接載入，不必重新 fit
joblib.dump({"scaler": scaler, "feature_cols": feature_cols}, MODEL_DIR / "std_scaler.pkl")
print("訓練:", len(train_aug), "驗證:", len(val_part), "測試:", len(test_df))

#%% ========== 3. MLP 建模（目標由 TARGET 決定）==========
#使用 PyTorch 計算 R-squared（決定係數）指標。
def r2_torch(y_pred, y_true):
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2) # 總離均差平方和
    ss_res = torch.sum((y_true - y_pred) ** 2) # 殘差平方和
    return 1 - ss_res / (ss_tot + 1e-8) # 1 - 殘差比率（加 1e-8 防止除以零）

#使用 PyTorch 計算 MAPE（平均絕對百分比誤差，%）。
def mape_torch(y_pred, y_true, eps=1e-3):
    return torch.mean(torch.abs((y_true - y_pred) / y_true.abs().clamp(min=eps))) * 100 # clamp 防止真值為 0


def metrics(y_pred, y_true):
    return {
        "loss": F.mse_loss(y_pred, y_true).item(),
        "mae": torch.mean(torch.abs(y_pred - y_true)).item(),
        "mape": mape_torch(y_pred, y_true).item(),
        "r2": r2_torch(y_pred, y_true).item(),
    }


tr_df = pd.read_csv(OUT_DIR / "train_aug.csv")
va_df = pd.read_csv(OUT_DIR / "val.csv")
te_df = pd.read_csv(OUT_DIR / "test_aug.csv")
feat_cols = [c for c in tr_df.columns if c not in TARGET_COLS]   # 用欄位名稱，不靠位置
feature_names = feat_cols


def get_xy(df):
    return df[feat_cols].to_numpy(dtype=float), df[TARGET].to_numpy(dtype=float)


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
    return torch.from_numpy(a).float()


def to_y(a):
    return torch.from_numpy(a).float().view(-1, 1)


X_tr_t, y_tr_t = to_x(X_tr_s), to_y(y_tr)
X_va_t, y_va_t = to_x(X_va_s), to_y(y_va)
X_te_t, y_te_t = to_x(X_te_s), to_y(y_te)

# drop_last=True：避免最後一批只有 1 筆，BatchNorm 會報錯
train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=64,
                          shuffle=True, drop_last=True)

model = MLPRegressor(input_dim=X_tr_t.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=30, min_lr=1e-6)  # 新版已無 verbose 參數

EPOCHS = 500
PATIENCE = 40
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

# ---- 載入最佳模型，測試集只在這裡評估一次 ----
model.load_state_dict(torch.load(best_model_path))
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

# ---- SHAP（使用與模型訓練相同尺度的資料，背景樣本隨機抽取）----
model = model.cpu().eval()
rng = np.random.default_rng(SEED)
background = to_x(X_tr_s)      # 全部訓練集都當背景
test_samples = to_x(X_te_s)    # 全部測試集都拿去解釋

explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(test_samples, check_additivity=False)
if isinstance(shap_values, list):
    shap_values = shap_values[0]
shap_values = np.asarray(shap_values).reshape(len(test_samples), -1)   # (樣本數, 特徵數)
assert shap_values.shape[1] == len(feature_names), "SHAP 特徵數與欄位數不符"

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_summary_df = (pd.DataFrame({"feature": feature_names,
                                 "mean_abs_shap_value": mean_abs_shap})
                   .sort_values("mean_abs_shap_value", ascending=False)
                   .reset_index(drop=True))
print(shap_summary_df)
shap_summary_df.to_csv(TARGET_DIR / f"{TARGET}_shap.csv", index=False)

plt.figure(figsize=(12, 12))
plt.barh(shap_summary_df["feature"], shap_summary_df["mean_abs_shap_value"],
         color="red", height=0.5)
plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")
plt.title(f"{LABEL} Feature Importance", fontsize=20, fontweight="bold")
plt.gca().invert_yaxis()
plt.yticks(fontsize=10)
plt.xticks(fontsize=11)
plt.tight_layout()
plt.savefig(TARGET_DIR / f"{TARGET}_shap.png", dpi=300, bbox_inches="tight")
plt.show()

# 方向性圖（論文不用，可略）：直接使用 shap 內建 colorbar
shap.summary_plot(shap_values, test_samples.numpy(),
                  feature_names=feature_names, show=False)
plt.gcf().set_size_inches(10, 10)
plt.xlabel(f"{LABEL} SHAP value (impact on model output)", fontsize=20)
plt.tight_layout()
plt.savefig(TARGET_DIR / f"Directionality_shap_{TARGET}.png", dpi=300)
plt.show()


#%% ========== 4. 用訓練好的模型預測新資料 ==========
# 需要先跑過第 0、2、3 格（或已存在 output/models 內的檔案）
bundle = joblib.load(MODEL_DIR / "std_scaler.pkl")
scaler = bundle["scaler"]
train_feature_cols = bundle["feature_cols"]
qt = joblib.load(MODEL_DIR / f"{TARGET}_quantile.pkl")

new_df = pd.read_csv(PRED_DIR / "2.2019_01_MICE.csv")
new_feat = new_df.iloc[:, 3:]                      # 前 3 欄不是特徵

# 用「欄位名稱」對齊，而不是硬把欄位改名（順序不同也不會錯位）
miss = [c for c in train_feature_cols if c not in new_feat.columns]
if miss:
    raise ValueError(f"預測資料缺少訓練時的特徵欄位：{miss}")
new_feat = new_feat[train_feature_cols]
if new_feat.isna().any().any():
    raise ValueError("預測資料仍有缺失值，請先補值")

# 與訓練完全相同的前處理：StandardScaler -> QuantileTransformer
X_new = to_x(qt.transform(scaler.transform(new_feat)))

model = MLPRegressor(input_dim=len(train_feature_cols))
model.load_state_dict(torch.load(MODEL_DIR / f"{TARGET}_best_mlp_model.pth",
                                 map_location="cpu"))
model.eval()
with torch.no_grad():
    pred = model(X_new).numpy().ravel()
pred = np.clip(pred, 0, None)                      # 濃度不會是負的

coords = pd.read_csv(PRED_DIR / "3.teds.csv")
assert len(coords) == len(pred), f"座標檔 {len(coords)} 列，預測 {len(pred)} 列，列數不符"

result = pd.concat([coords.reset_index(drop=True),
                    pd.DataFrame({"Prediction": pred})], axis=1)
pred_path = OUT_DIR / f"{TARGET}_201901_pred.csv"
result.to_csv(pred_path, index=False)
print("預測完成，結果已儲存：", pred_path)