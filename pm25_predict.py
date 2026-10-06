# -*- coding: utf-8 -*-
# 未知資料預測：
# 原始資料（含缺值） -> 套用訓練時的 MICE -> StandardScaler
# -> QuantileTransformer -> 已訓練好的 MLP -> 污染物預測值
#
# 注意：
# 這支程式不重新 fit MICE、Scaler、QuantileTransformer，也不重新訓練 MLP。
# 所有參數都來自 pm25_model_training.py 產生的 output/models。

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 0. 設定
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# 資料夾設定
# ============================================================

# 原始預測資料
INPUT_DIR = BASE_DIR / "2019_data"

# 所有輸出
OUT_DIR = BASE_DIR / "output"

# 建模產生的模型與分析資料
MODELING_DIR = OUT_DIR / "output_modeling"
MODEL_DIR = MODELING_DIR / "models"

# 預測產生的資料
PREDICT_DIR = OUT_DIR / "output_pre"

# ============================================================
# 本次要預測的檔案
# ============================================================

INPUT_FILE = INPUT_DIR / "2019_02.xlsx"

TARGET = "PM2.5"

# 確保輸出資料夾存在
MODELING_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PREDICT_DIR.mkdir(parents=True, exist_ok=True)
# ============================================================
# 1. MLP 模型架構
#    必須與建模程式完全一致
# ============================================================

class MLPRegressor(nn.Module):

    def __init__(self, input_dim, output_dim=1):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)

        self.fc4 = nn.Linear(16, output_dim)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)


# ============================================================
# 2. 檢查模型檔案
# ============================================================

mice_path = MODEL_DIR / "mice_imputers.pkl"
scaler_path = MODEL_DIR / "std_scaler.pkl"
quantile_path = MODEL_DIR / f"{TARGET}_quantile.pkl"
model_path = MODEL_DIR / f"{TARGET}_best_mlp_model.pth"

for path in [mice_path, scaler_path, quantile_path, model_path]:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到必要的模型檔案：{path}\n"
            "請先執行 model_training.py 建立模型。"
        )


# ============================================================
# 3. 讀取訓練時保存的 MICE、Scaler、QuantileTransformer
# ============================================================

mice_bundle = joblib.load(mice_path)
imputers = mice_bundle["imputers"]
mice_feature_cols = mice_bundle["feature_cols"]

scaler_bundle = joblib.load(scaler_path)
scaler = scaler_bundle["scaler"]
train_feature_cols = scaler_bundle["feature_cols"]

qt = joblib.load(quantile_path)

# 確認 MICE 與模型的特徵欄位一致
if list(mice_feature_cols) != list(train_feature_cols):
    raise ValueError(
        "MICE 與模型的特徵欄位不一致，請重新執行建模程式。"
    )


# ============================================================
# 4. 讀取「尚未 MICE」的未知資料
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"找不到預測輸入檔案：{INPUT_FILE}")

suffix = INPUT_FILE.suffix.lower()

if suffix in [".xlsx", ".xls"]:
    new_df = pd.read_excel(INPUT_FILE)
elif suffix == ".csv":
    new_df = pd.read_csv(INPUT_FILE)
else:
    raise ValueError("目前只支援 .xlsx、.xls 或 .csv")

print(f"讀取未知資料：{INPUT_FILE}")
print(f"資料筆數：{len(new_df)}")


# ============================================================
# 5. 依照訓練時的特徵名稱抓取資料
# ============================================================
#
# 不使用 iloc[:, 3:]。
# 直接依「訓練模型真正使用的欄位名稱」對齊。
#
# 這樣即使原始資料的欄位順序不同，也不會把特徵放錯。

missing_features = [
    c for c in train_feature_cols
    if c not in new_df.columns
]

if missing_features:
    raise ValueError(
        "未知資料缺少訓練模型需要的特徵欄位：\n"
        + "\n".join(missing_features)
    )

# 先依照訓練時的欄位順序取出特徵
new_feat = new_df[train_feature_cols].copy()

# 轉成數值，非數值轉為 NaN
new_feat = new_feat.apply(
    pd.to_numeric,
    errors="coerce"
)

# MICE 可以處理 NaN，但不能直接處理 inf
arr = new_feat.to_numpy(dtype=float)
inf_mask = np.isinf(arr)

if inf_mask.any():

    print(
        f"發現 {inf_mask.sum()} 個 inf / -inf，"
        "將其轉為 NaN 後交給 MICE 補值："
    )

    for r, c in np.argwhere(inf_mask)[:20]:
        print(
            f"  列 {r}, 欄 '{new_feat.columns[c]}' "
            f"= {arr[r, c]}"
        )

    new_feat = new_feat.mask(inf_mask)

else:
    print("沒有發現 inf / -inf")

print(f"模型特徵數：{len(train_feature_cols)}")
print(
    f"MICE 前缺失值數量："
    f"{new_feat.isna().sum().sum()}"
)
# ============================================================
# 6. 套用「訓練階段已 fit 好」的 MICE
# ============================================================
#
# 訓練時：
#   Train data
#      ↓
#   MICE #1 fit
#   MICE #2 fit
#   MICE #3 fit
#   MICE #4 fit
#
# 預測時：
#   新資料
#      ↓
#   分別 transform 4 次
#      ↓
#   4 組結果取平均
#
# 絕對不能在未知資料上重新 fit MICE。

new_imputed_list = []

for i, imp in enumerate(imputers, start=1):
    imputed = imp.transform(
        new_feat.to_numpy(dtype=float)
    )
    new_imputed_list.append(imputed)

    print(f"第 {i}/{len(imputers)} 組 MICE transform 完成")

new_avg = np.mean(new_imputed_list, axis=0)

mice_df = new_df.copy()
mice_df[train_feature_cols] = pd.DataFrame(
    new_avg,
    columns=train_feature_cols,
    index=mice_df.index
)

print(
    "MICE 後缺失比例：",
    mice_df[train_feature_cols].isna().to_numpy().mean()
)


# ============================================================
# 7. 保存 MICE 後的資料
# ============================================================
#
# 例如：
#   2019_02.xlsx
#       ↓
#   2019_02_MICE.csv


mice_output_name = f"{INPUT_FILE.stem}_MICE.csv"
mice_output_path = PREDICT_DIR / mice_output_name

mice_df.to_csv(
    mice_output_path,
    index=False
)

print(f"MICE 後資料已保存：{mice_output_path}")


# ============================================================
# 8. StandardScaler -> QuantileTransformer
# ============================================================

X_scaled = scaler.transform(
    mice_df[train_feature_cols]
)

X_quantile = qt.transform(
    X_scaled
)

X_new = torch.from_numpy(X_quantile).float()


# ============================================================
# 9. 載入最佳 MLP
# ============================================================

model = MLPRegressor(
    input_dim=len(train_feature_cols)
)

# === 修改後 ===
model.load_state_dict(
    torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )
)

model.eval()


# ============================================================
# 10. 預測污染物濃度
# ============================================================

with torch.no_grad():
    pred = model(X_new).numpy().ravel()

# PM2.5 濃度不允許為負值
pred = np.clip(pred, 0, None)


# ============================================================
# 11. 輸出「原始資料 + 預測值」
# ============================================================
#
# 直接保留原始資料中的座標與其他欄位，
# 再增加 Prediction。
#
# 因此若 2019_02.xlsx 本身已經有 tm2_e、tm2_n，
# 就不一定需要另外的 3.teds.csv。

result = new_df.copy()
result["Prediction"] = pred


# ============================================================
# 12. 保存預測結果
# ============================================================

pred_output_name = f"{INPUT_FILE.stem}_{TARGET}_pred.csv"
pred_output_path = PREDICT_DIR / pred_output_name

result.to_csv(
    pred_output_path,
    index=False
)

print()
print("========================================")
print("預測完成！")
print(f"MICE 資料：{mice_output_path}")
print(f"預測結果：{pred_output_path}")
print(f"預測筆數：{len(result)}")
print("========================================")
