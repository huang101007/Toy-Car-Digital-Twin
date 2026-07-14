# Toy Car Digital Twin Prediction System

## 專案簡介
這是我在機械工程概論課程中開發的「玩具車數位孿生」系統。本專案的目標是透過軟體開發與數據分析，解決實體車輛設計僅能憑藉窮舉與物理測量尋找最佳解的限制。透過建立預測模型與圖形化介面 (GUI)，系統能預先模擬並精準預測動力車在特定設計參數下的行駛距離與偏移量。

## 核心功能與技術
* 數據處理與模型訓練：針對齒輪比、齒輪數等 12 項特徵進行資料前處理，平行評估逾 10 種演算法 (包含 Lasso、Decision Tree、SGD、XGBoost 等)。
* 模型優化：針對訓練初期的過擬合問題，導入 5-fold Cross-Validation 評估模型泛化能力，並執行 Hyperparameter Tuning 與 Ensemble Learning。最終選用 Random Forest 預測直線距離 (R2 達 0.87)，XGBoost 預測側向偏移量。
* GUI 視覺化平台：獨立開發虛實整合介面，具備四大核心模組：動態參數設定面板、即時預測結果顯示、數據視覺化分析，以及軌跡動畫模擬。並加入邊界約束條件以過濾不合理的物理設計。

## 使用技術 (Tech Stack)
* 程式語言：Python
* 機器學習：Scikit-learn (Random Forest), XGBoost
* 資料處理：Pandas, NumPy
  
## 執行方式
1. 安裝必要的套件：`pip install pandas numpy xgboost scikit-learn`。
2. 在終端機執行 `python main.py` 即可開啟 GUI 介面進行參數設定與預測操作。
