# DeltaChineseQA

****** 改寫 https://github.com/HKUST-KnowComp/R-Net 將他移值到delta中文qa的data上 ******

## 運行環境
tensorflow 1.4.1(須可以執行cudnnGRU)
json 2.0.9, jieba 0.39(預設字典)
numpy 1.14.0
  
## 用法

目錄下有4個bash檔

```
./download.sh 
```
最先執行 會下載我們訓練好的最佳模型與w2v模型

```
./final_test.sh <<test.json path>> <<result.csv path>>
```
輸出最佳結果 
```
./final_train.sh <<train.json path>>
```
訓練新的模型 模型每隔500個iterations會出現在./model下
```
./custom_test.sh <<test.json path>> <<result.csv path>> <<model path>>
```
用新訓練的model進行預測 model的路徑會是 ./model/model_xxxx.ckpt
