# DeltaChineseQA

****** 改寫 https://github.com/HKUST-KnowComp/R-Net 將他移值到delta中文qa的data上 ****** <br />
基本上用微軟亞洲研究院的r_net架構並在最後的pointer net 運行機制的地方進行修改

## kaggle site
https://www.kaggle.com/c/ml-2017fall-final-chinese-qa/leaderboard <br />
比賽的基本概念為給定一段文字，並給定幾個問題，從文字裡面挑選出適當解答的位置，並利用f1 score作為衡量的分數<br />

裡面包含一個pretrain的model，在kaggle上面達到 f1 score : 0.65<br />

## 運行環境
tensorflow 1.4.1(須可以執行cudnnGRU)<br />
json 2.0.9, jieba 0.39(預設字典)<br />
numpy 1.14.0<br />
  
## 用法

目錄下有4個bash檔

```
./download.sh 
```
最先執行 會下載訓練好的最佳模型與dataset

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
