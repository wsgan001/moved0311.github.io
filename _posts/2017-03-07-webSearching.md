---
layout: post
title: webSearching 
---
Ch8 Evaluation in information retrival
評量search engine好壞
1. 搜到的index
2. 搜尋速度
3. 二氧化碳排放量
4. 和搜尋相關程度

相關程度
1. benchmark資料集
2. benchmark queries(問句)
3. 文章是否相關的標記(ground truths)

queries和information need有落差
想找的東西,不會下key word

Precision (P)
  Precision = 相關/物件總數
Recall (R)
  Recall = 相關/相關物件總數

true positive(TP) 機器判斷+且為真
false positive(FP)機器判斷+但是是假
false negatives(FN) 機器判斷-但判斷錯
true negatives(TN)  機器判斷-且判斷對

Precision/Recall Tradoff

調和平均數(harmonic mean)(1/F = 1/2(1/P+1/R) )

Accuracy vs Precision
accuracy猜是和猜不是都要算

