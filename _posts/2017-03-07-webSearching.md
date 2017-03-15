---
layout: post
title: WebSearching 
---
__CH8 Evaluation in information retrival__    
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


__Precision (P)__    
>  Precision = $$\frac{檢索到相關物件的數量}{物件總數}\$$  

__Recall (R)__  
>  Recall = $$\frac{檢索到相關物件的數量}{相關物件總數}\$$  

||Relevant|Nonrelevent|
|Retrieved|true positive(tp)|false positive(fp)|
|Not retrieved|false negatives(fn)|true negatives(tn)|

true positive(tp) 機器判斷+且為真  
false positive(fp)機器判斷+但是是假  
false negatives(fn) 機器判斷-但判斷錯  
true negatives(tn)  機器判斷-且判斷對  

P = $$\frac{tp}{tp+fp}\$$  

R = $$\frac{tp}{tp+fn}\$$  

__Accuracy vs Precision__   
accuracy = $$\frac{tp+tn}{tp+fp+fn+tn}\$$  
> accuracy不適合用在information retrieval,
因為通常Nonrelevant會非常的大,tn項非常大除分母fn和tn都非常大結果會趨近於1,所以才用Precision和Recall作為依據

__調和平均數(harmonic mean)__    
> H = $$\frac{n}{\frac{1}{x_1}+\frac{1}{x_2}+..+\frac{1}{x_n}}\$$ 

Precision/Recall Tradoff
使用調和平均數計算Precision和Recall的Tradoff  
量測的數值稱做F measure,&alpha;和1-&alpha;分別為P和R的權重,一般是取&alpha;=0.5 
> F = $$\frac{1}{\alpha\frac{1}{P}+(1-\alpha)\frac{1}{R}} = \frac{(\beta^2+1)PR}{\beta^2P+R}\$$ where $$\ \beta^2=\frac{1-\alpha}{\alpha}\$$

#### Ranked Evalution
P,R,F都是unordered(沒有等級)  
一個query會有一個Precision/Recall圖(崎嶇的坡)    
使用內插法(interpolated)可以得到一張較平滑的P-R圖(和機器學習ROC curve相似)
P-R curve的面積越大效能越佳(代表Precision掉越慢)    

__內插法__  
> $$\ p_{interp}= \max\limits_{r'\ge r}\ p(r{'})\$$

r代表recall,由後面相對較高的值往前填平,Precision基本上會有往下掉的趨勢    

__Mean Average Precision(MAP)__  
> $$\ MAP(Q) = \frac{1}{|Q|}\sum^{|Q|}_{j=1}\frac{1}{m_j}\sum_{k=1}^{m_j}Precision(R_{jk})\$$

第一個sum算query平均  
第二個sum算precision平均  

__Precision at k__  

Normalized Discounted Cumulative Gain(NDCG)  
2002出現  
3高度相關  
0沒關係  
G = <3,2,3,0,0,1,2,2,3,0,...>
1.Cumulative Gain(CG)
2.Discounted Cumulative Gain(DCG)
累加前一項,做成遞增感覺  
3.Normalized Discounted Cumulative Gain(NDCG)
做一個衰退,並和理想值相除  
通常看NDCG at 10

benchmark 資料集  
1.
2. TREC(nist.gov)  
	Ad-hoc 資料集(1992-1999)  
3. GOV2
	2500萬篇文章  
4. NICIR
	cross-language IR
5. Cross Language Evaluation
6. REUTERS

標記資料準則  
Kappa measure   
標記資料是否一致  
Kappa計算公式  

Pooled marginals

A/B testing
一次有兩套演算法並存  
一部分流量導到新的演算法  
並觀察AB的差異,新的較好再導過去

Result Summaries
* 10 blue link
* Static/Dynamic
抽前50個字/利用nlp技術  

quicklinks  
底下多的鍊結  

Traditional Information retreval (ch6,ch11)  
模型
1. Vector Space model
	TF-IDF
2. Probabilistic Information Retrieval
	BM25

|Empirical IR | Model-based IR|
|暴力法|有理論模型| 
|heuristic|數學假設|
|難推廣到其他問題|容易推廣到其他問題| 

1960 第一個機率模型
1970 
vector space model 1975
classic probabilistic model
1980
non-class logic  
1990
TREC benchmark  
goolge成立1996
2000
learning to rank
Markov model


vector space.py 主程式











