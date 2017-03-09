---
layout: post
title: SocialCloudComputing
---
#### 課程簡介
* Centrality Analysis  
* Community Detection  
* Link Prediction  
* Label Prediction
* Influence maximization
* Outbreak Detection  
消息擴散路徑  
* Role/Postion Analysis
* Social Relation Extraction
* Cloud Computing
<!--more-->
 
#### Journals
* Nature
* Science
* Physical Review
* Social Networks
* ACM Transactions on Knowledge Discovery from Data (TKDD)
* ACM Transactions on Intelligent Systems and Technology(TIST)
* ACM Transactions on Social Computing(TSC)
* IEEE Transactions on Knowledge and Data Engineering(TKDE)
* IEEE Transactions on Computational Social System

#### Social Networks

|Sociocentric|Egocentric|
|根據整群分析|根據個人分析,向外延伸|

* information Network  
  paper reference  
  web hyperlink  
  Language  
* Social Network  
  FB好友關係    
* Technology Network  
  電力系統(Power grid)   
* Biologycal Network  
  蛋白質互動關係,食物鏈    

為什麼要分這麼多類Network?  
> 因為要分析的點不同,可能在information Network中很重要的,卻在Social Network可能不是那麼重要  

**** Network Properties
1. small-world effect
    六度分離理論  
    靠點和點距離關係分析    
2. Transitivity
    朋友的朋友很可能也是你朋友  
	[Clustering Coeffieient](https://zh.wikipedia.org/wiki/%E9%9B%86%E8%81%9A%E7%B3%BB%E6%95%B0)  
3. Degree distribution
	Real world network: Power law		
	P<sub>k</sub> = CK<sup>-&alpha;</sup>  
	Heavy-tailed degree distribution  
	大量很低的數量,集合起來還是很驚人  
4. Network resilience
	如果拿掉一些點/邊,連通性會有什麼變化？(e.g.有些人掛了,離職)  
	連接path的長度變長,或是disconnect   
	廣告投放要投在哪個點影響力最大,如果是傳染病隔離哪個點最有效?  
5. Mixing patterns  
	探討兩邊節點的type,可能因為什麼關係成為朋友(職業/興趣/文化)
6. Degree Correlations
	觀察兩邊點的degree  
	內向和外向人(朋友多,degree高)觀察  
7. Community Structure 
	一群點邊的密度很高,稱作一個community    
	clique 判斷是否認兩個點是否都有邊相連(clique problem 分團問題)  
	clique problem 是 NP-Complete  
	
	Connected commponets :有連通的子圖  
8.  



CERN
[米爾格倫實驗 Milgram experiment](https://zh.wikipedia.org/wiki/%E7%B1%B3%E7%88%BE%E6%A0%BC%E5%80%AB%E5%AF%A6%E9%A9%97)服從威權實驗   
random graph  








#### reference
[社群雲端運算](http://newdoc.nccu.edu.tw/teaschm/1052/schmPrv.jsp-yy=105&smt=2&num=753868&gop=00&s=1&tea=101583.htm)
