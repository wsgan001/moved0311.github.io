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
8. Network motifs      
在音樂上motifs是一種作曲法,靈感的意思  
	在生物基因上是一些重複的pattern  
	在社群希望找到出現次數較高的motifs(最常出現的subgraph)  

CERN
[米爾格倫實驗 Milgram experiment](https://zh.wikipedia.org/wiki/%E7%B1%B3%E7%88%BE%E6%A0%BC%E5%80%AB%E5%AF%A6%E9%A9%97)服從威權實驗   
random graph  

__Central of Network__  
* 找到最重要的點(central)  

local  
> 1. Degree  








global  
> 2. Closeness  
3. Betweeness  
4. Eigenvector  
  
* Group Centrality 一群最有影響力的人  
在小世界理論中,如果送信到目標的前一步,都是經由特定的3個人,代表這三個人很重要,  
目前social network還無法透過社群網站判斷這些人  

__Social actors(群眾的智慧)__  
1. Connectors  
認識很多人,很擅長社交  
2. Mavens  
資訊專家,知道很多各式訊息  
3. Salesman  
容易說服別人,擅長協調  

判斷social network的四種centrality  
1. Degree centrality(local)  
點的重要性,若network的規模大小不同,做normalize(除總size-1)  
2. Betweeness Centrality  
Node<sub>i</sub> A到B的shortest path有幾條經過i  
3. Closeness Centrality
點i和所有點j的shortest path平均的距離  
4. Eigenvector Centrality    
這個點的重要性,看他朋友點的重要性  
eigenvector  
> 一個向量乘上一個矩陣(transform),方向不變但scale可能會變  
Ax = $$\lambda$$x  
A矩陣代表social network關係(1:朋友關係,0:不是朋友)  
x代表重要性  
概念類似PageRank,page rank的值是連到他網頁的值加總    

HIT   
Hub  
推薦的authoritative有多高  
Authoritative page  
有多少hub推薦  

__最短路徑演算法__  
unweighted graph
1. BFS
2. Floyd-Warshall

__Group centrality__  
找出social network中幾個最有影響力的人  
或指定某幾個人想觀察這幾人的影響力  

group centrality一群人一起看,影響幾個人(有連線)  
不能將每個單一人的degree加總,會有重複的  

Social Group Analysis
community detection algorithm

Properties of cohesion
1. Mutuality of ties  
    所有subgroup彼此都有編相連,在graph中就是完全圖的概念(clique)  
    要求有點太嚴格  
2. Closeness or reachability of subgroup members  
    不需要直接有邊相連,間接有相連就行了  
3. Frquency of ties among members
    第一個是說假設有n個人必須要和n-1個人相連,那只需要和n-k個人相連就可以了  
4. Relative frequency of ties among subgroup members compared to non-member  

Clique  
> maximal complete subgraph,最大的子圖任兩點都有邊相連  





N-Clique 
在grahp中,任兩個點之間的距離<N  
N-clan  
必須是N-Clique  
在subgraph中,任兩個點之間的距離<N
N-club
不必是N-Clique  

K-plex  
如果是clique每個點的degree是n-1  
如果是k-plex,每個點的degree是n-k  
假設subgraph有4個點,2-plex每個點的degree至少是2 

K-core
至少和k個人是朋友
每個點的degree至少是k

Community Detection Approaches
1. Kernighan-Lin Alog(KL algorithm)
2. Hierarchical Clustering
3. Modularity Maximization
4. Bridge-Cut Algo

KL algorithm  
有權重的圖weighted graph
input: weighted graph
output: 切成兩個subgraph且橫跨兩群的crossing(cut)的值越小越好,
切成兩半那條線橫跨的cost越小越好  
希望群和群的相似度越大,同群的相似度越像
1. 任意切成兩半
2. 交換其中兩點使cost下降
3. 交換直到收斂

external cost
    crossing的cost(連向別群的cost)
internal cost
    連向同群的cost
difference
    external cost - internal cost

ab交換
Gain = Da + Db - 2Wab
Difference a + Difference b - 2*weighted ab

z = crossing edge與ab無關的其他cost總和
原來crossing cost = z + Ea + Eb - Wab
交換完new crossing cost = z + la + lb + Wab

交換數回合,若遇到gain是負的嘗試做交換下去,到最後再找gain最好的

時間複雜度:O(n^3)





#### reference
[社群雲端運算](http://newdoc.nccu.edu.tw/teaschm/1052/schmPrv.jsp-yy=105&smt=2&num=753868&gop=00&s=1&tea=101583.htm)
