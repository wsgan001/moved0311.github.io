---
layout: post
title: SocialCloudComputing
---
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

#### Network Properties
1. small-world effect
    六度分離理論  
    靠點和點距離關係分析    
2. Transitivity
    朋友的朋友很可能也是你朋友  
	[Clustering Coeffieient](https://zh.wikipedia.org/wiki/%E9%9B%86%E8%81%9A%E7%B3%BB%E6%95%B0)  
3. Degree distribution
	Real world network : Power law		  
	> P<sub>k</sub> = CK<sup>-&alpha;</sup>  

	Heavy-tailed degree distribution  
	大量很低的數量,集合起來還是很驚人  
4. Network resilience
	如果拿掉一些點/邊,連通性會有什麼變化？  
    (e.g.有些人掛了,離職)  
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


<!-- 20170413 start --> 
__Properties of cohesion 凝聚力的判斷__  
1. Mutuality of ties  
    所有subgroup彼此都有編相連,在graph中就是完全圖的概念  
    e.g. clique   
2. Closeness or reachability of subgroup members  
    不需要直接有邊相連,間接有相連就行了  
    e.g. N-clique,N-clan,N-club  
3. Frquency of ties among members  
    Mutuality of ties是說假設有n個人必須要和n-1個人相連,  
    那Frquency of ties among members只需要和n-k個人相連就可以了  
    是Mutuality of ties放寬版本  
    e.g. K-plex,K-core  
4. Relative frequency of ties among subgroup members compared to non-member  

__Clique__    
> maximal complete subgraph,最大的子圖任兩點都有邊相連  
<hr>

![clique img](/img/cloudcomputing/community01.png)

* N-clique   
    在grahp中,任兩個點之間的距離<N    
    e.g. 2-cliques: {1,2,3,4,5},{2,3,4,5,6}  
* N-clan  
    必須是N-clique  
    在subgraph中,任兩個點之間的距離<N  
    e.g. 2-clan: {2,3,4,5,6}   
    (4->5要經過6,但只考慮1,2,3,4,5這個subgraph)
* N-club  
    不必是N-clique,但一定要是subgraph of n-cliques    
    2-clubs: {1,2,3,4},{1,2,3,5},{2,3,4,5,6}  
* K-plex  
    如果是clique每個點的degree是n-1  
    如果是k-plex,每個點的degree是n-k  
    假設subgraph有4個點,2-plex每個點的degree至少是2 
* K-core  
    至少和k個人是朋友
    每個點的degree至少是k


__Community Detection Approaches__  
1. Kernighan-Lin Alog(KL algorithm)
2. Hierarchical Clustering
3. Modularity Maximization
4. Bridge-Cut Algo

### KL algorithm  
> input: weighted graph  
output: 切成兩個equal-size subgraph,且橫跨兩群的crossing edge   
目的是相望群和群之間差異大,群內部的差異小  

**步驟**  
1. 任意切成兩半
2. 計算每一點的difference
3. 計算每個邊的gain  
4. 從gain最大的開始做交換,交換後的點不再考慮(lock)
5. 交換到直到全部的點都被lock住
6. 挑gain總和最大的就是最終交換結果  

交換數回合,若遇到gain是負的紀錄下來並繼續嘗試做交換,到最後再找gain最好的
交換完後的點就lock住不進入下一回合

* external cost  
    crossing edge的cost(連向別群的cost)(cut-size)   
* internal cost  
    連向同群的cost  
* difference  
    external cost - internal cost  

**Gain**
> 用來評估是否要交換的值
例如a,b屬於不同群,ab做交換  
Gain = $$ D_a + D_b - 2\times W_{ab} $$  
(Difference a + Difference b - 2*weighted ab)  

若考慮a,b交換  
old cost = $$ z + E_a + E_b - W_{ab} $$   
new cost = $$ z + I_a + I_b + W_{ab} $$  

> z (與a,b沒有連接的其他crossing edge總和)  
E (external cost)  
I (internal cost)

__KL algorithm複雜度__  
$$ O(n^2) $$ 找到最適合交換的兩點,有n pair要交換  
&rArr; $$ O(n^3) $$ 

<!-- 20170413 end --> 

hiraichiecal 
bottom-up  
每一回合都找兩個最像的做合併  
single link  
    距離取min
complete link
    距離取max

分群      
community 同群邊的值要越大越好

Distance Matrix
1. Approach1    
計算weights Wij 
* i到j的路徑越多代表i和j關係越好  
    * 只能找non-overlapped paths
    * 只要i到j的路徑都算(weighted by length)
    Xij = 1/Wij

    Reduction
        由A問題轉換到B問題  
2. Approach 2
    如果i和j視同一群,那他們有相似的behavior
    behavoir
        i和j到commuinty其他點的平均距離相似  
3. Approach 3
    J(i,j)/min(Ki,Kj)  
    看兩個人共同朋友個數,共同朋友越多J(i,j)越大
 
Edge-removal Approach  
不斷的拿掉邊,會出現越多的群數,直到符合要的群數  
拿掉bridge edge, 
1. betweeness
一開始想說可以用degree少,但不夠完全  
    在centrality的betweeness是以node考量  
    在這的betweeness是以edge考量

GN algorithm  
拿掉betweeness最高的邊 -> 重算betweeness -> 計算community
top-down(起始是一個commuinity,並分群下去)
計算邊的betweeness  
1. shortest path  
    任兩點最短路徑有多少條會經過邊  
2. Random-walk 
    計算a會走到b的機率  
    a走到b會經過邊v的機率  
3. Current-flow
    引進電路學概念的計算方法  

缺點    
1. 計算最短路徑耗時
    O(m^2n)  
    m edge (O(mn)betweeness)  
2. 什麼時候停?   

改善  
1. 
    Partial betweeness (Apprximation)  
    Randomly sampled by Monte Carlo Estimate  
2. 
    Edge clustering coefficient
      coefficient越高代表關係越好
    the smaller coefficient the higher betweeness
 
## Modularity  
    Modularity measure:
        how good a particular partition forms a community.
        評估community切分的好不好  

U 看internal edge的比例  
R 平均i和j會有邊的機率(期望值)        
Q = U - R 

Q = 0 no community  
Q ~ 1 prefect cut  

__Newman Fast Alogorithm__  
利用hireachcal合併,並每個步驟算modurity,並找出最高的Q做切分  
__Bridge cut__  
integrity一致性  
N(v)
d(v): degree of node
Density
Direct neighbor subgraph of v  
Clustering coeffiecient   
	觀察v的鄰居的朋友關係  
	例如v有4個朋友,那4個人最多有6個關係,算關係的比例  
	實際上有關係/最多有幾個關係  
Bridge Centrality  
	__rank__ of betweenness centrality * __rank__ of bridging coeffiecient  
	如果只考慮betweeness(global)會有一些情況不太好    
	加入bridge centrality可以考慮到local的特質  

__Community Search__  
給一個social network,並給一些query(其中幾個人),  
given grahp G, a set of query node  
goal: find a densely subgraph of G, and contains the query nodes  

__Induced Subgraph__  
xy edge在G中,xy edge也要在induce subgraph中  

goodness function
1. edge degree
時間複雜度太大  
2. average degree
3. minumin degree
這群人認識最少的人,讓這個人的值變大
induced subgraph的degree  
容易受到outlier影響  

Constrain  
distance constrain  
限制邀請來的人的最長距離  

__Monotone Function__  
* monotone increaing
* monotone decresing
* non-monotone

<hr>
<!-- 20170421 start --> 
>
* Link Prediction
* Node-wise Similarity Based Methods
* Topological Pattern Based Methods
* Probabilistic Model Based Methods

## Link Prediction  

__Goal__
1. Predict the existence of links
2. Predict the type of links

__Strategies of Prediction__  
1. Knowledge-driven strategy  
    專家系統(領域專家提供rule)    
2. Data-driven approach  


__problem__    
1. Link existence prediction  
    邊是否存在  
2. Link classification  
    關係的總類  
3. Link regression  
    最重要的邊是哪一個  

__Application__    
1. Web hyperlink creation
2. Collaborative filitering
3. Information retrieval
4. Clustering
5. Record linkage

## Node-wise Similarity Based Method    
> 計算兩個點的相似度,如果兩個點很相似他們可能就有link  
e.g. Similarity between words  
觀察word的前後文字來判斷相似程度  

__Learning-Based Similarity Measure__    
* Binary Classification Approach  
    * Decision Tree  
* Regression-based Approach(回歸)    
    e.g linear regression  
    $$ Y = \alpha + \beta_1X_1 + \beta_2X_2 + ... + \beta_nX_n $$  
    利用學習方式估計出$$ \alpha , \beta  $$

## Topological Pattern Based Methods  
> 計算兩點之間的分數,若大於某個值就表示他們之間有關係(連線)

* __Local Method__   
    * Common Neighbors(CN)    
        計算共同的鄰居  
    * Salton Index  
        類似cosine similiary  
    * Jaccard Coefficient(JC)  
        交集/聯集  
    * Leicht-Holme-Newman Index(LHN)  
    * Hub Promoted Index(HPI)  
        Hub概念像是入口網站,類似目錄連到很多子分支  
    * Hub Depres Index(HDI)    
    * Adamic/Adar(AA)   
        x和y是朋友的分數是x和y的共同朋友的鄰居-x和y 倒數總和  
    * Resource Allocation Index(RA)  
        和AA差在分母沒有取log  
    * Preferential Attachment  
        x和y是朋友的分數就是x的鄰居乘上y鄰居  

    Performance: RA > AA > CN > ... > PA

* __global Method__  
    * katz  
        看x到y距離是1,2,3..n的path有幾條,乘上一個參數做加總  
    * Hitting Time  
        x走到y,做random work的期望值作為比較條件  
    * PageRank  
    * SimRank  

## Probabilistic Model Based Methods  
e.g. relational Markov model  
<!-- 20170421 end --> 


<!-- 20170420 --> 
## Labeld Social Network  
Type of Labels  
1. Binary
2. Numeric  
3. cate
4. text-free

## Label Prediction
根據已知的label預測未知點的label  
1. Inference vs. Learning  
    Inference(unsupervised)  
    Learning(supervised)  
2.Disjoint vs. Collective  
    Disjoint
    沒有標籤的點就不考慮  
    Collective  
    沒有標籤的點也會放進去考慮  
3.Across-network vs. within-network learning  
    Across-network拿一個social network model去預測另一個social network  
    within-network拿全部資料做的model來做預測   

## Clues to Predict Labels  
* Label Independent approaches 特徵值沒有用到label的訊息  
1. Correlation between  
    Attribute of node i (年紀)
    Label of node i (身份)
2. Correlation between  
    Network Structures of node i (between centrality)
    Label of node i
* Label Dependent Approaches
3. Correlation between  
    知道鄰居來預設未知,用其他點來預測未知點   
4. Correlation between  
    利用unlabel點來預測  

Relational Neighbor Classifier  
    -> disjoint
    簡單說就是看鄰居多數是什麼就判斷node是什麼  
    問題：  
    如果已知的點很少,unknown的很多(Sparse label),若只用一個點就判斷就沒那麼可靠  
    解決：
    Iterative Relation Neighbor classifier    
    判斷分為好幾回合,若多數點是unknown那就判斷是unknown   
    unknown也是一種label  
<!-- 20170420 --> 

<!-- class --> 
__Ghost Edges for Node Label Prediction__  
將一些不是直接連接的node但有影響力的點,用ghost edge連起來  
那怎麼判斷點的重要性  
-> Random walk with Restart   
    有一定的機率會跳到起點

__Steady-state Probability__  
Markov Process    
    城市和郊區遷移問題

利用Random walk with restart計算所有點對某點的影響力  
在對這些影響力做等級劃分,依照機率分為ABCDEF...等級  

__Two Classifers__  
1. GhostEdgeNL
2. GhostEdgeL
    Logistic regression  

## Information Diffusion 
消息/疾病擴散  
想知道擴散方向,可以擴散到哪等等問題  

__3 elements of diffusion process__  
1. Senders
2. Receivers  
3. Medium(channel)

__Types of information Diffusion__  
* herd behavior(global information)  
    群眾行為,大多數人怎麼做,就有可能會跟著做  
* information cascades(local information)  
* diffusion of innovation
* eqidemics

__Diffusion Models__  
1. Descriptive models  
    機率模型  
2. Operational models  
    一步一步的擴散  
    Each node can be Active / Inactive  
    Assumption:
        * node can switch from inactive to active  
        * cannot switch from active to inactive  
    e.g.  
        1. Linear Threshold Model  
        2. Independent Cascade Model  

__Linear Threshold Model__    
每個人都會有一個threshold代表會變成active門檻值    
每個邊上會有影響力的值(可以是單向或雙向)    
如果你的鄰居加總的影響力大於你本身的threshold那你也會變成active  
由Senders開始一步一步的擴散  

__Independent Cascade Model__  
每一個人只能影響鄰居一次,失敗了不能再影響一次  
邊上是影響成功的機率  

## Influence Maximization Problem  
給一些起始的senders觀察最後有哪些人被影響  
給k個senders並且找出這k個senders是誰且最後影響的人數最多  
(應用: 廣告要放在哪裡)  
Constrained optimization problem    

問題難度: NP-hard   
證明這個問題的難度:  
    符合Submodular Function,且要在所有node中找出k個,使得f(k)為maximized  
    已經被證明為NP-hard問題  

__Submodular Function__  
* Non-negative
* Monotone
* Submodular 邊際效應遞減  
>   f(a,b,v) - f(a,b) > f(a,b,c,v) - f(a,b,c)  
    原來有a,b加入v所增加的量大於原來有a,b,c加入v所增加的量  
    例如原本能考60分,讀一天可以考到80分,  
    和原來能考90分,可是讀一天只能考到95分  

遇到NP-hard問題考慮的解法
* Approximation Approach  
* 平行運算  

__Approximation Approach__  
Greedy algorithm  
每回合找出Submodular最大的作為sender  

__Degree Discount__  
(下次講)

__Outbreak Detection__  
> 能不能透過放sensor提早知道消息的擴散  

給一個network G(V,E),找到placement A(sensor),to max R(A)
R(A) : reward  
c(A) : cost  

placement objective  
* detection likelihood  
* detection time  
    多久偵測到  
* 
<!-- class -->
