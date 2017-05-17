---
layout: post
title: WebSearching 
comments: yes 
---
## CH8 Evaluation in information retrival      
評量search engine好壞
1. 搜到的index
2. 搜尋速度
3. 二氧化碳排放量
4. 和搜尋相關程度

<!--more-->
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
(P和R重要程度相同) 
> F = $$\frac{1}{\alpha\frac{1}{P}+(1-\alpha)\frac{1}{R}} = \frac{(\beta^2+1)PR}{\beta^2P+R}\$$ where $$\ \beta^2=\frac{1-\alpha}{\alpha}\$$

#### Ranked Evalution
P,R,F都是unordered(沒有等級)  
一個query會有一個Precision/Recall圖      
使用內插法(interpolated)可以得到一張較平滑的P-R圖  
(和機器學習ROC curve相似)  
P-R curve的面積越大效能越佳(代表Precision掉越慢)    

__內插法__  
> $$\ p_{interp}= \max\limits_{r'\ge r}\ p(r{'})\$$

r代表recall,
作法是從目前往後找最高的點向前填平,並重新畫P-R圖  

__Mean Average Precision(MAP)__  
> $$\ MAP(Q) = \frac{1}{|Q|}\sum^{|Q|}_{j=1}\frac{1}{m_j}\sum_{k=1}^{m_j}Precision(R_{jk})\$$

第一個sum算query平均  
第二個sum算precision平均  

__Precision at k__  
第k個搜索結果的Precision  

__R-Precision__  
文件中總共有R篇相關文章,以R作為cut-off,計算Precision  
e.g. 總共有100篇文章,其中10篇是相關的  
且搜尋結果是:RNRNN RRNNN RNNRR ....  
R=10(只看RNRNN RRNNN)計算Precision  
R-Precision = 0.4  

__Normalized Discounted Cumulative Gain(NDCG)__    
作者：Kalervo Jarvelin, Jaana Kekalainen(2002)  
> 用來衡量ranking quality

e.g.  
G = <3,2,3,0,0,1,2,2,3,0,...>   
G表示一個搜索的結果(3高度相關, 0沒關係)  
步驟:  
1. Cumulative Gain(CG)  
    >
    $$\
    CG[i] = \left\{\begin{matrix}
    G[1], &if\ i=1 \\ 
    CG[i-1]+G[i], &otherwise 
    \end{matrix}\right.
    \$$

    目前項+＝前一項(做成一個遞增的函數)  
    CG'=<3,5,8,8,8,9,11,13,16,16,...>    
2. Discounted Cumulative Gain(DCG)
    >
    $$\
    DCG[i]=\left\{\begin{matrix}
    G[i], & if\ i=1\\ 
    DCG[i-1]+G[i]/log_b\ i, & otherwise
    \end{matrix}\right.
    \$$

    DCG'=<3,5,6.89,6.89,6.89,7.28,7.99,8.66,9.61,9.61,...> if b=2  
    i代表排名,對排名做懲罰(除log<sub>2</sub> i),排名越後面懲罰越重  
    代表如果搜尋的結果很差,和理想的排序分數會相差很多  
    
3. Normalized Discounted Cumulative Gain(NDCG)  
    理想的搜索結果I=<3,3,3,2,2,2,1,1,1,1,0,0,0,...>(高度相關的排越前面)   
    理想搜索結果DCGI=<3,6,7.89,8.89,9.75,10.52,10.88,11.21,...>  
    nDCG<sub>n</sub> = $$\ \frac{DCG_{n}}{IDCG_{n}}(\frac{相關程度排序}{理想相關程度},做正規化)\$$  
    NDCG=<1,0.83,0.87,0.77,0.70,0.69,0.73,0.77,...>  

__benchmark 資料集__    
1. Cranfield
2. TREC(nist.gov)  
	Ad-hoc 資料集(1992-1999)  
3. GOV2
	2500萬篇文章  
4. NICIR
	cross-language IR
5. Cross Language Evaluation
6. REUTERS

__標記資料準則__   
Kappa measure   
> 標記資料是否一致的衡量標準,若標記不一致資料中就沒有truth  
  
Kappa計算公式  
> $$\ \kappa = \frac{P(A)-P(E)}{1-P(E)}\$$

__搜尋結果的呈現 Result Summaries__  
* 搜尋結果呈現：10 blue link
* 搜尋結果下方文字說明分為Static和Dynamic
Static:固定抽前50個字
Dynamic:利用nlp技術,根據搜索關鍵字動態做變化    
* quicklinks  
底下多的連結  

## Ch6 Model    
* Vector Space Model
* Probabilistic Information Retrieval 

|Empirical IR | Model-based IR|
|暴力法|有理論模型| 
|heuristic|數學假設|
|難推廣到其他問題|容易推廣到其他問題| 

__IR model歷史__   
* 1960  
    第一個機率模型  
* 1970     
    vector space model(75)    
    classic probabilistic model(76)   
* 1980  
    non-class logic model(86)     
* 1990  
    TREC benchmark   
    BM25/Okapi(94)  
    google成立(96)    
    Language model(98)  
* 2000  
    Axiomatic model(04)  
    Markov Random Field(05)     
    Learning to rank(05)  


__Vector space__  

|Vocabulary|V = { $$ w_1,w_2,w_3,...w_v $$ }| 
|Query|q =$$ \{q_1,q_2,...,q_m\} $$| 
|Document 文章|$$  {d_i} = \{  w_1,w_2,...  \} $$|   
|Collection文章集合|C = { $$ d_1,d_2,d_3,... $$ }|   
|R(q) query的集合|R(q) &sub; C|

__目標是找到近似query的集合__  
策略:  
1. Document select  
    挑文件如果是相關就收到集合  
    absolute relevance(系統必須決定是相關還是不相關)    
2. Document ranking  
    query的結果>threshold 就收進去    
    relative relevance(不必是1或0,相近到一定程度就收進集合)  

__Probability Ranking Principle(PRP)__  
> Robertson (1977)  
相似度量測函數f滿足,  
$$ 
f(q,d_1) > f(q,d_2)\quad iff\quad p(Rel|q,d_1) > p(Rel|q,d_2)     
$$    
f()值越大表示有越大的機率越相似  

__Relevance流派__  
* Similarity 相似度  
	 Vector space model(Salton et al,75)  
* Probability of relevance 機率模型
    Classical probaility Model(Robertson&Sparck Jones,76)  
    Learning to Rank(Joachims,02, Berges et al,05)  
* Probability inference 機率推論

__Vector Space Model(VSM)__  
將query和document表示成向量形式(similar representation)  
假設Relevance(d,q) = similar(d,q)  
利用cosine算相似度(1 ~ -1)   
high dimension(index的維度通常在10萬左右)   
good dimension -> orthogonal  
(好的維度切割應該是維度間彼此獨立(orthogonal),  
但是通常很困難,例如nccu後面接university的機率很高)  
VSM優點: Empirically effective,直觀, 實作容易  

__VectorSpace範例程式__  
[Building a Vector Space Search Engine in Python](http://blog.josephwilk.net/projects/building-a-vector-space-search-engine-in-python.html)   

大致步驟  
* 將所有文章使用join()成為一個string包含所有文章內容  
* 做string clean去除`.` `,` `多餘空白`,並轉為小寫  
* 將clean好的string利用空白切分成words array,丟到[Porter stem](https://tartarus.org/martin/PorterStemmer/)(去除字尾)
    > Porter Stemming Algorithm  
      作者:Martin Porter(2006)
* 刪除重複的word,使用set讓出現的word唯一   
* 得到所有整理完的words,做成index(將每個word編號),類似字典的概念  
* 將每篇文章分別建立自己的vector,並統計每個word出現的次數(term frequecy)
* 將輸入的query做成vector  
* 利用兩個向量做cosine計算相關程度

__相似度計算__  
1. Cosine Similarity  
> cosine = $$ \frac{V_1 \cdot V_2}{\|V_1\|\|V_2\|}$$
2. Jaccard Similarity  
> 相似度 = $$ \frac{交集}{聯集}$$

__TF-IDF Weighting__    
* TF(Term Frequency)  
	word count,單純統計字數出現頻率    
* IDF(Inverse Document Frequency)(反向的TF)  
	字的獨特性,如果某些字在很多篇文章出現次數都很高(例如:the,a,to,...)   
    IDF值就會很低(沒有鑑別度)	  
	IDF(t) = 1 + log(n/k)  (n:篇數,k:字出現次數)  
    例如文章總數是1000(n=1000),所有文章都有出現cat(k=1000),  
    IDF = 1 + log(1000/1000) = 1  
    如果只有1篇文章有出現cat,  
    IDF = 1 + log(1000/1) = 4  

TF-IDF計算方法:     
> weight(t,d) = TF(t,d) * IDF(t)  

__TF-IDF範例程式__  
[Tutorial: Finding Important Words in Text Using TF-IDF](http://stevenloria.com/finding-important-words-in-a-document-using-tf-idf/)

<br/>
## Ch11 Probabilistic Information Retrieval    

__Probability theory__  
* Joint probability   
    $$P(A\cap B)$$   
* Conditional probability    
    $$ P(A | B) $$  
    probability of A given that event B occurred.
* Chain rule   
    $$ P(A,B) = P(A\cap B) = P(A|B)P(B) = P(B|A)P(A) $$
* Partition rule  
    $$ P(B) = P(A,B) + P(\bar{A},B) $$  

[Chain rule example wiki](https://en.wikipedia.org/wiki/Chain_rule_(probability))   
> 有兩個甕第一個甕放1個黑球2個白球,第二個甕放1個黑球3個白球  
事件A是選到第一個甕,事件B是選到白球  
$$ P(B|A)= \frac{2}{3} $$ 在選到第一個甕的情況下拿到白球  
$$ P(A,B)=P(B|A)P(A)=\frac{2}{3} \times \frac{1}{2} $$


__Bayes\' Rule__  
$$ P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$  

from chain rule: $$ P(A|B)P(B) = P(B|A)P(A) $$  
```text
P(A)    : 事前機率(prior probability)  
P(A|B)  : 事後機率(postior probability)  
P(B|A)  : likelihood   

The term likelihood is just a synonym of probability.   
```
__Odds__  
$$ O(A) = \frac{P(A)}{P(\bar{A})} = \frac{P(A)}{1-P(A)}$$
> an event provide a kind of multiplier for how probabilities change.

__Probability of Relevance__  
* Random variables:   
    * query Q 
    * document D 
    * relevance R &isin; {0,1}   
      (1:相關,0:不相關)
* Goal: P\(R=1\|Q,D\) to rank relevant  
    利用query和document相似度的機率做排名  


__Refining P(R=1\|Q,D) Methods__   
1. Conditional Models(Discriminative Models)  
    * 利用各種方法找出機率P = f(x)  
    * 利用資料訓練參數  
    * 利用model去排列未知的document  
    e.g. Learning to rank,類神經網路,...
2. Generative Models  
    * compute the odd of O(R=1\|Q,D) using Bayes\' rules  
    先找出資料的分佈再做預測  
    * How to define P(Q,D\|R)  
        * Document generation: P(Q,D\|R)=P(D\|Q,R)P(Q\|R)   
            query放到條件 (e.g RSJ model)   
        * Query generation: P(Q,D\|R)=P(Q\|D,R)P(D\|R)   
            document放到條件 (e.g language model)  


#### RSJ Model [(Binary Independence Model)](https://en.wikipedia.org/wiki/Binary_Independence_Model)
利用**Odd**值做ranking的依據:  
$$ O(R|D,Q) = \frac{P(R=1|D,Q)}{P(R=0|D,Q)} = \frac{ \frac{P(D|Q,R=1)P(R=1|Q)}{P(D|Q)} }{ \frac{P(D|Q,R=0)P(R=0|Q)}{P(D|Q)} } = \frac{P(D|Q,R=1)P(R=1|Q)}{P(D|Q,R=0)P(R=0|Q)} \quad(1)$$  

> $$ \frac{P(R=1|Q)}{P(R=0|Q)} $$ 
對document ranking沒有影響,視為常數  

$$ \frac{P(D|Q,R=1)}{P(D|Q,R=0)} = \prod_{t=1}^{M} \frac{P(D_t|Q,R=1)}{P(D_t|Q,R=0)} \quad(2)$$    

> 將document拆成多個獨立的document term連乘積,且$$ D_t \in \{0,1\} $$

$$ O(R|D,Q) = O(R|Q) \cdot \prod_{t=1}^{M} \frac{P(D_t|R=1,Q)}{P(D_t|R=0,Q)} \quad(3)$$    

> (2)代入(1)可以整理出(3)  

$$ O(R|D,Q) = O(R|Q) \cdot \prod_{t:D_t=1}^{M} \frac{P(D_t = 1|Q,R=1)}{P(D_t = 1|Q,R=0)} \cdot \prod_{t:D_t=0}^{M} \frac{P(D_t = 0|Q,R=1)}{P(D_t = 0|Q,R=0)}\quad(4)$$    

> 將document term分為出現或是不出現,(3)&rarr;(4)

$$ p_t = P(D_t = 1|Q,R=1) $$

$$ u_t = P(D_t = 1|Q,R=0) $$

> $$ p_t $$ 表示term出現在document且和query相關的機率  
$$ u_t $$ 表示term出現在doucment且和query不相關的機率  

$$ O(R|D,Q) = O(R|Q) \cdot \prod_{t:D_t=Q_t=1} \frac{p_t}{u_t} \cdot \prod_{t:D_t=0,Q_t=1} \frac{1-p_t}{1-u_t} \quad(5)$$    

> 假設$$ Q_t = 0 \; then \; p_t = u_t$$(假設可以做改變)    
意思是沒出現在query的term就不用考慮,只考慮$$ Q_t = 1 $$    
左邊連乘積表示 query term found in document  
右邊連乘積表示query term not found in document  


$$ O(R|D,Q) = O(R|Q) \cdot \prod_{t:D_t=Q_t=1} \frac{p_t(1-u_t)}{u_t(1-p_t)} \cdot \prod_{t:D_t=0,Q_t=1} \frac{1-p_t}{1-u_t} \cdot \frac{1-p_t}{1-u_t} \quad(6)$$    

> 右邊連乘積乘上 $$ \frac{1-p_t}{1-u_t}$$, 所以所邊必須要除$$ \frac{1-p_t}{1-u_t}$$才會相等   

$$ O(R|D,Q) = O(R|Q) \cdot \prod_{t:D_t=Q_t=1} \frac{p_t(1-u_t)}{u_t(1-p_t)} \cdot \prod_{t:Q_t=1} \frac{1-p_t}{1-u_t} \quad(7)$$    

> 右邊連乘積是query not found in document  
概念大概是將query found in document也計算進去,  
不管有沒有出現在document都乘  
整理後右邊連乘積的範圍就會和document無關   
在對document ranking時就視為常數  

$$ RSV_d = log \prod_{t:D_t=Q_t=1} \frac{p_t(1-u_t)}{u_t(1-p_t)} = \sum_{t:D_t=Q_t=1} log \frac{p_t(1-u_t)}{u_t(1-p_t)} \quad(8)$$    

> 取log後就得到Retrieval Status Value(RSV),  
log是monotonic function不會改變ranking順序  

__RSJ Model:No Relevance Info__  

$$ log O(R=1|Q,D) \approx  
\sum_{t=1,D_t=Q_t=1}^{k} log \frac{p_t(1-u_t)}{u_t(1-p_t)}  $$    

如果沒有給relevance judgements,  
* assume $$ p_t $$ to be a constant  
* Estimate $$u_t$$ by assume all documents to be non-relevant  

1979 Croft&Harper  

$$ log O(R=1|Q,D) \approx  
\sum_{t=1,D_t=Q_t=1}^{k} log \frac{N - n_t + 0.5}{n_t + 0.5}  $$    

>N: number of documents in collection  
$$n_t$$ : number of documents in which term $$D_t$$ occurs 

$$\sum log( \frac{總文章數 - 某個字出現在文章次數 + 0.5}{某個字出現在文章次數 + 0.5})$$

> 只看在document中和query相關的字,並加總每個字算出來的值  

__RSJ Model: with Relevance Info__  
* Maximum Lieklihood Estimate(MLE)
* Maximum A Posterior(MAP)

> RSJ model的performance還遠比不上vector space model 


__Improving RSJ__  
* adding TF  
* adding Doc.length  
* query TF  

改善後的最終公式稱作**BM25**

<hr>
<!-- 20170416 -->
## CH12 Language models for informatio retrieval  

__unigram language model__    
> 每個word只有單一的狀態,可以建立一個table放每個word對應到的機率 

一個string出現的機率就是每個word的機率連乘積   
Language model應用：語音系統的語言校正  

Language model屬於query generation process    
每一篇document視為一個language model  
ranking的計算是根據P(Q|D)  

__計算P(Q|D)__  
* [Multinomial model](https://en.wikipedia.org/wiki/Multinomial_distribution)   
    ![multinomial Distribustion](/img/websearching/MultinomialDistribution.png)  

$$ P(q|M_d) = P((t_1,...,t_{|q|})|M_d) = \prod_{1 \leq k \leq |q|} P(t_k|M_d) = \prod_{distinct\;term\;t\;in\;q}P(t|M_d)^{tf_{t,q}}$$   

> |q|: length of query  
$$t_k$$ : query的第k個位置的token  
$$tf_{t,q}$$ : term frequency of t in q  

__估計參數__  
* [Maximun Likelihood Estimation(MLE)](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)   
> $$\hat{P}(t|M_d) = \frac{tf_{t,d}}{|d|}$$

```text
hat符號表示估計值的意思  
```

__smooth the estimates to avoid zeros__  
> 避免0產生,相乘後結果很差  

smooth方法
1. Mixture model  
    $$P(t|d)=\lambda P(t|M_d) + (1-\lambda )P(t|M_c)$$  
    > $$M_c$$ : the collection model  
    $$\lambda$$ 的設定好壞會影響效能   

    e.g.  
    Collection = {$$d_1,d_2$$}  
    $$d_1$$: Jack wants to play game  
    $$d_2$$: Tom is cat  
    query q: Tom game  
    $$\lambda = \frac{1}{2}$$  
    
    P(d|$$d_1$$) = [(0/5 + 1/8)/2]$$\cdot$$[(1/5 + 1/8)/2] $$\approx$$ 0.0101   
    P(d|$$d_2$$) = [(1/3 + 1/8)/2]$$\cdot$$[(0/3 + 1/8)/2] $$\approx$$ 0.0143  
    rank $$d_2 > d_1$$  
2. [Laplace smoothing](https://www.youtube.com/watch?v=gCI-ZC7irbY)
<!-- 20170416  -->

<!-- 20170427 -->
__Text Generation with Unigram LM__  
* sampling    
    由一個特定主題的model,裡面會有各個字出現的機率(每一個model會有一個distribution,機率分佈),取出一些機率較高的字可以形成document
* estimation  
    拿到一個document,預估出model  

<!-- 20170427 -->

<!-- class -->
MLE -> smoothing(去除機率為0的值)用copers這個smooth方法  

smoothing方法  
* Jelinek-Mercer
* Dirichlet prior
* Absolute discontuning

## CH13 Text Classification and Naive Bayes  
* Text Classification
* Naive Bayes
* Naive Bayes Theory
* Evaluation of Text Classification

#### Text Classification  
__standard supervised__  
1. Pre-define categories and lebel document
2. classify new documents

分類的方法
1. 人工判斷
    準確但成本高
2. Rule-based
    很多if/else的rule,看到王建民就分類到體育新聞    
    e.g. google Alert  
3. Statistical/Probabilistic  
    * Instance-based classifiers  
        e.g. kNN  
    * Discriminative classifiers  
        學習出分隔的形式(一條線,一棵樹)  
        資料少容易overfit  
        e.g. Decision tree,Neural Network
    * Generative classifier  
        利用大量資料學習出分佈模型(mean,varience)  
        e.g Naive Bayes

__K-Nearest Neighbor Classifier(KNN)__  
Keep all train data  
優點:不需要training(每一個點都記錄下來)  
缺點:不能做大量資料  

## Naive Bayes Classifier  
arg max_c 找到一個c(類別)使P(c|d)最大
避免under flow -> 取log連乘變成連加

機率會有零產生->避免這種情形全部機率做加1

估計事前機率 是這個類別的機率和不是這個類別的機率
估計完這些係數training就結束

P(c|d) 給document判斷是哪個類別
P(c|d) = P(c)P(d|c)/P(d)  分母不考慮,和分類無關
且分子越大越好
P(d|c)可以拆成多個Term的連乘積
且假設每個字之間獨立  

Feture Selection
* Reduces training time
挑特定的字訓練模型

Two idea
* Mutual information
    計算字的交互作用,每一個字會有一個值,找gap最大的切開,拿比較相關的字做訓練  
* CHI-Square statistic
    用機率方法計算,算出機率高的就拿去做訓練資料

Evaluations
    測試資料和training data不能有overlapping

<!-- class -->

<!-- class -->
# CH14 Vector Space Classification
* Rocchio
* kNN classification
* linear classification

Vector Space Classification
1. 同一類文章如果同類會形成一個連續的空間
2. 如果不同類別則空間不會有overlap

#### Rocchio 
早期1970在SMART搜索系統中,負責用在relevance feedback
將同類文章標示成prototype vector
prototype = centroid of members of class
每一類算出一個重心(全部vector加起來做平均)
計算相似度可以使用distance

performance較naive bayes差

#### kNN classification
k Nearest Neighbors(kNN)
鄰近k個鄰居做投票決定分類結果  
k = 1 過於sensitive
k 太大過麼模糊
通常選則奇數(3,5,7)

計算相似度
Euclidean distance,tfidf,...
沒有任何學習

Bias-Variance Tradeoff
用來衡量
Bias 猜出的結果和真實結果差距多少
     差距越大Bisa越高
Variance
    每次猜出來的結果差異會不會很大

ideal情形是low Bias , low Variance

#### Linear classification
consider 2 class problem
Sum of WiXi
線性分類器在一二三維中分別表示點線面,一個分界面

e.g. Naive Bayes,Percept

#### more than two classed
* one-of
    每個資料只能分到一類
* Any-of or multi-label

<!-- class -->
