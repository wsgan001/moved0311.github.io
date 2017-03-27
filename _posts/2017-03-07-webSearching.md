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
(P和R重要程度相同) 
> F = $$\frac{1}{\alpha\frac{1}{P}+(1-\alpha)\frac{1}{R}} = \frac{(\beta^2+1)PR}{\beta^2P+R}\$$ where $$\ \beta^2=\frac{1-\alpha}{\alpha}\$$

#### Ranked Evalution
P,R,F都是unordered(沒有等級)  
一個query會有一個Precision/Recall圖(崎嶇的坡)    
使用內插法(interpolated)可以得到一張較平滑的P-R圖(和機器學習ROC curve相似)
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

__Result Summaries__  
* 搜尋結果呈現：10 blue link
* 搜尋結果下方文字說明分為Static和Dynamic
Static:固定抽前50個字
Dynamic:利用nlp技術,根據搜索關鍵字動態做變化    
* quicklinks  
底下多的連結  

__Ch6 Model__  
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


The notation of Relevance
* Relevance
	* Similarity 相似度  
		 Vector space model  
	* Probability of relevance 機率模型
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
* 做string clean去除. , 多餘空白,並轉為小寫  
* 將clean好的string利用空白切分成words array,丟到Porter stem(去除字尾)
    > Porter Stemming Algorithm  
      作者:Martin Porter(2006)
* 刪除重複的word,使用set讓出現的word唯一   
* 得到所有整理完的words,做成index(將每個word編號)
* 將每篇文章分別建立自己的index,並統計每個word出現的次數
* 將輸入的query做成vector  
* 利用計算相關程度

__index值使用TF-IDF__    
* TF(Term Frequency)  
	word count,單純統計字數出現頻率    
* IDF(Inverse Document Frequency)(反向的TF)  
	字的獨特性,如果某些字在很多篇文章出現次數都很高    
	例如:the,a,to,...  
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

下禮拜講Ch11  





