---
layout: post
title: JavaScript
---
多使用const並用let取代var  
宣告物件別用new直接用obj = {}  
`...`展開運算子    
<!--more-->
#### 陣列複製
```js
a = [1,2,3,4,5];
b = a;  //(X) -> 當修改b時,a會跟著修改

b = [...a];
```
#### 字串串接
```
`hello, my name is ${name}.`
```
```js
function concatenateAll(...args){
    return args.join('');
}
```
__callback function__  
.run().then()  
執行非同步呼叫執行完run後拿回控制權,並交代執行完再用then接住call back

__promise用法__  



#### reference
[Airbnb JavaScript Style Guide](https://github.com/airbnb/javascript#hoisting)
