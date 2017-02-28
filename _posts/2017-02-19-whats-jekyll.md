---
layout: post
title: jekyll
---
## 安裝  
檢查ruby版本  
`ruby -v`   
用gem安裝jekyll和bundler  
`gem install jekyll bundler`  
new一個blog  
<!--more-->
`jekyll new moved0311.github.io`   
執行server (localhost:4000)  
`bundle exec jekyll serve`   
## 編輯文章
#### 強制換行
> 行尾加入兩個以上空白

## 額外功能 
#### 加入Read More  
在index.html找到`{_{ post.content }_}`  
取代成  
(記得把_拿掉 jekyll不能出現連續{ { 或是 { % ) 
```text
{_% if post.content contains '<!--more-->' %_}  
  {_{ post.content | split:'<!--more-->' | first }_}  
    <p class="more">  
      <a href="{_{ post.url }_}">Read More &raquo;</a>  
    </p>  
{_% else %_}  
  {_{ post.excerpt }_}  
{_% endif %_}  
```

