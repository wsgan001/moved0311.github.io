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
`(記得把_拿掉)`
<hr>
#### 加入Read More  
在index.html找到`{_{ post.content }_}`  
取代成  
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
<hr>
#### Facebook Comments Plugin
`_include/facebook-div.html`  
```html
<div class="fb-comments" data-href="{{ page.url | remove_first: '/' | prepend: site.baseurl | prepend: site.url }}" data-width="auto" data-numposts="5"></div>
```
`_include/facebook-script.html`  
```html
<div id="fb-root"></div>
<script>(function(d, s, id) {
  var js, fjs = d.getElementsByTagName(s)[0];
  if (d.getElementById(id)) return;
  js = d.createElement(s); js.id = id;
  js.src = "//connect.facebook.net/en_US/sdk.js#xfbml=1&version=v2.8&appId=228652297542969";
  fjs.parentNode.insertBefore(js, fjs);
}(document, 'script', 'facebook-jssdk'));</script>
```
`_layouts/post.html`加入
```html
  {_% if page.comments != false %_}
    {_% include facebook-div.html %_}
  {_% endif %_}
```
`_layouts/default.html`<body>內加入 
```html
{_% if page.comments != false %_}{_% include facebook-script.html %_}{_% endif %_}
```
[內嵌留言](https://developers.facebook.com/docs/plugins/embedded-comments#how-to-get-a-comments-url)
<hr>
