---
layout: post
title: command line
---
`>` redirects and overwriting  
`>>` redirects and appending  
`|` redirects standard output to another command  
<!--more-->

`sudo apt-get update`  
`sudo apt-get upgrade`  
> 更新所有有新版本的套件




`sed 's/find/replace/g' filename`  
sed (stream editor)  
s: substitution替換  
g: global  



/*============================================================*/
					   概念
/*============================================================*/
自動化管理
	1.查詢登入檔
	2.追蹤流量
	3.監控使用者主機狀態
	4.主機各項硬體設備狀態
	5.主機軟體更新查詢
/*============================================================*/
						指令
/*============================================================*/
echo 
 	-n 不換行
打錯要清除
 	ctrl+u
切換到root
 	sudo -i / sudo sudo
查看linux版本
	lsb_release -a
	uname -a
cat filename
	-n 印出行數
wc (word count)
  
解壓縮
  tar -xf packetName.tgz
    -x 解壓壓
	-f 解壓縮的檔名
  zip
	unzip filename.zip
  rar
	unrar x filename.rar
關機
  sudo shutdown -h now
編輯	
  ed
  a			附加
  xxxx		內容
  .			結束
  w junk	寫入檔案
  q			離開
od (octal dump)
  -c (iterpret bytes as characters)
  -b (show bytes as octal)
  -x (print in hex)

du(disc usage)
可以看到目錄下目錄的檔案大小
  -a 印出目錄下所有檔案

chmod(change mode)
   e.g. 
	chmod 765 filename , chmod +x filename , chmod -w filename		4 read	
	2 write
	1 execute

權限後面的數字代表連結（link)數量
link
	ln junk linktojunk
	類似捷徑的功能link相同代表指到同一個檔案

讓程式在背景跑,跑玩再回傳結果 (&)
e.g. sleep 5 ; date &

關閉螢幕背光
	xset dpms force off

設定快捷鍵(alias)
	alias c = 'sleep 1 && xset dpms force off'	

netcat (nc)   nc -h (看document)(網路瑞士小刀)
	port 0~1023 需要root權限 1024以上開放一般使用者
	reference : netcat	

	測試遠端port是否有開啟
	→ 	nc  -v  IP address  Port

	兩台主機間複製檔案
	→nc  -l  PortNumber  >  my.jpg
	→ nc  targetIP PortNumber  <  my.jpg 

	在server端 nc -l -p 9999 > filename		(要收的先監聽)
	  cient端 nc 192.167.105.17 9999 < filename  

wget
	-t 重試次數,超時或中斷會重新連接 (-t0代表無窮多次)
	-c 設定續傳功能
	-r 遞迴下載
	-P 指定到某個目錄

改Bash樣式
  export PS1='\u@\h\W\$'

將bin加入PATH
  export PATH="$PATH:/xxx/xxx/bin" (暫時,關閉terminal消失)
  ->改bashrc

test
	-e 檔名是否存在
		test -e filename && echo "exist" || echo "not exist"
	-f 檔名是否存在且為檔案
	-d 檔名是否存在且為目錄
	-z string 判斷字串是否為0 空字串return true
		[-z "${HOME}" ];echo $?
		[_"${HOME}"_] 資料判斷,運算元兩端必須要有空白

ftp
ncftp (可以上傳資料夾)
	ncftp -u username -p password hostname
	get -r dirname (download)

修改檔案最後修改日期
touch -d 20170101 fileName
/*============================================================*/
							vim
/*============================================================*/
:vsp filename (ctrl-w + V)
 垂直分割 
:sp filename (ctrl-w + S)
 水平分割

ctrl+w = 各一半
ctrl+w + 放大
ctrl+w - 縮小

set mouse=a 啟用滑鼠功能

多行註解
	1.ctrl-v (linux)
	  ctrl-q (windows)
	  select block
	2.I (upper-case)
	3.// , # ...
	4.ESC
選取相同字  
	滑鼠移到字上按*,會顯示所有相同字  
/*============================================================*/
							疑難雜症
/*============================================================*/
無法將 /var/lib/dpkg/lock 鎖定
	lsof (list open files) 
	sudo lsof /var/lib/dpkg/lock
	sudo kill processID

連線憑證失效
ssh-keygen -f "/home/taiyi/.ssh/known_hosts" -R 140.119.164.118

網路設定 (static)
sudo vim /etc/networking/interfaces
	auto io
	iface io inet loopback
	auto eth0
	iface eth0 inet static
	address 140.119.164.118
	gateway 140.119.164.254
	netmask 255.255.255.0
sudo vim /etc/resolv.conf  (config DNS) (12.04開始捨棄)
	nameserver 140.119.1.110
改在/etc/network/interfaces
	dns-nameservers 140.119.1.110
設定完後更新
	sudo /etc/init.d/networking restart

ping 8.8.8.8 		能通代表有網路
ping google.com		能通代表可以看懂網址(DNS domain name)

/*============================================================*/  	
					shell Script
/*============================================================*/  
#!/bin/bash		宣告script類型
#1.功能與內容
#2.版本資訊
#3.作者聯絡方式
#4.建檔日期
#5.歷史紀錄

執行shell script
1. sh script
2. chmod +x script
   ./script

sh -x script 可以印出script每個執行的指令

exit n (n is a number)
echo $? 可以顯示n (可以自訂錯誤訊息)

註解 ＃

變數
	var=string (等號兩邊不能空白)
變數使用方法
	echo $x
	echo ${x}

cmd arg1 arg2 ... arg9
$0  $1   $2       $9

$* 所有args
$# args個數
shift 可將args 往前移一格

read
	echo "enter your name:"
	read name
	echo "hello $name."
expr
	expr 2 \* \( 3 + 4 \) 各項中間必須空白

/*===========================================================*/  						
				套件(plugin)
/*===========================================================*/ 
tmux
	prefix : ctrl + b
	ctrl-b + ctrl不放 + 上下左右可以調整視窗大小
	ctrl-b + 方向 移動視窗
	ctrl-b + % 垂直分割
	ctrl-b + z 放大/縮小視窗
	ctrl-b + c 再開一個新視窗
	ctrl-b + , 重新命名視窗名稱  
ssh
	sudo apt-get install openssh-server
	sudo /etc/init.d/ssh start
welcom banner
	sudo apt-get install toilet figlet
	echo "toilet -f mono12 -F gay taiyi" >> ~/.bashrc
	echo "fortune | cowsay | toilet --gay -f term" >> .bashrc
sshfs
	掛載遠端資料夾
	sshfs user@IP:/home/user/dictionary test 
	將遠端dictionary下的東西掛載到test
	停止掛載
		fusermount -u test
		停止掛載test


