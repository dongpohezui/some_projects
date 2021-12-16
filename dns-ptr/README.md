# dns-ptr

dns中的ptr记录用于将一个IP地址映射到对应的主机名。

用法：

```
python  dns-ptr.py  8.8.8.0/24  8.8.4.4  8.8.0.0/16 #将ip段或ip作为参数
```

或

```
vim dns-ptr.sh #修改ip
sh ./dns-ptr.sh 
```

或

```
nslookup -qt=ptr 1.1.1.1
```

TODO：

目前，rapiddns.io不包括ptr记录。

扫描所有Ipv4地址的prt记录，获取域名，做一个类似rapiddns.io的网站。
