apache access logを解析するデモです。

## 環境

```
% sw_vers 
ProductName:	macOS
ProductVersion:	12.6.3
BuildVersion:	21G419
```


## LogFormat

今回のログ解析では以下のlog format指定で出力される`/var/log/apache2/access_log`を対象とします。

```
/etc/apache2/httpd.conf
```

LogFormat
```
LogFormat "%h %l %u %t \"%r\" %>s %b \"%{Referer}i\" \"%{User-agent}i\"" common
```
