#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：emhui 
@File    ：aa.py.py
@Author  ：wanghuifen
@Date    ：2021/11/2 19:30 
'''

import wget

'''
python中的多线程无法利用多核优势，如果想要充分地使用多核CPU的资源，在python中大部分情况需要使用多进程。Python提供了multiprocessing。
multiprocessing模块用来开启子进程，并在子进程中执行我们定制的任务（比如函数），multiprocessing模块的功能众多：支持子进程、通信和共享数据、执行不同形式的同步，提供了Process、Queue、Pipe、Lock等组件。
与线程不同，进程没有任何共享状态，进程修改的数据，改动仅限于该进程内 
url = 'http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/v2/trainsets/shoes_train.zip'
outpath = 'D:/Projects/_20210115_Object identification/Code/emhui/corpora/WDC/train/'
wget.download(url, outpath)
url2 = 'http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/v2/trainsets/all_train.zip'
outpath2 = 'D:/Projects/_20210115_Object identification/Code/emhui/corpora/WDC/train/'
wget.download(url2, outpath2)
'''

#方法二 继承式调用
import time
import random
from multiprocessing import Process


class Run(Process):
    def __init__(self,url,local_path):
        super().__init__()
        self.url = url
        self.local_path = local_path
    def run(self):
        #进行文件的下载
        wget.download(self.url, self.local_path)
        time.sleep(1)

if __name__ == '__main__':
    local_path = 'D:/Projects/_20210115_Object identification/Code/emhui/corpora/WDC/test/'
    url1 = 'http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/v2/goldstandards/computers_gs.json.gz'
    url2 = 'http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/v2/goldstandards/cameras_gs.json.gz'
    url3 = 'http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/v2/goldstandards/watches_gs.json.gz'
    url4 = 'http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/v2/goldstandards/shoes_gs.json.gz'
    url5 = 'http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/v2/goldstandards/all_gs.json.gz'
    p1=Run(url1, local_path)
    p2=Run(url2, local_path)
    p3=Run(url3, local_path)
    p4=Run(url4, local_path)
    p5=Run(url5, local_path)

    p1.start() #start会自动调用run
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    print('主线程')