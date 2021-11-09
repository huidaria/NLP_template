#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：emhui 
@File    ：zip_use
@Author  ：wanghuifen
@Date    ：2021/11/4 15:35 
@url     ：https://pypi.tuna.tsinghua.edu.cn/simple
'''
import requests
from zipfile import ZipFile
from pathlib import Path

DATASETS = [
    'http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/v2/repo-download/normalized_data.zip'
]


def download_datasets():
    for link in DATASETS:

        '''iterate through all links in DATASETS 
        and download them one by one'''

        # obtain filename by splitting url and getting
        # last string
        file_name = link.split('/')[-1]

        print("Downloading file:%s" % file_name)

        # create response object
        r = requests.get(link, stream=True)

        # download started
        with open(f'../corpora/wdc2/{file_name}', 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        print("%s downloaded!\n" % file_name)

    print("All files downloaded!")
    return


def unzip_files():
    for link in DATASETS:
        file_name = link.split('/')[-1]
        # opening the zip file in READ mode
        with ZipFile(f'../corpora/wdc2/{file_name}', 'r') as zip:
            # printing all the contents of the zip file
            zip.printdir()

            # extracting all the files
            print('Extracting all the files now...')
            zip.extractall(path='../corpora/wdc2/')
            print('Done!')


if __name__ == "__main__":
    Path('../corpora/wdc2/').mkdir(parents=True, exist_ok=True)
    download_datasets()
    unzip_files()