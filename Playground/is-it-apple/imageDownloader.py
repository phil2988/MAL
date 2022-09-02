from math import prod
import requests # request img from web
import shutil # save img locally

from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
from webdriver_manager.chrome import ChromeDriverManager

from pathlib import Path
import os


def getImagesFomWebsite(baseUrl: str, filename: str, directory: str):
    print("Setting up")
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get(baseUrl)
    content = driver.page_source
    soup = BeautifulSoup(content)

    print("Finding Apples from website: " + baseUrl)
    appleUrls = []

    for img in soup.find_all('img'):
        appleUrls.append(img.get('src'))

    print("Found " + str(appleUrls.count) + " urls")

    print("Downloading images")

    if not os.path.exists(directory):
        os.makedirs(directory)

    index = len(os.listdir(directory))
    for url in appleUrls:
        res = requests.get(url, stream = True) 
        if res.status_code == 200:
            with open(filename + str(index) + ".png",'wb') as f:
                shutil.copyfileobj(res.raw, f)
            print('Image successfully Downloaded: ',filename)
        else:
            print('Image Couldn\'t be retrieved')
        index += 1
