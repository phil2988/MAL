from imageDownloader import *
import os, os.path

def fetchData():
    folderPath = "is-it-apple/apples"
    folderPath2 = "is-it-apple/randoms"

    for filename in os.listdir(folderPath):
        file_path = os.path.join(folderPath, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    getImagesFomWebsite(
        directory=folderPath, 
        filename="is-it-apple/apples/apple", 
        baseUrl="https://www.shutterstock.com/da/search/red-apple")

    getImagesFomWebsite(
        directory=folderPath, 
        filename="is-it-apple/apples/apple", 
        baseUrl="https://www.shutterstock.com/da/search/red-apple?page=2")

    getImagesFomWebsite(
        directory=folderPath, 
        filename="is-it-apple/apples/apple", 
        baseUrl="https://www.shutterstock.com/da/search/red-apple?page=3")

    for filename in os.listdir(folderPath2):
        file_path = os.path.join(folderPath2, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    getImagesFomWebsite(
        directory=folderPath2, 
        filename=folderPath2 + "/random", 
        baseUrl="https://www.shutterstock.com/da/search/random")

    getImagesFomWebsite(
        directory=folderPath2, 
        filename=folderPath2 + "/random", 
        baseUrl="https://www.shutterstock.com/da/search/random?page=2")

    getImagesFomWebsite(
        directory=folderPath2, 
        filename=folderPath2 + "/random", 
        baseUrl="https://www.shutterstock.com/da/search/random?page=3")