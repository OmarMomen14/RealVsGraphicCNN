from image_loader import Database_loader
from model import Model
from PIL import Image
import requests

model = Model(config = 'Config1', filters = [32, 64], feature_extractor = 'Stats', batch_size = 50, using_GPU=False)

def is_real(imgPath):
    ''' 
        Call to know if an image is real or graphic

        params:
            imgPath: Path of image to be tested

        returns:
            Boolean: True=Real
                     False=Graphic
    '''
    image = Image.open(imgPath)
    prediction = model.classify_one_image(image)
    if prediction == 0:
        return True
    else:
        return False

def are_real(imgPathList):
    ''' 
        Call to know if a list of images are real or graphic

        params:
            imgPathList: List of paths of images to be tested

        returns:
            List of boolean: (Same order as images in the passed list)
                    True=Real
                    False=Graphic
    '''
    images = []
    results= []

    for imgPath in imgPathList:
        image = Image.open(imgPath)
        images.append(image)
    
    predictions = model.classify_list_of_images(images)
    
    for p in predictions:
        if p == 0:
            results.append(True)
        else:
            results.append(False)
    
    return results

def are_real_URLS(imgURLList):
    ''' 
        Call to know if a list of images (URLS) are real or graphic

        params:
            imgURLList: List of URLS of images to be tested

        returns:
            List of boolean: (Same order as images in the passed list)
                    True=Real
                    False=Graphic
    '''
    images = []
    results= []

    for url in imgURLList:
        image = Image.open(requests.get(url, stream=True).raw)
        images.append(image)
    
    predictions = model.classify_list_of_images(images)
    
    for p in predictions:
        if p == 0:
            results.append(True)
        else:
            results.append(False)
    
    return results