from keras.applications.nasnet import NASNetLarge
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input


from keras.applications import Xception

from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import cv2
import os

# Поиск по категориям в ImageNet: http://www.image-net.org/search?q=cat&rank=popularity
# Для прогрузки модели поменять путь до папки с картинками для обработки и выбрать архитектуру. 
# Коты по датасету ImageNet: 'tabby', 'tabby_cat', 'tiger_cat', 'Persian_cat', 'Siamese_cat', 'Siamese', 'Egyptian_cat'


MODELS = {
    "vgg16": VGG16,
    #"vgg19": VGG19,
    #"inception": InceptionV3,
    #"xception": Xception,
    #"resnet": ResNet50,
    #"NASNet": NASNetLarge,
    #"InceptionResNetV2": InceptionResNetV2
}

cats_labels = ['tabby', 'tabby_cat', 'tiger_cat', 'Persian_cat', 'Siamese_cat', 'Siamese', 'Egyptian_cat']
dogs_labels = ['Newfoundland', 'Newfoundland_dog', 'Eskimo_dog', 'husky', 'dalmatian', 'coach_dog', 
               'carriage_dog', 'German_shepherd', 'German_shepherd_dog', 'German_police_dog', 'alsatian',
               'African_hunting_dog', 'hyena_dog', 'Cape_hunting_dog', 'Lycaon_pictus',
               'dogsled', 'dog_sled', 'dog_sleigh']


image_paths = []
folderpath = '/Users/user/Desktop/dogs_vgg16/no_dogs/'


for dirpath, dirnames, filenames in os.walk(folderpath):
    for filename in filenames:
        image_paths.append(os.path.join(dirpath, filename))


for model_name, model in zip(list(MODELS.keys()), MODELS.keys()):
    if model in ("inception", "xception", "InceptionResNetV2"):
        inputShape = (299, 299)
        preprocess = preprocess_input
    elif model in ("NASNet"):
        inputShape = (331, 331)
        preprocess = preprocess_input
    else:
        inputShape = (224, 224)
        preprocess = imagenet_utils.preprocess_input

    Network = MODELS[model] # to recheck
    model = Network(weights="imagenet")

    counter = 0
    for image_path in image_paths:
        #print("[INFO]: Loading and preprocessing image...")
        try:
            image = load_img(image_path, target_size=inputShape)
            counter += 1
        except:
            continue
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess(image)

        # classify the image
        #print("[INFO]: classifying image with {}".format(model))
        preds = model.predict(image)
        
        P = imagenet_utils.decode_predictions(preds)

        #print(model_name, image_path)
        target_sum = 0
        for (i, (imagenetID, label, prob)) in enumerate(P[0]):
            print(image_path, (imagenetID, label, prob))
            if label in dogs_labels:
                print(i, '= yes')
                target_sum += 1
                break
            else:
                print(i, '= no')
                target_sum += 0

        new_label = image_path.split('/')[0:-1]
        print(new_label)
        if target_sum > 0:
            new_label.append('__{}_1.jpg'.format(counter))
            new_label = '/'.join(new_label)
            os.rename(image_path, new_label)
        else:
            new_label.append('__{}_0.jpg'.format(counter))
            new_label = '/'.join(new_label)
            os.rename(image_path, new_label)    # в случае обычного переименования до <имя_файла>_0

            # если нужно отфильтровать выборку
            #try:
            #    os.remove(image_path)
            #except:
            #    os.rename(image_path, new_label)
