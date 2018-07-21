# Детекция лиц в видеопотоке

Создан симбиоз методов - в качестве алгоритма детекции лиц используется dlib, в качестве определния пола и возраста используется multitask convolutional neural network (MTCNN).       
Алгоритм выдает 80% точности при определнии пола и 60% (на глаз) при определнии возраста.      

## Запуск         
1. Подгрузить веса в папку `pretrained_models`: https://www.dropbox.com/s/rf8hgoev8uqjv3z/weights.18-4.06.hdf5
2. `python2 main.py`      
Для запуска необходим tensorflow для Python версии 2.

## Бонус
Research показал, что хорошие результаты выдаёт также вот этот алгоритм: https://github.com/dpressel/rude-carnie/
