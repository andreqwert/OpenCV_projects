# Наработки по OpenCV (OpenCV developments)



1. `human_detection` - обнаружение людей на фотографии (human detection);       
2. `images_scanner` - нахождение листов бумаги с помощью детектора границ Canny (Canny edge detector);      
3. `simple_tracking` - слежение на видео за синими предметами круглой формы (tracking for the round objects);         
4. `meanshift` - сдвиг окна для нахождения лица человека в видеопотоке (meanshift for human detection on the videostream);     
5. `find_brightest` - нахождение самой яркой области изображения (image brightest area detection);               
6. **`unet` - семантическая сегментация свёрточной нейронной сетью** (semantic segmentation with CNN);       
7. `facial_landmark` - нахождение лиц на изображениях при помощи dlib (faces detection with dlib);          
8. `hamming` - проверка на идентичность изображений с помощью расстояния Хэмминга (similarity calculation with Hamming distance);
9. **`face_detection` - детекция лиц в видеопотоке для определения их пола и примерного возраста** (human faces detection and identification)

-----

`examples.ipynb` - некоторые базовые алгоритмы, средни них:      
- развороты/повороты изображений, афинные преобразования над изображениями (flipping, rotating and Affine transforms);    
- построение линий, эллипсов, кругов и прочего на изображениях (drawing circles, ellipses on the images);      
- SIFT;     
- FAST;      
- детектор углов Харриса (Harris edge detector);       
- Чтение, сохранение и поиск объектов с цветом заданного диапазона на видео (reading, writing and search for the objects with specific colour on the video);
- Feature matching;
- детектор границ Canny (Canny edge detector);
- каскад Хаара (Haar cascade);
- box-фильтрация (box-filtration);
- cropping, изменение разрешения и сохранение видео (cropping, video resizing and writing).
