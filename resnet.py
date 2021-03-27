from imageai.Detection import ObjectDetection
import os
import imageai
import keras
import tensorflow

#execution_path=os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(r"C:\Users\omkar desai\OneDrive\Desktop\Artificial Inteligence\resnet50_coco_best_v2.0.1.h5")
detector.loadModel()

a=detector.CustomObjects(person=True)
detections = detector.detectCustomObjectsFromImage(a,r'C:\Users\omkar desai\OneDrive\Desktop\Artificial Inteligence\image1.jpg',r'C:\Users\omkar desai\OneDrive\Desktop\Artificial Inteligence\image3.jpg')

for i in detections:
    print(i["name"],":",i["percentage_probability"])

b=detector.CustomObjects(bicycle=True)
detections = detector.detectCustomObjectsFromImage(b,r'C:\Users\omkar desai\OneDrive\Desktop\Artificial Inteligence\image2.jpg',r'C:\Users\omkar desai\OneDrive\Desktop\Artificial Inteligence\image4.jpg')

for i in detections:
    print(i["name"],":",i["percentage_probability"])



'''
There are 80 possible objects that you can detect with the
ObjectDetection class, and they are as seen below.

    person,   bicycle,   car,   motorcycle,   airplane,
    bus,   train,   truck,   boat,   traffic light,   fire hydrant,   stop_sign,
    parking meter,   bench,   bird,   cat,   dog,   horse,   sheep,   cow,   elephant,   bear,   zebra,
    giraffe,   backpack,   umbrella,   handbag,   tie,   suitcase,   frisbee,   skis,   snowboard,
    sports ball,   kite,   baseball bat,   baseball glove,   skateboard,   surfboard,   tennis racket,
    bottle,   wine glass,   cup,   fork,   knife,   spoon,   bowl,   banana,   apple,   sandwich,   orange,
    broccoli,   carrot,   hot dog,   pizza,   donot,   cake,   chair,   couch,   potted plant,   bed,
    dining table,   toilet,   tv,   laptop,   mouse,   remote,   keyboard,   cell phone,   microwave,
    oven,   toaster,   sink,   refrigerator,   book,   clock,   vase,   scissors,   teddy bear,   hair dryer,
    toothbrush.
    
    To detect only some of the objects above, you will need to call the CustomObjects function and set the name of the
object(s) yiu want to detect to through. The rest are False by default. In below example, we detected only chose detect only person and dog.
"""
custom = detector.CustomObjects(person=True, dog=True)
