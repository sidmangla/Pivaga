Approach:

Since it's a very crowded and fast-paced environment I have used the Yolov7 model for object detection

Pre-trained model links:

yolov7 : https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt

yolov7-tiny: https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov7-tiny.weights


Trained model link:

yolov7 best: https://drive.google.com/file/d/1-kfHV-7VTgJE-yh9mJ8qsRF4c4KPq2Fr/view?usp=sharing

yolov7-tiny best: https://drive.google.com/file/d/1-QJCAIxEdFham5y7CgEA0KMntxuvQfTI/view?usp=sharing


For Paddle OCR: used pre-trained model for english language

  Detection_model - https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar
  
  Reacognition model - https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar
  
  Text Direction classification model - https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
  
  charaacter dictionary path - > PaddleOCR/ppocr/utils/en_dict.txt


  
video file : youtube link -> https://youtu.be/-QsCM4ddr-g?si=3BMbWyYDTsecLRPV  (The first six minutes)

          : google link 
                6 mins video: https://drive.google.com/file/d/1tdEMMwTJiPDqQeCdLCg3n-x1aNOHMypL/view?usp=sharing
                
                40 secs video: https://drive.google.com/file/d/1mJqZXn5LpowArfyussl9FC6X3E3xWEKP/view?usp=sharing
 
ouput video link -> https://drive.google.com/file/d/1Dw4ekEaWwQvaDTUpAJ90WLllZR5woY_m/view?usp=sharing

Files used:

yolov7/extract.py - > to extract images from videos for annotation purpose

yolov7/no_annotation.py -> to create empty txt files for images having no annotations

yolov7/brand_custom.py -> File which reads the video file, detects the brands names and logo using the trained model. Once we get the coordinates from the detections made. we pass the cropped images to paddle OCR
                          Paddle OCR then gives the output using pre trained models mentioned above.
                          We create a dictionary of brands and append the fist frame, last frame, missed frame for the corresponding brand
                          then we create a excel file and write the information in the excel file.
                          
Excel file -> **mall_walk.xls** (link: https://github.com/sidmangla/Pivaga/blob/master/Mall_walk_brand.xls)


**Challenges:**
1. Accuracy: Using Yolov7 improved a lot of Accuracy.

Params: These parameters increased the accuracy further changing the inference size from 640 to 960 improved accuracy. Wanted to multi-scale training but it was running out of memory as inference sizes in multi-scale were going to 1300 and son on.

Batch:16

epochs:400

inference size: 960 x 960

Precision: 0.912

mAP(0.5): 0.962

mAP(0.5-0.9): 0.745 

2. Different lighting conditions and different angles of brands and logo
   **solution**: Used Augmentation by tweaking Saturation, Brightness, and contrast of images by 25%. Therefore mitigating the light issues. Also flipped, and rotated the images to counter different angles of the logos and brand names
   
4. Occlusions: 
**Solution**: Occlusions were handled by annotating images where there were heavy occlusions and the usage of Yolov7 also helped.

5. Large number of classes:   
**Solution**: First I trained on yolov7-tiny but since our model consisted of 46 classes we were not able to get good accuracy. Therefore I had to train the yolov7 model to achieve better accuracy.

6. Scalability: The model is easily scalable as most of the occurrences of the brands were annotated. (refer to images)
   
7. Cost-effective: The fps without writing the video was quite good. got 13 fps while testing. Therefore this detection system can be deployed on edge devices and would not require heavy GPUs
                    It utilized only 1.5 GB RAM while testing. (user-friendly).

**Command:** !python3 brand_custom.py --weights yolov7/trained/v7_best.pt --img 960 --conf 0.4 --source yolov7/videos/input/city_6mins.mp4
Note: For a couple of classes the confidence **0.4** was used, for the rest 44 classes **0.7** confidence was used

**The accuracy of the Detection model was good and with some processing, I was able to get multiple timings of occurrence and disappearance of the shops from the video with very high accuracy. On the other hand was not able to train a**
**custom model for Paddle OCR therefore there is a scope for improvement in that area**

**Future Work: We can capture far away text like the brand Fab India was not recognized at all, but could be recognized with the implementation of the Super Resolution model or custom training the OCR**


