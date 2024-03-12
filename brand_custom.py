import argparse
import time
from pathlib import Path
import cv2
import os
import sys
import xlwt
from xlwt import Workbook
import torch
import torch.backends.cudnn as cudnn
from numpy import random
__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(__dir__)
sys.path.append(__dir__)
sys.path.insert(0,__dir__)
from Thingtrax.PaddleOCR_change.paddleocr import PaddleOCR
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def detect(save_img=False):
    source, weights,save_txt, imgsz, trace  = opt.source, opt.weights,opt.save_txt, opt.img_size, not opt.no_trace
    

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    brand_dict ={}
    missed_flag = 150
    wb = Workbook()
    sheet1 = wb.add_sheet("Mall Walk")
    sheet1.write(0,0,"Shop Name")
    sheet1.write(0,1,"OCR Output")
    sheet1.write(0,2,"1st Time Frame")
    sheet1.write(0,3,"2nd Time Frame")
    sheet1.write(0,4,"3rd Time Frame")
    fps = 60
    low_list = ["SH","SanCha"]
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    fr = 0

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    names =  ['Andrea-s', 'Armani Exchange', 'Bath & Body Works', '7 Bazaari', 'Brooks Brothers', 'Calvin Klien', 'Gant', 'Kiehl-s', 'Levis', 'Maison Des Parfums', 'Mango', 'Raymond', 'Reebok', 'Reebok_logo', 'SBI', 'Shahnaz Husain', 'SSBeauty', 'San Cha Tea Boutique', 'Scotch and Soda', 'Starbucks', 'The gloss box', 'Tim Hortons', 'United Colors of Benetton', 'US polo', 'Versace', 'Versace_logo', 'Zara', 'Aldo', 'Apple', 'Bodyshop', 'cafe al shalom', 'Dior', 'Estee Lauder', 'Ethos', 'fabIndia', 'Harajuku', 'Harajuku_logo', 'Jo Malone', 'Kimirica', 'Mac', 'Oma', 'Scentido', 'Sketchers', 'Swarovski', 'Toofaced', 'Watches']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        class_list =[]
        fr = fr +1 # Current frame
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()


        # Process detections
        print (pred)
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()


                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]}'
                    color=colors[int(cls)]
                    x1=int(xyxy[0].item())
                    y1=int(xyxy[1].item())
                    x2=int(xyxy[2].item())
                    y2=int(xyxy[3].item())
                    xm = int (x1 + (x2 -x1)/2)
                    ym = int (y1 + (y2 - y1)/2)
                    if label  not in low_list and conf < 0.7:
                        continue
                    length = x2-x1
                    #print(label)
                    if "_" not in label:
                        class_list.append(label)
                        cropped_img = im0[y1:y2,x1:x2]
                        result_temp = ocr.ocr(cropped_img,cls=True)
                        #print(result,type(result[0][0][0]))
                        if result_temp[0] != None:
                            result_temp=result_temp[0]
                            if len(result_temp)>1:
                                area_list=[]
                                for string in result_temp:
                                    coord = string[0] 
                                    sx1 = min(coord[0][0],coord[3][0])
                                    sy1 = min(coord[0][1],coord[1][1])
                                    sx2 = max(coord[1][0],coord[2][0])
                                    sy2 = max(coord[2][1],coord[3][1])
                                    area=((sx2-sx1)*(sy2-sy1))
                                    area_list.append(area)
                                if area_list[0] > area_list[1]:
                                    result = (result_temp[0][-1])
                                elif area_list[0] < area_list[1]:
                                    result = (result_temp[1][-1])
                                #print(result, len(result))                  
                            else:
                              result=result_temp[0][-1]
                            #print("result: "+str(result))
                        elif result_temp[0] == None:
                            result = result_temp

                    elif "_" in label:
                        cropped_img = None
                        result = ["logo"]
                        label = label.split("_")[0]
                        class_list.append(label)
                    #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                    if label not in brand_dict:
                        brand_dict[label] = {"Brand":label,"length":length,"OCR_string":result,1:{"start_frame":fr,"last_frame":fr,"missed_frame":0,"detected": 1}}
                    elif label in brand_dict:
                        # Updating last frame
                        index = len(brand_dict[label]) - 3

                        if brand_dict[label][index]["missed_frame"] > missed_flag: 
                            if brand_dict[label][index]["detected"] < 20:
                                brand_dict[label][index] = {"start_frame":fr,"last_frame":fr,"missed_frame":0,"detected": 1}
                            elif brand_dict[label][index]["detected"] >= 20:
                                brand_dict[label][index+1] = {"start_frame":fr,"last_frame":fr,"missed_frame":0,"detected": 1}

                        new_index = len(brand_dict[label]) - 3
                        brand_dict[label][new_index]["last_frame"] = fr
                        brand_dict[label][new_index]["detected"] = brand_dict[label][new_index]["detected"] +1

                        # missed Frame check
                        if brand_dict[label][new_index]["missed_frame"] < missed_flag:
                            brand_dict[label][new_index]["missed_frame"] = 0                         

                        # Updating OCR string    
                        if brand_dict[label]["OCR_string"][0] == None or brand_dict[label]["OCR_string"][0] == "logo":
                            brand_dict[label]["OCR_string"]= result
                        if result[0]=="logo"or result[0]==None:
                            continue
                        else:
                            #print(result)
                            if type(result[0]) == str:
                                if len((result[0].strip()).lower()) == len((label.lower()).strip()):
                                    #print("String",result[0])
                                    """if len(brand_dict[label]["OCR_string"][0]) < len(result[0]):
                                        brand_dict[label]["OCR_string"] = result"""
                                    #if len(brand_dict[label]["OCR_string"][0]) == len(result[0]):
                                    if float(brand_dict[label]["OCR_string"][1]) < float(result[1]):
                                        brand_dict[label]["OCR_string"] = result
                                """else:
                                    if len(brand_dict[label]["OCR_string"][0]) < len(result[0]) and (len((result[0].strip()).lower()) < len((label.lower()).strip())) :
                                        brand_dict[label]["OCR_string"] = result
                                    if len(brand_dict[label]["OCR_string"][0]) == len(result[0]):
                                        if float(brand_dict[label]["OCR_string"][1]) < float(result[1]):
                                            brand_dict[label]["OCR_string"] = result"""


        #print (class_list, brand_dict)
        for key,value in brand_dict.items():
            index = len(brand_dict[key])-3
            if key not in class_list:
                brand_dict[key][index]["missed_frame"]= brand_dict[key][index]["missed_frame"] +1

                




    print(brand_dict)
    num = 0
    for key,value in brand_dict.items():
        num = num +1
        #print("The Brand {} dict size: {} ".format(key,len(brand_dict[key])))
        if len(brand_dict[key]) >= 4:
            if brand_dict[key][1]["detected"] > 20:
                first = str(float("{:.2f}".format((brand_dict[key][1]["start_frame"])/60)))+" - "+str(float("{:.2f}".format((brand_dict[key][1]["last_frame"])/60)))
            else:
                first = None
            if len(brand_dict[key]) == 4:
                second,third = None,None
        if len(brand_dict[key]) >= 5:
            if brand_dict[key][2]["detected"] > 20:
                second =  str(float("{:.2f}".format((brand_dict[key][2]["start_frame"])/60)))+" - "+str(float("{:.2f}".format((brand_dict[key][2]["last_frame"])/60)))
            else:
                second = None
            if len(brand_dict[key]) == 5:
                third = None
        if len(brand_dict[key]) >= 6:
            if brand_dict[key][3]["detected"] > 20:
                third =  str(float("{:.2f}".format((brand_dict[key][3]["start_frame"])/60)))+" - "+str(float("{:.2f}".format((brand_dict[key][3]["last_frame"])/60)))
            else:
                third = None
          
        #print(first,second,third)



        sheet1.write(num,0,key)
        sheet1.write(num,1,brand_dict[key]["OCR_string"][0])
        sheet1.write(num,2,first)
        sheet1.write(num,3,second)
        sheet1.write(num,4,third)


        sheet1.write
    wb.save("Mall_walk.xls")








         


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))
    detect()
