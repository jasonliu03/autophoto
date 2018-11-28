#!coding=utf-8
import cv2
import time
import pydiagnosis
import numpy as np

from PIL import Image, ImageDraw, ImageFont

from predict import predictGender

save_path = './face/'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#camera = cv2.VideoCapture(0) # 参数0表示第一个摄像头
camera = cv2.VideoCapture("vedio5.avi") # 参数0表示第一个摄像头

# 判断视频是否打开
if (camera.isOpened()):
    print('Open')
else:
    print('摄像头未打开')

i = 0
faceRect = []
b_gender_on = False
rst_gender = -1

def putText(img, text, position, fillColor, font=ImageFont.truetype('NotoSansCJK-Black.ttc', 40)):
    # 图像从OpenCV格式转换成PIL格式
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, text, font=font, fill=fillColor)
    img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
    return img

while True:
    start = time.time()
    grabbed, frame_lwpCV = camera.read() # 读取视频流
    gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY) # 转灰度图

    if not grabbed:
        break
    end = time.time()

    # 人脸检测部分
    #faces = face_cascade.detectMultiScale(gray_lwpCV, 1.3, 5)


    img_encode = cv2.imencode('.jpg', frame_lwpCV)[1]
    data_encode = np.array(img_encode)
    str_encode = data_encode.tostring()
    result = pydiagnosis.autoPhoto(str_encode)
    print("result:", result)
    rst = result.strip().split(',')
    d = int(rst[0])
    x = int(rst[1])
    y = int(rst[2])
    w = int(rst[3])
    h = int(rst[4])
    if x!=0 and y!=0 and w!=0 and h!=0:
        #cv2.rectangle(frame_lwpCV, (x, y), (x + w, y +h), (255, 0, 0), 2)
        #if rst_gender == 0:
        #    cv2.rectangle(frame_lwpCV, (x, y), (x + w, y +h), (0, 255, 0), 2)
        #    cv2.putText(frame_lwpCV,u'man',(x, y),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
        #elif rst_gender == 1:
        #    cv2.rectangle(frame_lwpCV, (x, y), (x + w, y +h), (0, 0, 255), 2)
        #    cv2.putText(frame_lwpCV,u'woman',(x, y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),1)
        #    
        #roi_frame_lwpCV = frame_lwpCV[y:y + h, x:x + w] # 检出人脸区域后，取上半部分，因为眼睛在上边啊，这样精度会高一些
        #faceImg = cv2.resize(roi_frame_lwpCV, (224, 224))
        #cv2.imwrite('tmpGender.jpg', faceImg) # 将检测到的人脸写入文件
        faceRectIndex1 = 0
        faceRectIndex2 = 49
        faceDistance = 50
        absDistance = 20
        minDistance = 100
        maxDistance = 325
        
        cv2.rectangle(frame_lwpCV, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if d == 1:
            frame_lwpCV = putText(frame_lwpCV, u"请靠近一点", (w // 2 + x, y - h // 8), (0,0,255))
            faceRect = []
        elif d == 2:
            frame_lwpCV = putText(frame_lwpCV, u"请后退一点", (w // 2 + x, y - h // 8), (0,0,255))
            faceRect = []
        else: 
            if len(faceRect) == faceDistance:
                faceRect.pop(0)
                faceRect.append([x,y,w,h])
            else:
                faceRect.append([x,y,w,h])
            if len(faceRect) == faceDistance:
                flag = True
                for r in faceRect:
                    if not r:
                        flag = False
                        pass
                if not flag:
                    pass
                else:
                    x1 = faceRect[faceRectIndex1][0]
                    y1 = faceRect[faceRectIndex1][1]
                    w1 = faceRect[faceRectIndex1][2]
                    h1 = faceRect[faceRectIndex1][3]
                    x9 = faceRect[faceRectIndex2][0]
                    y9 = faceRect[faceRectIndex2][1]
                    w9 = faceRect[faceRectIndex2][2]
                    h9 = faceRect[faceRectIndex2][3]
                    if abs(x1-x9) > absDistance or abs(y1-y9) > absDistance:
                        pass
                    elif abs(w1-w9) > absDistance or abs(h1-h9) > absDistance:
                        pass
                    else:
                        frame_lwpCV = putText(frame_lwpCV, u"请保持姿态，自动抓拍中...", (0, 0), (255,0,0))
                        cv2.imshow('lwpCVWindow', frame_lwpCV)
                        cv2.waitKey(3000)

                        faceRect = []
            else:
                pass

        #roi_frame_lwpCV = frame_lwpCV[y:y + h, x:x + w] # 检出人脸区域后，取上半部分，因为眼睛在上边啊，这样精度会高一些
        #faceImg = cv2.resize(roi_frame_lwpCV, (224, 224))
        #
        #img_encode = cv2.imencode('.jpg', faceImg)[1]
        #data_encode = np.array(img_encode)
        #str_encode = data_encode.tostring()
        
        if b_gender_on and i % 3 == 0:
            rst_gender = predictGender('tmpGender.jpg')
            print("rst_gender:", rst_gender)
            
        i += 1
        
    cv2.imshow('lwpCVWindow', frame_lwpCV)
    key = cv2.waitKey(1) & 0xFF
    # 按'q'健退出循环
    if key == ord('q'):
        break
# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()
