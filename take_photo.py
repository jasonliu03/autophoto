#!coding=utf-8
import cv2
import time
#import pydiagnosis
import numpy as np

from PIL import Image, ImageDraw, ImageFont

import requests
import json
from glob import glob
from concurrent.futures import ThreadPoolExecutor
import json

#from predict import predictGender

save_path = './face/'
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_tongue.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

camera = cv2.VideoCapture(0) # 参数0表示第一个摄像头
#camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#camera = cv2.VideoCapture("vedio4.avi") # 参数0表示第一个摄像头

# 判断视频是否打开
if (camera.isOpened()):
    print('Open')
else:
    print('摄像头未打开')

i = 0
faceRect = []
b_gender_on = False
b_faceMatch = True
rst_gender = -1
rst_faceVerify = -1

IP = "192.168.1.102"

def putText(img, text, position, fillColor):
    # 图像从OpenCV格式转换成PIL格式
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, text, fill=fillColor)
    img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
    return img

def getEmd(imgName):
    url = "http://" + IP + ":5000" + '/api/photos/getEmdDirect'
    f= open(imgName, 'rb')
    files = {'photo': f}
    res = requests.post(url, files=files)
    f.close()
    rst_emd= json.loads(res.text).get('status')
    #rst_emd = bytes(rst_emd, encoding='utf-8')
    emd = json.loads(rst_emd)
    fw = open(imgName+'.txt', 'w', encoding='utf-8')
    json.dump(emd, fw)
    fw.close()

imgNames = [e for e in glob('./pics/*.jpg')]
matchList = []
start = time.time()
for index, imgName in enumerate(imgNames):
    getEmd(imgName)
end = time.time()
print("time_getEmds:", end-start)

while True:
    grabbed, frame_lwpCV = camera.read() # 读取视频流
    #frame_lwpCV = cv2.resize(frame_lwpCV, (360, 270))
    gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY) # 转灰度图

    if not grabbed:
        break
    
    # 人脸检测部分
    #start = time.time()
    #faces = face_cascade.detectMultiScale(gray_lwpCV, 1.3, 5)
    #end = time.time()
    #print("origin time:{}".format(end-start))
    #for face in faces:
    #    x = face[0]
    #    y = face[1]
    #    w= face[2]
    #    h= face[3]
    #    cv2.rectangle(frame_lwpCV, (x, y), (x + w, y +h), (255, 0, 0), 2)
    #    cv2.imshow('lwpCVWindow', frame_lwpCV)
    #    cv2.waitKey(1)
    #cv2.imshow('lwpCVWindow', frame_lwpCV)
    #cv2.waitKey(1)
    #continue

    img_encode = cv2.imencode('.jpg', frame_lwpCV)[1]
    data_encode = np.array(img_encode)
    str_encode = data_encode.tostring()
    start = time.time()
    #result = pydiagnosis.autoPhotoTongue(str_encode)
    #result = pydiagnosis.glassDetect(str_encode)

    cv2.imwrite('tmp4web.jpg', frame_lwpCV) 
    photo = './tmp4web.jpg'
    url = "http://" + IP + ":5000" + '/api/photos/autoPhoto'
    files = {'photo': open(photo, 'rb')}
    res = requests.post(url, files=files)
    end = time.time()
    print("algorithm time: {}".format(end-start))

    print("res.text:", res.text)
    result = {}

    result['status'] = json.loads(res.text).get('status')
    result['x_point'] = json.loads(res.text).get('x_point')
    result['y_point'] = json.loads(res.text).get('y_point')
    result['width'] = json.loads(res.text).get('width')
    result['height'] = json.loads(res.text).get('height')
    print("result:", result)
    d = int(result['status'])
    x = int(result['x_point'])
    y = int(result['y_point'])
    w = int(result['width'])
    h = int(result['height'])
    #d = x = y = w = h = 10
    if x!=0 or y!=0 or w!=0 or h!=0:
        cv2.rectangle(frame_lwpCV, (x, y), (x + w, y +h), (255, 0, 0), 2)
        if rst_gender == 0:
            cv2.rectangle(frame_lwpCV, (x, y), (x + w, y +h), (0, 255, 0), 2)
            cv2.putText(frame_lwpCV,u'man',(x, y),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
        elif rst_gender == 1:
            cv2.rectangle(frame_lwpCV, (x, y), (x + w, y +h), (0, 0, 255), 2)
            cv2.putText(frame_lwpCV,u'woman',(x, y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),1)
            
        if rst_faceVerify != -1:
            cv2.putText(frame_lwpCV,str(rst_faceVerify),(x, y),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)

        roi_frame_lwpCV = frame_lwpCV[y:y + h, x:x + w] # 检出人脸区域后，取上半部分，因为眼睛在上边啊，这样精度会高一些
        faceImg = cv2.resize(roi_frame_lwpCV, (224, 224))
        cv2.imwrite('tmpGender.jpg', faceImg) # 将检测到的人脸写入文件
        faceRectIndex1 = 0
        faceRectIndex2 = 49
        faceDistance = 50
        absDistance = 20
        minDistance = 100
        maxDistance = 325
        
        cv2.rectangle(frame_lwpCV, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if d == 1:
            frame_lwpCV = putText(frame_lwpCV, u"Closer", (w // 2 + x, y - h // 8), (0,0,255))
            faceRect = []
        elif d == 2:
            frame_lwpCV = putText(frame_lwpCV, u"Back OFF", (w // 2 + x, y - h // 8), (0,0,255))
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
                        frame_lwpCV = putText(frame_lwpCV, u"Keep Still, Photoing...", (0, 0), (255,0,0))
                        cv2.imshow('lwpCVWindow', frame_lwpCV)
                        cv2.waitKey(1000)

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
            photo = './tmpGender.jpg'
            url = "http://" + IP + ":5000" + '/api/photos/genderDetect'
            files = {'photo': open(photo, 'rb')}
            res = requests.post(url, files=files)
            print("res_gender.text:", res.text)
            rst_gender = json.loads(res.text).get('status')
            #rst_gender = predictGender('tmpGender.jpg')
            print("rst_gender:", rst_gender)

        # use backend pics    
    #    if b_faceMatch and i % 5 == 0:
    #        photo = './tmpGender.jpg'
    #        url = "http://" + IP + ":5000" + '/api/photos/faceVerify'
    #        files = {'photo': open(photo, 'rb')}
    #        res = requests.post(url, files=files)
    #        print("res_faceVerify.text:", res.text)
    #        rst_faceVerify = json.loads(res.text).get('status')
    #        print("rst_faceVerify:", rst_faceVerify)
    #        
         # use frontend pics
        def takeSecond(elem):
            return elem[1]
        def doMatch(index, imgName):
            photo = './tmpGender.jpg'
            url = "http://" + IP + ":5000" + '/api/photos/faceVerify'
            f1 = open(photo, 'rb')
            f2= open(imgName, 'rb')
            files = {'photo01': f1, 'photo02':f2}
            res = requests.post(url, files=files)
            f1.close()
            f2.close()
            rst_faceVerify = json.loads(res.text).get('status')
            matchList.append((index, rst_faceVerify, imgName.split('\\')[-1].split('.')[0]))
 
        def doMatchByEmd(index, imgName):
            url = "http://" + IP + ":5000" + '/api/photos/faceVerifyEmd'
            f1 = open('tmpGender.jpg.txt', 'r', encoding='utf-8')
            f2= open(imgName, 'r', encoding='utf-8')
            emd1 = json.load(f1)
            emd2 = json.load(f2)
            body = {'photo01': emd1, 'photo02':emd2}
            res = requests.post(url, data=json.dumps(body))
            f1.close()
            f2.close()
            rst_faceVerify = json.loads(res.text).get('status')
            matchList.append((index, rst_faceVerify, imgName.split('\\')[-1].split('.')[0]))

        if b_faceMatch and i % 5 == 0:
            imgNames = [e for e in glob('./pics/*.txt')]
            matchList = []
            start = time.time()
            #with ThreadPoolExecutor(3) as executor:
            #    for index, imgName in enumerate(imgNames):
            #        executor.submit(doMatch, index, imgName)
            getEmd('./tmpGender.jpg')
            for index, imgName in enumerate(imgNames):
                #doMatch(index, imgName)
                doMatchByEmd(index, imgName)
            end = time.time()
            print("time_match:", end-start)
            if matchList:
                matchList.sort(key=takeSecond)
                print("matchList", matchList)
                threshold = 1.5
                if matchList[0][1] < threshold:
                    rst_faceVerify = matchList[0][2] + ":" + str(matchList[0][1])
    
        i += 1
        
    cv2.imshow('lwpCVWindow', frame_lwpCV)
    key = cv2.waitKey(1) & 0xFF
    # 按'q'健退出循环
    if key == ord('q'):
        break
# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()
