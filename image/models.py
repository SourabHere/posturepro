# from django.db import models
from distutils.command.upload import upload
from email.policy import default
from io import BytesIO
from json import load
from sqlite3 import Timestamp
from django.db import models
from django.forms import IntegerField, JSONField
from django.core.files.base import ContentFile
from keras.models import load_model
import easyocr

from PIL import Image
import math
import cv2 
import numpy as np
import face_recognition
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def gray(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    return img

def faceDetection(img):
    faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_alt.xml")
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(imgGray,1.1,4)
    res='no face detected'
    if faces == ():
        return None,res
    for (x,y,w,h) in faces:
        res='face detected'
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    return img,res

def face_extract(img):
    face_cas=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_alt.xml")

    face=face_cas.detectMultiScale(img,1.3,5)

    if face == ():
        return None
    arr=[]
    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face=img[y:y+h,x:x+w]
        arr.append([x,y,w,h])
    return cropped_face,arr



def face_extract_mood(imginp):
    img=imginp
    moods={1:"happy",2:"angry",3:"sad",4:"fear"}
    model=load_model('image/emotionnew.h5') 
    output=""
    face_cas=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_alt.xml")

    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=face_cas.detectMultiScale(imgGray,1.1,4)

    if face == ():
        return imginp,"No Mood Detected"

    for (x,y,w,h) in face:
        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face=img[y:y+h,x:x+w]

        faces=cv2.resize(cropped_face,(150,150))
        rgb=cv2.cvtColor(faces,cv2.COLOR_BGR2RGB)

        im=Image.fromarray(rgb,"RGB")
        img_arr=np.array(im)

        img_arr=np.expand_dims(img_arr,axis=0)
        img_arr=np.vstack([img_arr])
        pred=model.predict(img_arr)
        pred=np.argmax(pred)
        prediction=moods[pred+1]
        output+=" "+prediction

        # cv2.putText(imginp, prediction , (x, y+20), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)

    return imginp,output

def tedxt_extract(img):
    reader=easyocr.Reader(['en'],gpu=True)
    result=reader.readtext(img)
    arr=[]

    for j in range (len(result)-1):
        top_left=(result[j][0][0])
        bottom_right=(result[j][0][2])

        print(result[j])
        if float(result[j][0][0][0])==int(result[j][0][0][0]):
            # print(str(result[j][1]))
            arr.append(str(result[j][1]))

    return arr

        




# def Gender(imginp):
#     img1=imginp
#     model=load_model("image\gender240.h5")
#     output=""
#     labels=["Female","Male"]

#     img1=cv2.resize(img1,[240,240])
#     img1=np.array([img1])
#     pred=np.argmax(model.predict(img1))
#     output=labels[np.argmax(pred)]

#     return imginp,output
    
    



def face_match(img1,img2):
    image1=img1
    imgTest=img2
    # image1 = face_recognition.load_image_file(img1)
    image1=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
    # imgTest=face_recognition.load_image_file(img2)
    imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

    faceLoc=face_recognition.face_locations(image1)[0]
    encodeim1=face_recognition.face_encodings(image1)[0]
    # cv2.rectangle(image1,(faceLoc[3],faceLoc[4]),(faceLoc[1]),faceLoc[2],(0,0,255),2)

    faceLoc2=face_recognition.face_locations(imgTest)[0]
    encodeim2=face_recognition.face_encodings(imgTest)[0]
    # cv2.rectangle(imgTest,(faceLoc2[3],faceLoc2[4]),(faceLoc2[1]),faceLoc2[2],(0,0,255),2)

    results=face_recognition.compare_faces([encodeim1],encodeim2)
    resultDis=face_recognition.face_distance([encodeim1],encodeim2)
    return img1,str(results)+" "+str(resultDis)

def posed(image):
    i,mood=face_extract_mood(image)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    res='cannot detect pose'
    def calculate_shoulder_angle(landmarks):
        if len(landmarks) >= 13:
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]

            # Calculate the angle between the shoulders
            angle_radians = abs(left_shoulder.x - right_shoulder.x) / abs(left_shoulder.y - right_shoulder.y)
            angle_degrees = round(angle_radians * (180.0 / 3.14159265359), 2)
            print(angle_degrees)
            return angle_degrees
        return None
    
    def calculate_angle(vector1, vector2):
        dot_product = sum(x * y for x, y in zip(vector1, vector2))
        magnitude1 = math.sqrt(sum(x ** 2 for x in vector1))
        magnitude2 = math.sqrt(sum(y ** 2 for y in vector2))
        cosine_angle = dot_product / (magnitude1 * magnitude2)
        angle = math.degrees(math.acos(cosine_angle))
        return angle
    
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if results.pose_landmarks is not None:
        
        landmarks = results.pose_landmarks.landmark
        
        nose_index = 0
        shoulder2_index = 14
        nose = landmarks[nose_index]
        shoulder1 = landmarks[11]
        shoulder2 = landmarks[12]
        
        vector_nose_shoulder1 = [shoulder1.x - nose.x, shoulder1.y - nose.y]
        vector_nose_shoulder2 = [shoulder2.x - nose.x, shoulder2.y - nose.y]
        angle_nose_shoulder1 = calculate_angle(vector_nose_shoulder1, [1, 0])
        angle_nose_shoulder2 = calculate_angle(vector_nose_shoulder2, [1, 0])

        if abs(angle_nose_shoulder1) < 80 and abs(angle_nose_shoulder2 - 60) < 80 and (angle_nose_shoulder1) > 40 and (angle_nose_shoulder2-60) > 40:
            print("Both angles are approximately 60 degrees.")
            res='sitting straight'
        else:
            print("One or both angles are not approximately 60 degrees.")
            res='Sit Straight a bit'
                # print(f"Angle between shoulders: {angle_degrees} degrees")
                
        for landmark in landmarks:
            height, width, _ = image.shape
            cx, cy = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1) 
                
        height, width, _ = image.shape
        left_shoulder_x = int(shoulder1.x * width)
        left_shoulder_y = int(shoulder1.y * height)
        right_shoulder_x = int(shoulder2.x * width)
        right_shoulder_y = int(shoulder2.y * height)
        nose_1 = int(nose.x * width)
        nose_2 = int(nose.y * height)

        cv2.circle(image, (left_shoulder_x, left_shoulder_y), 5, (255,0, 0), -1)  # Red dot for left shoulder
        cv2.circle(image, (right_shoulder_x, right_shoulder_y), 5, (255,0, 0), -1)  # Red dot for right shoulder
        cv2.circle(image, (nose_1, nose_2), 5, (0, 0, 255), -1)  # Red dot for right shoulder
    
    j,face=faceDetection(image)
        
    print(mood)
    return image,res,str(mood),str(face)
    # with mp_holistic.Holistic(
    #     min_detection_confidence=0.5,
    #     min_tracking_confidence=0.5) as holistic:

       
    #     image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        
    #     image.flags.writeable = False
    #     results = holistic.process(image)
    #     if results.pose_landmarks is not None:
    #         landmarks = results.pose_landmarks.landmark
    #         left_shoulder = landmarks[11]
    #         right_shoulder = landmarks[12]

    #         # Calculate the angle between the shoulders
    #         angle_radians = abs(left_shoulder.x - right_shoulder.x) / abs(left_shoulder.y - right_shoulder.y)
    #         angle_degrees = round(angle_radians * (180.0 / 3.14159265359), 2)

    #         print(f"Angle between shoulders: {angle_degrees} degrees")
        
    #     image.flags.writeable = True
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        
    #     image=cv2.flip(image,1)
    #     return image
    

# Create your models here.
class images(models.Model):
    img_id=models.AutoField(primary_key=True)
    image=models.FileField(upload_to="images/imagedata",default="")
    image2=models.FileField(upload_to="images/imagedata",default="")
    img_name=models.CharField(max_length=50)
    
        
    prediction_face=models.CharField(max_length=50,default="")
    prediction_mood=models.CharField(max_length=50,default="")
    prediction_pose=models.CharField(max_length=50,default="")
    choice=models.IntegerField(default=0)

    def __str__(self):
        return self.img_name
      
    def save(self,*args,**kwargs):
        open_img=Image.open(self.image)

        cv2_img=np.array(open_img)

        predict=""
        img=cv2_img
        # img=gray(cv2_img)
        if self.choice==0:

            img=faceDetection(cv2_img)
        elif self.choice==1:
            open_img2=Image.open(self.image2)
            cv2_img2=np.array(open_img2)
            # img,predict=face_match(open_img,open_img2)
            img,predict=face_match(cv2_img,cv2_img2)

        elif self.choice==2:
            img,predict=face_extract_mood(cv2_img)
        elif self.choice==3:
            predict=tedxt_extract(cv2_img)
        # elif self.choice==4:
        #     img,predict=Mask(cv2_img)
        elif self.choice==5:
            img,pose,mood,face=posed(img)
        elif self.choice==6:
            pass
        

        close_img=Image.fromarray(img)

        buffer=BytesIO()
        close_img.save(buffer,format='png')
        image_png=buffer.getvalue()

        self.prediction_pose=pose
        self.prediction_mood=mood
        self.prediction_face=face
        self.image.save(str(self.image),ContentFile(image_png),save=False)

        super().save(*args,**kwargs)