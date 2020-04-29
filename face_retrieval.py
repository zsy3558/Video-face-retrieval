import time

import cv2
import os
import numpy as np


def detect_face(img):
    face_CascadeClassifier = cv2.CascadeClassifier(
        'C:/Users/zhang/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data'
        '/haarcascade_frontalface_default.xml')
    faces = face_CascadeClassifier.detectMultiScale(img, scaleFactor=1.18, minNeighbors=3)  # Detect face
    if len(faces) == 0:
        # print("?")
        return None, []
    # print(faces[0])
    x, y, w, h = faces[0]  # The x and y coordinates of the face area
    return img[y:y + h, x:x + w], faces[0]


# This function will read all training images, detect faces from each image, and save the successfully detected faces
def Batch_detect_face(path):
    train_imgs = os.listdir(path)
    faces = []
    lables = []
    for img_name in train_imgs:
        img_path = path + "/" + img_name
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        detected_img, loc = detect_face(img)
        save_name = "./detected_img/" + img_name
        if not os.path.exists("./detected_img/"):
            os.makedirs("./detected_img/")
        #print(loc)
        if len(loc) != 0:
            cv2.imwrite(save_name, detected_img)
            faces.append(detected_img)
            lables.append(0)
        else:
            print(img_name)
    return faces, lables


def video_retrieval(train_img_path, video_path):
    train_faces, labels = Batch_detect_face(train_img_path)
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(train_faces, np.array(labels))
    vc = cv2.VideoCapture(video_path)
    fourcc = int(vc.get(cv2.CAP_PROP_FOURCC))
    fps = int(vc.get(cv2.CAP_PROP_FPS))
    size = (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter('output.flv', cv2.VideoWriter_fourcc('F','L','V','1'), fps, size)
    is_capturing, frame = vc.read()
    save_name = "./result/"
    if not os.path.exists("./result/"):
        os.makedirs("./result/")
    i = 0
    while is_capturing:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_CascadeClassifier = cv2.CascadeClassifier(
            'C:/Users/zhang/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data'
            '/haarcascade_frontalface_default.xml')
        '''eyes_CascadeClassifier = cv2.CascadeClassifier('C:/Users/zhang/AppData/Local/Programs/Python/Python37/Lib'
                                                       '/site-packages/cv2/data/haarcascade_eye.xml')'''
        faces_loc = face_CascadeClassifier.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=3)
        if len(faces_loc) != 0 :
            for (x,y,w,h) in faces_loc:
                '''eyes_loc = eyes_CascadeClassifier.detectMultiScale(gray_frame[y:y + h, x:x + w], scaleFactor=1.2,
                                                                   minNeighbors=3)
                if len(eyes_loc)!=0:'''
                id, probability = face_recognizer.predict(gray_frame[y:y+h, x:x+w])
                #print(id, probability)
                if probability < 100 :
                    name = "target"#name = "trump P=" + str(100-probability)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                else:
                    name = "Unknown"#name = "Unknown p=" + str(100-probability)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x + 5, y - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2, color= (255,255,255))
        #cv2.imwrite(save_name+ str(i) + ".jpg", frame)
        #i+=1
        out.write(frame)
        is_capturing, frame = vc.read()
    vc.release()
    out.release()


def main():
    start = time.clock()
    train_img_path = "./train_img"
    video_path = "./165988417-1-192.mp4"
    video_retrieval(train_img_path, video_path)
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)


if __name__ == '__main__':
    main()
