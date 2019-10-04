import dlib
import cv2
from scipy.spatial import distance


# creating models' objects
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

# proccessing the first picture
image_1 = cv2.imread('elon_before.jpg')
faces_1 = detector(image_1, 1)

shape = sp(image_1, faces_1[0])
face_descriptor_1 = facerec.compute_face_descriptor(image_1, shape)

# proccessing the second picture
image_2 = cv2.imread('elon_after.jpg')
faces_2 = detector(image_2, 1)
shape = sp(image_2, faces_2[0])
face_descriptor_2 = facerec.compute_face_descriptor(image_2, shape)

# calculating euclidean distance
result = distance.euclidean(face_descriptor_1, face_descriptor_2)

if result < 0.6:
    print('Most likely there is one person')
else:
    print('Most likely there are two different people')
