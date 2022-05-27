import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

with_mask = np.load('with_mask.npy')
without_mask = np.load('without_mask.npy')

# print(with_mask.shape)
# print(without_mask.shape)

with_mask = with_mask.reshape(200, 50 * 50 * 3)
without_mask = without_mask.reshape(200, 50 * 50 * 3)

# print(with_mask.shape)

x = np.r_[with_mask, without_mask]
# print(x.shape)

labels = np.zeros(x.shape[0])
labels[200:] = 1.0

names = {0: 'Mask', 1: 'No mask'}

x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.25)

# print(x_train.shape)
pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train)

# print(x_train[0])
# print(x_train.shape)

x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.30)
svm = SVC()
svm.fit(x_train, y_train)

# x_test = pca.transform(x_test)#not needed
y_pred = svm.predict(x_test)

# print(accuracy_score(y_test, y_pred))

haar_data = cv2.CascadeClassifier(
    "C:\haar-cascade-files-master\haarcascade_frontalface_default.xml")
data = []
font = cv2.FONT_HERSHEY_COMPLEX
capture = cv2.VideoCapture(0)
while True:
    flag, img = capture.read()
    if flag:

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = haar_data.detectMultiScale(gray)
        # print(faces)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (50, 50, 255), (4))
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50, 50))
            face = face.reshape(1, -1)
            # face = pca.transform(face)
            pred = svm.predict(face)
            n = names[int(pred)]
            cv2.putText(img, n, (x, y), font, 1, (240, 0, 255), (4))
            print(n)
            # cv2.putText(img, "Faces Detected:{} ".format(
            #     len(faces)), (x, y+200), font, 1, (255, 0, 0), (4))

    cv2.imshow('result', img)
    if cv2.waitKey(2) == 27:
        break


cv2.destroyAllWindows()
