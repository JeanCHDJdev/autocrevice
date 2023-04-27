import cv2
import numpy as np
import keras
import matplotlib.pyplot as plt

model = keras.models.load_model("./Model", compile = False)
model.compile()



def detect(file, threshold = 0.0019):
    x, y = 0, 0
    img = cv2.imread(file)
    clear = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

    width = len(img[0])
    height = len(img)
    while x + 32 < width:
        while y + 32 < height:
            img_part = clear[y:y+32, x:x+32]
            recon_part = model(img_part[np.newaxis, :,  :, :])
            recon_part = np.array(recon_part)
            for i in range(32):
                for j in range(32):
                    if (np.sum(np.square(img_part[i][j]-recon_part[0][i][j])/(3*255*255))) > threshold:
                        img[y+i][x+j] = [0, 0, 255]

            y = y + 32
        x = x + 32
        y = 0
    
    cv2.imshow("reconstruted", img)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'):  # wait for 's' key to save and exit
        cv2.imwrite("detected.png", img)
        cv2.destroyAllWindows()
    return(img)


def compare(file_detect, file_verif, threshold = 0.0019):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    verif = cv2.imread(file_verif)
    detected = detect(file_detect, threshold)
    result1 = np.copy(detected)
    result2 = np.copy(detected)
    if np.shape(verif) != np.shape(detected):
        raise Exception("Files do not match, can't verify the accuracy")
    width = len(detected[0])
    height = len(detected)
    for i in range(height):
        for j in range(width):
            if (verif[i][j] == np.array([255, 0, 0])).all() and (detected[i][j] == np.array([0, 0, 255])).all():
                TP += 1
                result1[i][j] = [0, 255, 0]
                result2[i][j] = [0, 255, 0]
            elif (verif[i][j] != np.array([255, 0, 0])).all() and (detected[i][j] == np.array([0, 0, 255])).all():
                FP += 1
                result1[i][j] = [0, 0, 255]
                result2[i][j] = [0, 0, 255]
            elif (verif[i][j] == np.array([255, 0, 0])).all() and (detected[i][j] != np.array([0, 0, 255])).all():
                FN += 1
                result1[i][j] = [255, 0, 0]
                result2[i][j] = [255, 0, 0]
            elif (verif[i][j] != np.array([255, 0, 0])).all() and (detected[i][j] != np.array([0, 0, 255])).all():
                TN += 1
                result2[i][j] = [0, 255, 255]
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = 2*P*R / (P + R)
    A = (TP + TN) / (TP + TN + FP + FN)
    print("Precision : ", P)
    print("Recall rate : ", R)
    print("F1 score : ", F1)
    print("Accuracy rate : ", A)
    #cv2.imshow("result1", result1)
    #cv2.waitKey(0)
    #cv2.imshow("result2", result2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return (P, R, F1, A)


TH = [i / 10000 for i in range(15, 26)]
P_list = []
R_list = []
F1_list = []
A_list = []
for threshold in TH:
    P, R, F1, A = compare("./crevasse4.jpg", "./crevasse4theorie.png", threshold)
    P_list.append(P)
    R_list.append(R)
    F1_list.append(F1)
    A_list.append(A)

figure, axis = plt.subplot(2, 2)
axis[0, 0].plot(TH, P_list)
axis[0, 0].set_title("Precision")
axis[0, 1].plot(TH, R_list)
axis[0, 1].set_title("Recall rate")
axis[1, 0].plot(TH, F1_list)
axis[1, 0].set_title("F1 score")
axis[1, 1].plot(TH, A_list)
axis[1, 1].set_title("Accuracy")

plt.show()