from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imutils
import time
import sys
import os
class Ui_Detection(object):
    def alertmsg(self, title, Message):
        self.warn = QtWidgets.QMessageBox()
        self.warn.setIcon(QtWidgets.QMessageBox.Information)
        self.warn.setWindowTitle(title)
        self.warn.setText(Message)
        self.warn.setStandardButtons(QtWidgets.QMessageBox.Ok)
        self.warn.exec_()
    def choose(self):
        fileName,_ = QtWidgets.QFileDialog.getOpenFileName(None, "Select File", "*")
        self.lineEdit.setText(fileName)
    def detection(self):
        try:
            filename = self.lineEdit.text()
            if filename== "null" or filename=="":
                self.alertmsg("Failed","Please select the File First")
            elif filename.endswith(".jpg"):
                image = cv2.imread(filename)
                classes = None
                with open('../DetectionCounting/models/coco.names', 'r') as f:
                    classes = [line.strip() for line in f.readlines()]
                    print(classes)

                Width = image.shape[1]
                Height = image.shape[0]

                # read pre-trained model and config file
                net = cv2.dnn.readNet('../DetectionCounting/models/yolov3.cfg',
                                      '../DetectionCounting/models/yolov3.weights')

                # create input blob
                # set input blob for the network
                net.setInput(cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False))

                # run inference through the network
                # and gather predictions from output layers

                layer_names = net.getLayerNames()
                output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
                outs = net.forward(output_layers)

                class_ids = []
                confidences = []
                boxes = []

                # create bounding box
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.1:
                            center_x = int(detection[0] * Width)
                            center_y = int(detection[1] * Height)
                            w = int(detection[2] * Width)
                            h = int(detection[3] * Height)
                            x = center_x - w / 2
                            y = center_y - h / 2
                            class_ids.append(class_id)
                            confidences.append(float(confidence))
                            boxes.append([x, y, w, h])

                indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)

                # check if is people detection
                person = 0
                for i in indices:

                    i = i[0]
                    box = boxes[i]
                    if class_ids[i] == 0:
                        person = person + 1
                        label = str(classes[class_id])
                        cv2.rectangle(image, (round(box[0]), round(box[1])),
                                      (round(box[0] + box[2]), round(box[1] + box[3])), (51, 204, 51), 2)
                        cv2.putText(image, label, (round(box[0]) - 10, round(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4, (0, 0, 255), 1)
                cv2.putText(image, "Total Persons :" + str(person), (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255), 2)
                self.alertmsg("Successfull", "Person detection and Counting Successfull from Image")
                cv2.imshow("../DetectionCounting/testimages/output.jpg", image)
                cv2.waitKey(0)
            elif filename.endswith(".mp4"):
                labelsPath = ("../DetectionCounting/models/coco.names")
                LABELS = open(labelsPath).read().strip().split("\n")
                print(LABELS)
                np.random.seed(42)
                COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                                           dtype="uint8")
                weightsPath = ("../DetectionCounting/models/yolov3.cfg")
                configPath = ("../DetectionCounting/models/yolov3.weights")
                print("[INFO] loading YOLO from disk...")
                net = cv2.dnn.readNetFromDarknet(weightsPath, configPath)
                ln = net.getLayerNames()
                ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
                vs = cv2.VideoCapture(filename)
                writer = None
                (W, H) = (None, None)
                try:
                    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                        else cv2.CAP_PROP_FRAME_COUNT
                    total = int(vs.get(prop))
                    print("[INFO] {} total frames in video".format(total))
                except:
                    print("[INFO] could not determine # of frames in video")
                    print("[INFO] no approx. completion time can be provided")
                    total = -1
                while True:
                    (grabbed, frame) = vs.read()
                    if not grabbed:
                        break
                    if W is None or H is None:
                        (H, W) = frame.shape[:2]
                    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                                 swapRB=True, crop=False)
                    net.setInput(blob)
                    start = time.time()
                    layerOutputs = net.forward(ln)
                    end = time.time()

                    boxes = []
                    confidences = []
                    classIDs = []

                    for output in layerOutputs:

                        for detection in output:

                            scores = detection[5:]
                            classID = 0 #np.argmax(scores)
                            confidence = scores[classID]

                            if confidence > 0.5:

                                box = detection[0:4] * np.array([W, H, W, H])
                                (centerX, centerY, width, height) = box.astype("int")

                                x = int(centerX - (width / 2))
                                y = int(centerY - (height / 2))

                                boxes.append([x, y, int(width), int(height)])
                                confidences.append(float(confidence))
                                classIDs.append(classID)

                    idxs = cv2.dnn.NMSBoxes(boxes, confidences, float(0.5), float(0.3))


                    if len(idxs) > 0:

                        for i in idxs.flatten():

                            (x, y) = (boxes[i][0], boxes[i][1])
                            (w, h) = (boxes[i][2], boxes[i][3])


                            color = [int(c) for c in COLORS[classIDs[i]]]
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                            text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                                       confidences[i])
                            cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            cv2.putText(frame, "Total Persons :" + str(len(idxs)), (40, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                        1.0,(0, 0, 255), 2)

                    # check if the video writer is None
                    if writer is None:
                        # initialize our video writer
                        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                        writer = cv2.VideoWriter("../DetectionCounting/testvideo/output1.mp4", fourcc, 30,
                                                 (frame.shape[1], frame.shape[0]), True)

                        # some information on processing single frame
                        if total > 0:
                            elap = (end - start)
                            print("[INFO] single frame took {:.4f} seconds".format(elap))
                            print("[INFO] estimated total time to finish: {:.4f}".format(
                                elap * total))

                    # write the output frame to disk
                    writer.write(frame)

                # release the file pointers
                print("[INFO] cleaning up...")
                writer.release()
                vs.release()
                self.alertmsg("sucess","Person detection and counting successfull from video")
                cap = cv2.VideoCapture('../DetectionCounting/testvideo/output1.mp4')
                # Check if camera opened successfully
                if (cap.isOpened() == False):
                    print("Error opening video  file")
                # Read until video is completed
                while (cap.isOpened()):
                    # Capture frame-by-frame
                    ret, frame = cap.read()
                    if ret == True:
                        # Display the resulting frame
                        cv2.imshow('Frame', frame)
                        # Press Q on keyboard to  exit
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                    else:
                        break
                cap.release()
                # Closes all the frames
                cv2.destroyAllWindows()
            else:
                self.alertmsg("Failed","Please Select the proper file")

        except Exception as e:
            print("Line",e)
            tb_lineno = sys.exc_info()
            print(tb_lineno)


    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(751, 309)
        Dialog.setStyleSheet("QDialog{background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(51, 178, 241, 218));}")
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setGeometry(QtCore.QRect(40, 80, 681, 171))
        self.frame.setStyleSheet("QFrame{background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(215, 215, 215, 219));}")
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame.setLineWidth(2)
        self.frame.setObjectName("frame")
        self.lineEdit = QtWidgets.QLineEdit(self.frame)
        self.lineEdit.setGeometry(QtCore.QRect(20, 40, 491, 41))
        self.lineEdit.setObjectName("lineEdit")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(20, 10, 311, 31))
        self.label.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:1 rgba(58, 194, 55, 0));\n"
"font: 11pt \"Arial\";")
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.frame)
        self.pushButton.setGeometry(QtCore.QRect(510, 39, 128, 43))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.pushButton.setFont(font)
        self.pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.frame)
        self.pushButton_2.setGeometry(QtCore.QRect(240, 100, 181, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(9)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_2.setStyleSheet("font: 75 14pt \"Arial\";\n"
"background-color: rgb(176, 70, 99);\n"
"color: rgb(255, 255, 255);")
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(170, 20, 501, 41))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        self.pushButton.clicked.connect(self.choose)
        self.pushButton_2.clicked.connect(self.detection)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Select File For Detection"))
        self.pushButton.setText(_translate("Dialog", "Choose"))
        self.pushButton_2.setText(_translate("Dialog", "Detect"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Detection()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
