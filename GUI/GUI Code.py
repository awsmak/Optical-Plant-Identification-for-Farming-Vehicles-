# importing required libraries
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import os
import sys
import time
import subprocess
from subprocess import *

import cv2
import numpy as np


# Main window class
class MainWindow(QMainWindow):

    # constructor
    def __init__(self):
        super().__init__()

        # setting geometry
        self.setGeometry(100, 100,
                         800, 600)

        # setting style sheet
        self.setStyleSheet("background : lightgrey;")

        # getting available cameras
        self.available_cameras = QCameraInfo.availableCameras()

        # if no camera found
        if not self.available_cameras:
            # exit the code
            sys.exit()

        # creating a status bar
        self.status = QStatusBar()

        # setting style sheet to the status bar
        self.status.setStyleSheet("background : white;")

        # adding status bar to the main window
        self.setStatusBar(self.status)

        # path to save
        self.save_path = ""

        # creating a QCameraViewfinder object
        self.viewfinder = QCameraViewfinder()

        # showing this viewfinder
        self.viewfinder.show()

        # making it central widget of main window
        self.setCentralWidget(self.viewfinder)

        # Set the default camera.
        self.camera_selction(0)

        # creating a tool bar
        toolbar = QToolBar("Camera Tool Bar")

        # adding tool bar to main window
        self.addToolBar(toolbar)

        # creating start button
        self.btn_action = QAction("Start")

        # adding status tip
        self.btn_action.setStatusTip("This will Start the system")

        # adding tool tip
        self.btn_action.setToolTip("Starting")

        # adding action
        self.btn_action.triggered.connect(self.start)

        # adding to toolbar
        toolbar.addAction(self.btn_action)

        # creating stop button
        self.btn1_action = QAction("Stop")

        # adding status tip
        self.btn1_action.setStatusTip("This will stop the system")

        # adding tool tip
        self.btn1_action.setToolTip("Stoping")

        # adding action
        self.btn1_action.triggered.connect(self.stop)

        # adding to toolbar
        toolbar.addAction(self.btn1_action)

        # creating a photo action to take photo
        self.click_action = QAction("Click photo")

        # adding status tip to the photo action
        self.click_action.setStatusTip("This will capture picture")

        # adding tool tip
        self.click_action.setToolTip("Capture picture")

        # adding action to it
        # calling take_photo method
        self.click_action.triggered.connect(self.click_photo)

        # adding this to the tool bar
        toolbar.addAction(self.click_action)

        # creating a combo box for selecting camera
        self.camera_selector = QComboBox()

        # adding status tip to it
        self.camera_selector.setStatusTip("Choose camera to take pictures")

        # adding tool tip to it
        self.camera_selector.setToolTip("Select Camera")
        self.camera_selector.setToolTipDuration(2500)

        # adding items to the combo box
        self.camera_selector.addItems([camera.description()
                                       for camera in self.available_cameras])

        # adding action to the combo box
        # calling the select camera method
        self.camera_selector.currentIndexChanged.connect(self.camera_selction)

        # adding this to tool bar
        toolbar.addWidget(self.camera_selector)

        # setting tool bar stylesheet
        toolbar.setStyleSheet("background : white;")

        # setting window title
        self.setWindowTitle("Plant Identification(Rumex Accetosa)")

        # showing the main window
        self.show()

    # method to select camera
    def camera_selction(self, i):
        # getting the selected camera
        self.camera = QCamera(self.available_cameras[i])

        # setting view finder to the camera
        self.camera.setViewfinder(self.viewfinder)

        # setting capture mode to the camera
        self.camera.setCaptureMode(QCamera.CaptureStillImage)

        # if any error occur show the alert
        self.camera.error.connect(lambda: self.alert(self.camera.errorString()))

        # start the camera
        self.camera.start()

        # creating a QCameraImageCapture object
        self.capture = QCameraImageCapture(self.camera)

        # showing alert if error occur
        self.capture.error.connect(lambda error_msg, error,
                                          msg: self.alert(msg))

        # when image captured showing message
        self.capture.imageCaptured.connect(lambda d,
                                                  i: self.status.showMessage("Image captured : "
                                                                             + str(self.save_seq)))

        # getting current camera name
        self.current_camera_name = self.available_cameras[i].description()

        # inital save sequence
        self.save_seq = 0

    # method to take photo
    def click_photo(self):
        # time stamp
        timestamp = time.strftime("%d-%b-%Y-%H_%M_%S")

        # capture the image and save it on the save path
        self.capture.capture(os.path.join(self.save_path,
                                          "%s-%04d-%s.jpg" % (
                                              self.current_camera_name,
                                              self.save_seq,
                                              timestamp
                                          )))

        # increment the sequence
        self.save_seq += 1

    # method to start the system
    def start(self):
        self.btn_action.setEnabled(False)
        self.click_action.setEnabled(False)
        # python script for detection algorithm
        import yolo_object_detection

        # import yolo_object_detection_video

        # call({"python", "yolo_object_detection_video.py"})

    # method to stop the system
    def stop(self):
        self.btn_action.setEnabled(True)
        self.click_action.setEnabled(True)
        quit()


# Driver code
if __name__ == "__main__":
    # create pyqt5 app
    App = QApplication(sys.argv)

    # create the instance of the Window
    window = MainWindow()

    # start the app
    sys.exit(App.exec())
