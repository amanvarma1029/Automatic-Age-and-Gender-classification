
import cv2 as cv
import math
import time
import argparse
import playsound
def EAR(drivereye):
	point1 = dist.euclidean(drivereye[1], drivereye[5])
	point2 = dist.euclidean(drivereye[2], drivereye[4])
	distance = dist.euclidean(drivereye[0], drivereye[3])
	# compute the eye aspect ratio
	ear_aspect_ratio = (point1 + point2) / (2.0 * distance)
	return ear_aspect_ratio

def MOR(drivermouth):
	# compute the euclidean distances between the horizontal
	point	= dist.euclidean(drivermouth[0], drivermouth[6])
	# compute the euclidean distances between the vertical
	point1	= dist.euclidean(drivermouth[2], drivermouth[10])
	point2	= dist.euclidean(drivermouth[4], drivermouth[8])
	# taking average
	Ypoint	 = (point1+point2)/2.0
	# compute mouth aspect ratio
	mouth_aspect_ratio = Ypoint/point
	return mouth_aspect_ratio


	
	

def getFaceBox(net, frame, conf_threshold=0.7):
	frameOpencvDnn = frame.copy()
	frameHeight = frameOpencvDnn.shape[0]
	frameWidth = frameOpencvDnn.shape[1]
	blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

	net.setInput(blob)
	detections = net.forward()
	bboxes = []
	for i in range(detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > conf_threshold:
			x1 = int(detections[0, 0, i, 3] * frameWidth)
			y1 = int(detections[0, 0, i, 4] * frameHeight)
			x2 = int(detections[0, 0, i, 5] * frameWidth)
			y2 = int(detections[0, 0, i, 6] * frameHeight)
			bboxes.append([x1, y1, x2, y2])
			cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
	return frameOpencvDnn, bboxes


parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')

args = parser.parse_args()

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)

cap = cv.VideoCapture(args.input if args.input else 0)
padding = 20
while cv.waitKey(1) < 0:
	# Read frame
	t = time.time()
	hasFrame, frame = cap.read()
	###################
	if not hasFrame:
		cv.waitKey()
		break

	frameFace, bboxes = getFaceBox(faceNet, frame)
	if not bboxes:
		print("No face Detected, Checking next frame")
		continue

	for bbox in bboxes:
		face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

		blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
		genderNet.setInput(blob)
		genderPreds = genderNet.forward()
		gender = genderList[genderPreds[0].argmax()]
		print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

		ageNet.setInput(blob)
		agePreds = ageNet.forward()
		age = ageList[agePreds[0].argmax()]
		print("Age Output : {}".format(agePreds))
		print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

		label = "{},{}".format(gender, age)
		cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
		cv.imshow("Age Gender Demo", frameFace)