# USAGE
# python C:\Users\user\Desktop\Real-Time-Object-Detection-master\real_time_object_detection.py --prototxt C:\Users\user\Desktop\MobileNetSSD_deploy.prototxt.txt --model C:\Users\user\Desktop\MobileNetSSD_deploy.caffemodel


from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat","chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor","mobilephone"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

""" bağımsız değişkeni oluşturmak, bağımsız değişkenleri çözümlemek ve ayrıştırmak """
def agr_parser():
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--prototxt", required=True,
		help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-m", "--model", required=True,
		help="path to Caffe pre-trained model")
	ap.add_argument("-c", "--confidence", type=float, default=0.2,
		help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())
	return args

""" serileştirdiğimiz modeli diskten yükledik."""
def serializing_model(args):
	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
	return net
"""video akışını başlatın, kamera sensörünün ısınmasına izin verin ve FPS sayacını başlatın."""
def initilizing_video_stream():
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
	fps = FPS().start()
	return vs, fps

"""kareyi akıtılan video akışından alın ve maksimum genişliği 400 piksel olacak şekilde yeniden boyutlandırın. """
def preprocess_video_frames(vs):
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# convert frame into a blob
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)
	return blob, frame

"""blob'u ağ üzerinden geçirin ve tespitleri ve tahminleri alın."""
def get_detections_from_frames(blob, net):
	net.setInput(blob)
	detections = net.forward()
	return detections

""""güven" i sağlayarak zayıf tespitleri filtreleyin.
minimum güvenden daha büyük."""
def filter_out_detections(args, detections, confidence, key):
	if confidence > args["confidence"]:
		idx = int(detections[0, 0, key, 1])
	else:
		idx = None
	return idx

"""Filtrelenmiş algılamalar için sınırlayıcı kutular çizin"""
def draw_bounding_box(detections, frame, key):
	(h, w) = frame.shape[:2]
    #sınırlayıcı kutular için koordinatlar alın
	box = detections[0, 0, key, 3:7] * np.array([w, h, w, h])
	return box

"""Tespitler için etiket alın"""
def predict_class_labels(confidence, idx):
	label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
	return label

"""tahmini çerçeveye çiz"""
def draw_predictions_on_frames(box, label, frame, idx):
	(startX, startY, endX, endY) = box.astype("int")
	cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
	y = startY - 15 if startY - 15 > 15 else startY + 15
	cv2.putText(frame, label, (startX, y),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
	return frame


def main():
    #Gerçek zamanlı nesne tespiti için argümanların ayrıştırılması
	args = agr_parser()

    #Önceden eğitilmiş modeli seri hale getirme
	net = serializing_model(args)

    #Video akışını kare olarak başlatma (fps - saniyedeki kare sayısı)
	vs, fps = initilizing_video_stream()

    #video akışındaki karelerin üzerinden geç
	while True:
        #Algılamaları elde etmek için çerçeveleri önceden koruma
		blob, frame = preprocess_video_frames(vs)

        #Her çerçeveden tespitlerin alınması
		detections = get_detections_from_frames(blob, net)
		
        # algılamaların üzerinden geçmek
		for key in np.arange(0, detections.shape[2]):
            #tahminle ilişkili güveni çıkarmak
			confidence = detections[0, 0, key, 2]

            # zayıf tespitleri filtreleyin
			idx = filter_out_detections(args, detections, confidence, key)

			if(idx == None):
				break
			else:
                #Algılamalar için sınırlayıcı kutuları alın
				box = draw_bounding_box(detections, frame, key)
				
                #Kutular için tahminler alın
				label = predict_class_labels(confidence, idx)

                #Çerçevelere tahminler çizin
				frame = draw_predictions_on_frames(box, label, frame, idx)

		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		# update the FPS counter
		fps.update()

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()

if __name__ == '__main__':
	main()
