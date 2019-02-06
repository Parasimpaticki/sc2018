from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def start(videoName):
	#Ucitavanje MNIST skupa podataka koji cemo koristit za treniranje klasifikatora
	#Za prepoznavanje cifara
	#print(bcolors.HEADER +"Ucitavanje MNIST dataset-a..."+bcolors.ENDC)
	#Sajt im ne radi
	#Skinuti manuelno sa: https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat
	#I sacuvati u ~/scikit_learn_data/mldata/
	dataset = fetch_mldata('MNIST original')
	trainData = dataset.data
	#print(bcolors.OKBLUE+"Data loaded!"+bcolors.ENDC)	

	#print(bcolors.HEADER+"Priprema training set-a..."+bcolors.ENDC)
	#Train data
	train = dataset.data[:60000]
	#Train labels
	train_labels = dataset.target[:60000]
	#Test data
	test = dataset.data[60000:]
	#Test labels
	test_labels = dataset.target[60000:]
	test_labels_sample = test_labels[::100]
	#print(bcolors.HEADER+"Treniranje modela..."+bcolors.ENDC)

	model = KNeighborsClassifier(n_neighbors=4, algorithm='brute').fit(train, train_labels)
	#print(bcolors.HEADER+"Provjera modela..."+bcolors.ENDC)
	test_sample = test[::100]
	test_sample[:,299].mean()
	#tacnost = model.score(test_sample, test_labels_sample)
	'''
	if(tacnost >0.96):
		print(bcolors.OKGREEN+"Tacnost: "+str(tacnost)+bcolors.ENDC)
	else:
		print(bcolors.FAIL+"Tacnost: "+str(tacnost)+bcolors.ENDC)
	'''

	sum_of_nums=0
	#Ucitavanje videa
	#frejm po frejm
	frame_num=0
	cap = cv2.VideoCapture(videoName)
	#indeksiranje frejmova
	cap.set(1,frame_num)
	testBaza=[]

	#analiza frejm po frejm
	while True:
		frame_num += 1
		ret_val, frame = cap.read()

		#ako frejm nije zahvacen
		if not ret_val:
			break
		
		#Detekcija presjeka
		# create an image filled with zeros, single-channel, same size as img.
		blank = np.zeros( frame.shape[0:2] )
		detekcijaPresjeka_linija = blank.copy()
		detekcijaPresjeka_broj = blank.copy()
		#frame - predstavlja trenutni frejm
		#HOG
		#Prvo prebaciti u gray
		frame_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)	
		#GaussianBlur
		kernel_size = 5
		frame_blur = cv2.GaussianBlur(frame_gray,(kernel_size, kernel_size),0)
		#Edge detection
		low_threshold = 50
		high_threshold = 150
		edges = cv2.Canny(frame_blur, low_threshold, high_threshold)
		threshold = 15  # minimum number of votes (intersections in Hough grid cell)
		minLineLength = 100  # minimum number of pixels making up a line
		maxLineGap  = 10  # maximum gap in pixels between connectable line segments	
		lines = cv2.HoughLinesP(edges,1,np.pi/180,100,threshold,minLineLength,maxLineGap)
		for x1,y1,x2,y2 in lines[0]:
			cv2.line(detekcijaPresjeka_linija,(x1+3,y1-3),(x2-3,y2-3),(255,255,255),1)

		#Detect numbers
		ret,thresh = cv2.threshold(frame_gray,160,255,cv2.THRESH_TOZERO)
		im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE )
		novaBaza=[]
		for c in contours:
			novi=True
			(x, y, w, h) = cv2.boundingRect(c)
			for cont,dalJeSabran in testBaza:
				(x2, y2, w2, h2) = cv2.boundingRect(cont)
				razlika = abs(x-x2) + abs(y-y2) + abs(w-w2) + abs(h-h2)
				if(razlika < 20):
					novi=False
					novaBaza.append((c,dalJeSabran))
					break
			if(novi):
				novaBaza.append((c,False))
		testBaza=novaBaza
		digit_size = 19
		novaBaza=[]
		for c,dalJeSabran in testBaza:
			(x, y, w, h) = cv2.boundingRect(c)
			preskocitBroj=True
			if w > digit_size or h > digit_size:			
				detekcijaPresjeka_broj = blank.copy()
				#Da se ne bi duplirali skratim malo sliku gore lijevo
				cv2.rectangle(detekcijaPresjeka_broj, (x+2,y+2), (x+w,y+h), (255, 255, 255), 1)
				intersection = np.logical_and( detekcijaPresjeka_linija, detekcijaPresjeka_broj )
				if(np.count_nonzero(intersection)>0):
					if(dalJeSabran==False):			
						preskocitBroj=False
						novaBaza.append((c,True))
			if(preskocitBroj):
				novaBaza.append((c,dalJeSabran))
				continue
			else:
				croped_img = thresh[y-5:y+h+5,x-5:x+w+5]
				digit_to_be_predicted = cv2.resize(croped_img,(28,28), interpolation = cv2.INTER_NEAREST).reshape(1, -1)
				prediction = model.predict(digit_to_be_predicted)
				broj = int(prediction[0])
				sum_of_nums+=broj
				cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
				cv2.putText(frame,str(broj),(x+15,y+15),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),2)
		testBaza=novaBaza

	cap.release()
	print(str(sum_of_nums))
	return sum_of_nums

if __name__=="__main__":
	start(sys.argv[1])
