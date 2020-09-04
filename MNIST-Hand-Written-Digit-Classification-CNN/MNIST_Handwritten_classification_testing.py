#@title  { display-mode: "form" }
#@markdown Predict
# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
from termcolor import colored
import cv2
from gtts import gTTS
from IPython.display import Audio
from scipy.io import wavfile
from pydub import AudioSegment

# load and prepare the image
def load_image(filename):
	# load the image
	img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
	img = cv2.resize(img,(28,28),interpolation=cv2.INTER_LINEAR)
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img


def playSound(mytext):
	language = 'en'
	myobj = gTTS(text=mytext, lang=language, slow=False)
	myobj.save("welcome.mp3") 

	sound = AudioSegment.from_mp3("welcome.mp3")
	sound.export("welcome.wav", format="wav")

	data = wavfile.read('welcome.wav')
	framerate = data[0]
	sounddata = data[1]
	time      = np.arange(0,len(sounddata))/framerate

	# calc.plot11(time,sounddata,"Sound data loaded from wav file","Time (s)","Values")

	Audio(sounddata,rate=framerate,autoplay=True)

# load an image and predict the class
def run_example():
	
	# load the image
	img = load_image('drawing.png')
	
	# load model
	model = load_model("final_model.h5")
	
	# predict the class
	digit = model.predict_classes(img)

	print(colored("I think you have drawn : {}", 'green', attrs=['bold']).format(digit[0]))
	
	##Play audio##
	mytext = "I think you have drawn" + digit[0]
	return mytext

# entry point, run the example
mytext = run_example()
playSound(mytext)