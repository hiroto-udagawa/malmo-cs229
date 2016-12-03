import numpy as np

def getPixels( frame ): 

	width = frame.width
        height = frame.height
        channels = frame.channels
        pixels = np.array(frame.pixels, dtype = int)
        img = np.reshape(pixels, (width, height, channels))
        #height = 640
        #width = 480
        return (img[:,:,0]+ img[:,:,1]+ img[:,:,2])/3

