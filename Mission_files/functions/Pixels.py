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

def resize_to_80( image ):
	
	dim1 = image.shape[0]
	dim2 = image.shape[1]
	stride1 = dim1 / 80 
	stride2 = dim2 / 80
	return image[::stride1, ::stride2]

# sleep_time = 0.25
# count = 0
# got_image = False
# #Loop until mission ends:
# while world_state.is_mission_running:
#     sys.stdout.write(".")
#     print world_state  
#     if len(world_state.video_frames) > 0:
# 	pixels = getPixels(world_state.video_frames[0]) 
# 	print pixels.shape
# 	np.save("image", pixels)
# 	got_image = True  
#     else: 
# 	print "received no frame"

#     time.sleep(sleep_time)
#     world_state = agent_host.getWorldState()
#     for error in world_state.errors:
#         print "Error:",error.text
   
#     if got_image:
# 	break
#     count += 1
	
