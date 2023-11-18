import os
import cv2
import datetime

video_capture = cv2.VideoCapture("rtsp://admin:ultrabook75@192.168.0.64:554/ISAPI/Streaming/Channels/101")

now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('video/' + now + '_fo_test.avi', fourcc, 29.97, (3840, 2160))

# Размер начальной картинки 3840*2160
# Размер картинки для вывода на экран
desired_width = 1080
aspect_ratio = desired_width / 3840
desired_height = int(2160 * aspect_ratio)
dim = (desired_width, desired_height)

# #Write video from testing
# # Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('output.avi', fourcc, 29.97, (3840, 2160))

while True:
    ret, frame = video_capture.read()

    if not ret:    	
        break

    output_movie.write(frame)

    # Display the resulting image
    frame = cv2.resize(frame, dsize=dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('Input', frame)
   
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
output_movie.release()
cv2.destroyAllWindows()
