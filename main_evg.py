import cv2
import pafy
#import dlib.cuda as cuda
import face_recognition
import numpy as np

skip_frame = 3

# Open the input movie file
input_movie = cv2.VideoCapture("video/hamilton_clip.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('video/output.avi', fourcc, 29.97, (640, 360))

# Get a reference to webcam
#print(cuda.get_num_devices())
# url = "https://youtu.be/1sHyjr-86uY"
# video = pafy.new(url)
# best = video.getbest(preftype="mp4")

#video_capture = cv2.VideoCapture("rtsp://admin:ultrabook75@192.168.0.64:554/ISAPI/Streaming/Channels/101")
#video_capture = cv2.VideoCapture(best.url)
#video_capture = cv2.VideoCapture(0)

face_locations = []
face_encodings = []

# Create arrays of known face encodings and their names
known_face_encodings = []

index = 0
while True:
    # ret, frame = video_capture.read()
    ret, frame = input_movie.read()
    
    if not ret:    	
        break

    if index == skip_frame:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # rgb_frame = rgb_frame[100:500, 100:1000]

        # Находим лица в области rgb_frame
        face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        # for top, right, bottom, left in face_locations:
        #     cv2.rectangle(frame, (left+100, top+100), (right+100, bottom+100), (0, 0, 255), 2)        
        

        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

            poisk = False
            if True in matches:
                #print("Совпадение найдено")
                poisk = True

            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            if poisk:
                #print("Присутстует в базе")
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, "OK", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            else:
                print("Добавить в базу нового")
                print(len(known_face_encodings))
                known_face_encodings.append(face_encoding)
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, "NEW", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Or instead, use the known face with the smallest distance to the new face
            #face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            #best_match_index = np.argmin(face_distances)
            # if matches[best_match_index]:
            #     name = known_face_names[best_match_index]

        index = 0
        # Рисуем зеленый квадрат вокруг области поиска лиц
        # cv2.rectangle(frame, (100, 100), (1000, 500), (0, 255, 0), 2)

        cv2.imshow('Input', frame)
    index += 1

    # Display the resulting image
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# video_capture.release()
input_movie.release()
cv2.destroyAllWindows()
