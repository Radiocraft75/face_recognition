# import libraries
import cv2
import pafy
#import dlib.cuda as cuda
import face_recognition
import numpy as np

# Open the input movie file
input_movie = cv2.VideoCapture("hamilton_clip.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('output.avi', fourcc, 29.97, (640, 360))

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
# known_faces = []
# face_names = []

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden"
]

index = 0
while True:
    # ret, frame = video_capture.read()
    ret, frame = input_movie.read()
    
    if not ret:    	
        break

    if index == 5:
        # Ковертируем изображение из формата BGR (используемый OpenCV) в формат RGB (используемый face_recognition)
        # rgb_frame = frame[:, :, ::-1]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # rgb_frame = rgb_frame[100:500, 100:1000]

        # Находим лица в области rgb_frame
        face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
        # for top, right, bottom, left in face_locations:
        #     cv2.rectangle(frame, (left+100, top+100), (right+100, bottom+100), (0, 0, 255), 2)
        
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # name = None
        # for face_encoding in face_encodings:
        #     match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)
        #     if match:
        #         name = "Присутсвует в БД"
        #     else:
        #         known_faces.append(face_encoding)

        # for (top, right, bottom, left), name in zip(face_locations, face_names):
        #     if not name:
        #         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        #     else:
        #         cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        #         font = cv2.FONT_HERSHEY_DUPLEX
        #         cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)


        index = 0
        # Рисуем зеленый квадрат вокруг области поиска лиц
        # cv2.rectangle(frame, (100, 100), (1000, 500), (0, 255, 0), 2)

        cv2.imshow('Video2', frame)
        # cv2.imshow('Video1', rgb_frame)
        output_movie.write(frame)
    index += 1

    # Display the resulting image
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
# video_capture.release()
input_movie.release()
cv2.destroyAllWindows()
