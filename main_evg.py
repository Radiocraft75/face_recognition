import os
import cv2
import pafy
import face_recognition
import numpy as np
import pickle

skip_frame = 3

video_capture = cv2.VideoCapture("rtsp://admin:ultrabook75@192.168.0.64:554/ISAPI/Streaming/Channels/101")
file_path = "kfe.npy"
# Размер начальной картинки 3840*2160
# Размер картинки для вывода на экран
desired_width = 1080
aspect_ratio = desired_width / 3840
desired_height = int(2160 * aspect_ratio)
dim = (desired_width, desired_height)

"""
Облабть обработки
#     x1,y1_____________
#     |                 |
#     |                 |
#     |_________________x2,y2
"""
x1 = 1600
x2 = 1750
y1 = 800
y2 = 1050

face_locations = []
face_encodings = []

# Create arrays of known face encodings and their names
if os.path.isfile(file_path):
    known_face_encodings = np.load("kfe.npy").tolist()
else:
    known_face_encodings = []

print("Чтение данных")
print("Total in the database:")
print(len(known_face_encodings))

# #Write video from testing
# # Create an output movie file (make sure resolution/frame rate matches input video!)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# output_movie = cv2.VideoWriter('output.avi', fourcc, 29.97, (3840, 2160))

index = 0
while True:
    ret, frame = video_capture.read()

    # output_movie.write(frame)
    
    # ббрезаем изображение до ожлажти поипка
    frame2 = frame[y1:y2, x1:x2]
    
    if not ret:    	
        break

    if index == skip_frame:
        rgb_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        
        # Находим лица в области rgb_frame
        face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)      

        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.8)

            poisk = False
            if True in matches:
                #print("Совпадение найдено")
                poisk = True

            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            if poisk:
                #print("Присутстует в базе")
                # Draw a label with a name below the face
                cv2.rectangle(frame2, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame2, "OK", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                # Draw a box around the face
                cv2.rectangle(frame2, (left, top), (right, bottom), (0, 255, 0), 2)
            else:
                print("Добавить в базу нового")
                known_face_encodings.append(face_encoding)
                print(len(known_face_encodings))
                # Draw a label with a name below the face
                cv2.rectangle(frame2, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame2, "NEW", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                # Draw a box around the face
                cv2.rectangle(frame2, (left, top), (right, bottom), (0, 0, 255), 2)

        index = 0       
        
        cv2.imshow('Input1', frame2)
        
    index += 1
    
    # Рисуем зеленый квадрат вокруг области поиска лиц
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    # Display the resulting image
    frame = cv2.resize(frame, dsize=dim, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)

   
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        if (len(known_face_encodings) != 0):
            np.save("kfe.npy", known_face_encodings)
        break

video_capture.release()
# output_movie.release()
cv2.destroyAllWindows()
