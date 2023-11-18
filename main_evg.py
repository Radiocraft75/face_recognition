import os
import cv2
import pafy
import face_recognition
import numpy as np
import schedule

skip_frame = 3

# video_capture = cv2.VideoCapture("rtsp://admin:ultrabook75@192.168.0.64:554/ISAPI/Streaming/Channels/101")

video_capture = cv2.VideoCapture("video/fo_test.avi")
length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

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
# x1 = 1600
# x2 = 1750
# y1 = 800
# y2 = 1050

x1 = 1250
x2 = 2400
y1 = 750
y2 = 1800

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


# Настройка расписания очистки списка известных лиц
def clear_array():
    print("Clear known face")
    known_face_encodings.clear()
    index = 0

schedule.every().day.at("00:01").do(clear_array)

index = 0
while True:
    schedule.run_pending()
    ret, frame = video_capture.read()
    
    # ббрезаем изображение до ожлажти поипка
    frame2 = frame[y1:y2, x1:x2]

    if not ret:    	
        break

    if index == skip_frame:
        rgb_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        
        # Находим лица в области rgb_frame
        # model="hog/cnn" 
        #   Гистограмма направленных градиентов(HOG) работает быстро, вполне достаточно CPU, но распознает хуже и только фронтальные лица.
        #   Алгоритм на базе сверточных нейронных сетей(CNN) целесообразно использовать только на GPU, зато распознает гораздо лучше и во всех возможных позах.
        # number_of_times_to_upsample - сколько раз увеличивать выборку изображения в поисках лиц. Чем больше число, тем меньше лица.
        face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2, model="cnn")
        
        # model="small/large" модель поиска элементов лица (поиск опознавательных точек) 
        #        small — углы глаз, нос, соотношение пропорций расстояний между глазами и носом
        #        large - также распознавание рта, овала лица и бровей 
        # num_jitters - сколько раз повторять выборку лица при расчете кодировки. Чем больше, тем точнее, но медленнее.
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1, model="large")   

        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.65)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            poisk = False
            # проверка массивов на пустоту
            if len(face_distances) > 0 and len(matches) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    poisk = True

            # poisk = False
            # if True in matches:
            #     #print("Совпадение найдено")
            #     poisk = True

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
cv2.destroyAllWindows()
