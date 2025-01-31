import dlib
import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
from keras.models import load_model
import pygame

# Initialisation de pygame pour jouer des sons
pygame.mixer.init()

# Initialisation des détecteurs et prédicteurs
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"shape_predictor_68_face_landmarks.dat")
fa = face_utils.FaceAligner(predictor, desiredFaceWidth=256)

# Chargement du modèle pour prédire l'état des yeux (assurez-vous que le modèle existe)
eye_model = load_model(r'C:\Users\User\Documents\Desktop\Projet\somnolance.keras')

# Seuils pour la détection de bâillement et de somnolence
LIP_AR_THRESH = 0.4  # Seuil pour le ratio d'ouverture de la bouche
EYE_AR_THRESH = 0.25  # Seuil pour le ratio d'ouverture des yeux
LIP_PER_FRAME = 1  # Nombre de frames consécutifs pour déclencher une alerte

# Fonction pour calculer le ratio d'ouverture de la bouche
def calculate_lip(lip_points):
    dist1 = dist.euclidean(lip_points[2], lip_points[6])
    dist2 = dist.euclidean(lip_points[0], lip_points[4])
    return dist1 / dist2

# Fonction pour jouer un son
def play_sound(sound_url):
    pygame.mixer.music.load(sound_url)
    pygame.mixer.music.play()

# Alerte de somnolence et réveil
def alert():
    print("ALERTE: Somnolence détectée!")
    play_sound("path_to_local_sound/beep_sound.ogg")  # Utiliser un fichier local si nécessaire

def reveil():
    print("Réveil: Le conducteur est alerte.")
    play_sound("path_to_local_sound/reveil_sound.ogg")

def pause():
    print("Pause: Le conducteur doit se reposer.")
    play_sound("path_to_local_sound/pause_sound.ogg")

# Démarrer le flux vidéo
def video_stream():
    global video_capture
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise ValueError("La caméra ne peut pas être ouverte.")

def video_frame():
    ret, frame = video_capture.read()
    if not ret:
        print("Erreur de lecture de l'image")
        return False
    # Redimensionner l'image si elle est trop grande
    frame = cv2.resize(frame, (640, 480))
    return frame

# Initialisation des variables
counter = 0
bbox = ''

# Démarrer le flux vidéo
video_stream()

# Boucle principale de détection
while True:
    # Capturer une image du flux vidéo
    img = video_frame()
    if img is False:
        break

    # Créer un overlay transparent pour la boîte englobante
    bbox_array = np.zeros([480, 640, 4], dtype=np.uint8)

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Vérification si l'image est bien en niveaux de gris (8 bits)
    if len(gray.shape) != 2:
        print("Erreur: l'image n'est pas en niveaux de gris")
        continue  # Ignorer cette image et passer à la suivante

    # Détecter les visages
    rects = detector(gray, 2)
    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face_aligned = fa.align(img, gray, rect)
        face_aligned_gray = cv2.cvtColor(face_aligned, cv2.COLOR_BGR2GRAY)

        # Détecter les yeux
        eyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml").detectMultiScale(face_aligned_gray, 1.1, 4)
        predictions = []
        for (ex, ey, ew, eh) in eyes:
            eye = face_aligned[ey:ey+eh, ex:ex+ew]
            eye = cv2.resize(eye, (32, 32))
            eye = np.expand_dims(eye, axis=0)
            
            # Prédiction des yeux
            ypred = eye_model.predict(eye)
            ypred = np.argmax(ypred[0], axis=0)
            predictions.append(ypred)

        # Vérifier si les yeux sont fermés
        if all(i == 0 for i in predictions):
            cv2.rectangle(bbox_array, (x, y), (x+w, y+h), (255, 0, 0), 8)
            cv2.putText(bbox_array, 'Sleeping', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            alert()
            reveil()
        else:
            cv2.rectangle(bbox_array, (x, y), (x+w, y+h), (0, 255, 0), 8)
            cv2.putText(bbox_array, 'Not Sleeping', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Détecter les bâillements
        lip_points = [60, 61, 62, 63, 64, 65, 66, 67]
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        lip_shape = shape[lip_points]
        LAR = calculate_lip(lip_shape)

        # Dessiner les contours des lèvres
        lip_hull = cv2.convexHull(lip_shape)
        cv2.drawContours(bbox_array, [lip_hull], -1, (0, 255, 0), 1)

        # Vérifier si la bouche est ouverte
        if LAR > LIP_AR_THRESH:
            counter += 1
            if counter > LIP_PER_FRAME:
                cv2.putText(bbox_array, "YAWNING ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                pause()
        else:
            counter = 0

        # Afficher le ratio d'ouverture de la bouche
        cv2.putText(bbox_array, "LAR: {:.2f}".format(LAR), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Afficher l'image
    cv2.imshow("Frame", img)

    # Quitter si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la capture et fermer les fenêtres
video_capture.release()
cv2.destroyAllWindows()
