import cv2

def detect_camera():
    # Cargar el clasificador de Haar para la detección de rostros
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Inicializa el contador de rostros
    face_count = 0

    # Intenta abrir la cámara
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se puede acceder a la cámara")
        return

    print("Cámara detectada. Presiona 'q' para salir.")

    while True:
        # Captura frame por frame
        ret, frame = cap.read()

        if not ret:
            print("No se pudo recibir frame (streaming finalizado?). Saliendo ...")
            break

        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecta rostros
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Actualiza el contador de rostros
        face_count = len(faces)

        # Dibuja un rectángulo alrededor de cada rostro detectado
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Muestra el contador de rostros en el frame
        cv2.putText(frame, f"Faces detected: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Muestra el frame resultante
        cv2.imshow('Frame', frame)

        # Presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera el capturador de video cuando todo esté hecho
    cap.release()
    cv2.destroyAllWindows()

    print(f"Total faces detected: {face_count}")

if __name__ == "__main__":
    detect_camera()
