import cv2
import numpy as np
import sqlite3
import pyttsx3
from deepface import DeepFace
import pickle
import threading

# --- Lokal DB nümunəsi ---
DB_PATH = "users.db"


def init_db(path=DB_PATH):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, encoding BLOB)")
    conn.commit()
    conn.close()


def add_user(name, image_path, path=DB_PATH):
    """DeepFace ilə üz encoding çıxarır və DB-yə əlavə edir"""
    try:
        # DeepFace ilə embedding çıxarma
        embedding_objs = DeepFace.represent(img_path=image_path,
                                            model_name="Facenet512",
                                            enforce_detection=True)

        if len(embedding_objs) == 0:
            raise ValueError("Üz tapılmadı: " + image_path)

        # İlk üzün embedding-ini götür
        embedding = np.array(embedding_objs[0]["embedding"])
        encoding = pickle.dumps(embedding)  # Pickle ilə serialize et

        conn = sqlite3.connect(path)
        c = conn.cursor()
        c.execute("INSERT INTO users (name, encoding) VALUES (?, ?)", (name, encoding))
        conn.commit()
        conn.close()
        print(f"{name} uğurla əlavə edildi!")
    except Exception as e:
        print(f"Xəta: {e}")


def load_known_encodings(path=DB_PATH):
    """DB-dən bütün istifadəçilərin encoding-lərini yükləyir"""
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("SELECT name, encoding FROM users")
    rows = c.fetchall()
    conn.close()

    known_names = []
    known_encs = []
    for name, enc_blob in rows:
        enc = pickle.loads(enc_blob)
        known_names.append(name)
        known_encs.append(enc)
    return known_names, known_encs


def cosine_similarity(v1, v2):
    """İki vektor arasında cosine similarity hesablayır"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# --- TTS init ---
tts = pyttsx3.init()


def say(text):
    tts.say(text)
    tts.runAndWait()


def process_face_async(face_img, known_names, known_encs, threshold, callback):
    """Üz tanıma işlərini arxa planda yerinə yetirir"""
    try:
        # DeepFace ilə embedding çıxar
        embedding_objs = DeepFace.represent(img_path=face_img,
                                            model_name="Facenet512",
                                            enforce_detection=False)

        if len(embedding_objs) > 0:
            current_enc = np.array(embedding_objs[0]["embedding"])

            # DB-dəki encodings ilə müqayisə et
            best_match = None
            best_similarity = 0.0

            for known_name, known_enc in zip(known_names, known_encs):
                similarity = cosine_similarity(current_enc, known_enc)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = known_name

            callback(best_match, best_similarity, threshold)
    except Exception as e:
        pass  # Xəta baş verərsə sadəcə keç


# --- Main real-time loop ---
def main():
    init_db()
    cap = cv2.VideoCapture(0)

    known_names, known_encs = load_known_encodings()

    if len(known_names) == 0:
        print("DB-də istifadəçi yoxdur. Əvvəlcə add_user() ilə istifadəçi əlavə edin!")
        return

    process_every_n_frames = 10  # Hər 10 frame-də bir yoxla (daha az yük)
    frame_count = 0
    threshold = 0.6

    # Haar Cascade ilə üz aşkarlama (daha sürətli)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("Kamera başladı. Çıxmaq üçün 'q' düyməsini basın.")

    processing = False  # İşləm bayrağı
    recognized_person = None
    last_warning_time = 0  # Son xəbərdarlıq vaxtı

    def on_face_recognized(name, similarity, threshold):
        nonlocal recognized_person, last_warning_time
        if similarity > threshold:
            recognized_person = name
        else:
            # Üz tanınmadı - spam-dan qaçmaq üçün 3 saniyədə bir xəbərdarlıq
            import time
            current_time = time.time()
            if current_time - last_warning_time > 3:
                print(f"\n⚠️  ÜZ TANINMADI! Kənar şəxs aşkar edildi!")
                print(f"   Oxşarlıq dərəcəsi: {similarity:.2f} (Tələb olunan: {threshold})\n")
                say("Kənar şəxs aşkar edildi. Zəhmət olmasa geri qayıdın.")
                last_warning_time = current_time

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Əgər tanınmışsa, kameranı bağla
        if recognized_person:
            say(f"Salam, {recognized_person}. Xoş gəldin.")
            print(f"\n{'=' * 50}")
            print(f"✓ Salam {recognized_person}, Xoş gəldin!")
            print(f"{'=' * 50}\n")

            cap.release()
            cv2.destroyAllWindows()
            return

        frame_count += 1

        # Hər N frame-də bir üz tanıma işləmini başlat
        if frame_count % process_every_n_frames == 0 and not processing:
            # Kiçik frame ilə işlə (performans)
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            # Üz aşkarlama
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                # İlk üzü götür
                (x, y, w, h) = faces[0]

                # Koordinatları scale et
                x *= 2;
                y *= 2;
                w *= 2;
                h *= 2

                # Üzü kəs
                face_img = frame[y:y + h, x:x + w].copy()

                # Arxa planda işlə (threading)
                processing = True
                thread = threading.Thread(
                    target=process_face_async,
                    args=(face_img, known_names, known_encs, threshold, on_face_recognized)
                )
                thread.daemon = True
                thread.start()

                # Thread-i sıfırla (növbəti dəfə yenidən başlaya bilsin)
                def reset_processing():
                    nonlocal processing
                    thread.join()
                    processing = False

                threading.Thread(target=reset_processing, daemon=True).start()

        # Kameranı göstər (heç bir donma olmadan)
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # İstifadə nümunəsi:
    # add_user("Ali", "ali.jpg")  # Əvvəlcə istifadəçi əlavə edin
    main()