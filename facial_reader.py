from deepface import DeepFace
import cv2
from PIL import Image
import numpy as np
import cv2

# /Users/angellyncervantes/.cache/kagglehub/datasets/ziya07/facial-micro-expression-recognition/versions/1

def facial_reader():
    cap = cv2.VideoCapture(0)

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotions = result[0]['emotion']

            y = 30
            for emotion, score in emotions.items():
                text = f"{emotion}: {round(score, 2)}%"
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y += 25

        except Exception as e:
            print("Error:", e)

        cv2.imshow("Emotion Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def read_image(image):
    #  img = read_image_as_cv2(image)
    img = cv2.imread(image)
    result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

    # print(result[0]['emotion'])

    negative = ['contempt', 'anger', 'fear', 'disgust', 'sad']
    positive = ['surprised', 'happy']

    pos = 0
    neg = 0
    for k, v in result[0]['emotion'].items():
        if k in positive:
            pos += v
        else:
            neg += v
            
    return ("positive", pos) if pos > neg else ("negative", neg)
     

    # max = 0
    # category = ""
    # for k, v in result[0]['emotion'].items():
    #     if v:
    #         max = v
    #         category = k

    # return (category, k)

def read_image_as_cv2(file):
    image = Image.open(file.stream).convert('RGB')
    np_img = np.array(image)
    cv2_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    return cv2_img


if __name__ == "__main__":
    emotions = read_image("./ck+/happy/S010_006_00000013.png")
    print(emotions)
