from model import CNN
import torch
from PIL import ImageGrab, Image
from torchvision import transforms
import numpy as np
import cv2
from dataset import emotional_classes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

transform = transforms.Compose([
    transforms.Resize((48, 48)),       # Resize image to 48x48
    transforms.ToTensor(),               # Convert to tensor [0, 1]
    # Optional: Normalize (e.g., for pretrained models like ResNet)
])
model_id = 'Emotional_Recognition'
model_path = f"models/{model_id}/model_latest.pth"


# def test(path):
#     model = CNN.load(model_path).to(device)
#     img = Image.open(path)
#     img = transform(img).reshape(1,48,48)
#     img = img.to(device)
#     output, emotion = model(img)
#     emotion = emotional_classes[np.argmax(output)]
#     print(f'Emotion: {emotion}')

if __name__ == '__main__':
    model = CNN.load(model_path).to(device)
    model.eval()
    while True:  
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        ret, img = cap.read()  
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        faces = face_cascade.detectMultiScale(gray_img, 1.25, 4) 
    
        for (x,y,w,h) in faces: 
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)  
            rec_gray = gray_img[y:y+h, x:x+w] 
            rec_color = img[y:y+h, x:x+w]

            rec_gray = cv2.resize(rec_gray, (48, 48))
            rec_gray = np.array(rec_gray, dtype=np.float32)[None, None, :, :]
            rec_gray = torch.from_numpy(rec_gray).to(device)
            output = model(rec_gray)
            emotion = emotional_classes[np.argmax(output.detach().cpu().numpy())]
            cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        cv2.imshow('Face Recognition',img)
    
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
    
    cap.release() 
    cv2.destroyAllWindows()

# if __name__ == '__main__':
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert to grayscale for face detection
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Detect faces
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#         for (x, y, w, h) in faces:
#             roi_gray = gray[y:y+h, x:x+w]
#             roi_gray = cv2.resize(roi_gray, (48, 48))
#             roi_gray = roi_gray.astype("float") / 255.0
#             roi_gray = img_to_array(roi_gray)
#             roi_gray = np.expand_dims(roi_gray, axis=0)

#             # Predict emotion
#             preds = model.predict(roi_gray, verbose=0)[0]
#             label = emotion_labels[np.argmax(preds)]
#             confidence = np.max(preds)

#             # Draw rectangle and emotion label
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

#         # Display the frame
#         cv2.imshow("Emotion Recognition", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()