import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('sign_language_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (48, 48))
    image = image.reshape(1, 48, 48, 1).astype('float32') / 255.0
    return image

# Labels (assuming you trained on A-Z and blank)
labels = [chr(i) for i in range(65, 91)] + ['blank']  # ['A', 'B', ..., 'Z', 'blank']

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
    crop_frame = frame[40:300, 0:300]
    
    # Preprocess the frame
    preprocessed_frame = preprocess_image(crop_frame)
    
    # Make prediction
    pred = model.predict(preprocessed_frame)
    prediction_label = labels[np.argmax(pred)]
    
    # Display prediction
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
    if prediction_label == 'blank':
        cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        accu = "{:.2f}".format(np.max(pred) * 100)
        cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("output", frame)
    if cv2.waitKey(27) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
