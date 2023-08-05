import cv2
import dlib
import face_recognition

def load_image_and_encode(image_path):
    image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    return face_encoding

def detect_face(face_encoding):
    # Load the webcam
    webcam = cv2.VideoCapture(0)

    # Create a face detector using dlib
    detector = dlib.get_frontal_face_detector()

    while True:  # Continuous loop for continuous face detection
        # Read a frame from the webcam
        ret, frame = webcam.read()

        # Convert the frame to RGB (dlib requires BGR format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame using dlib
        face_locations = detector(rgb_frame)

        # Convert the frame back to BGR for visualization
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # Check if your face is detected
        if face_locations:
            for face_location in face_locations:
                top, right, bottom, left = face_location.left(), face_location.right(), face_location.bottom(), face_location.top()
                face_image = frame[top:bottom, left:right]

                # Convert the face image to RGB (face_recognition library expects RGB format)
                face_image_rgb = face_image[:, :, ::-1]

                # Compute face encoding for the detected face
                face_encodings = face_recognition.face_encodings(face_image_rgb)

                # Check if any face encoding is detected
                if face_encodings:
                    face_encoding_in_frame = face_encodings[0]

                    # Compare face encodings
                    match = face_recognition.compare_faces([face_encoding], face_encoding_in_frame)

                    if match[0]:
                        # Draw a rectangle around the detected face
                        cv2.rectangle(rgb_frame, (left, top), (right, bottom), (255, 0, 0), 2)

                        # Print "OK" when your face is detected
                        print("OK")
                else:
                    print("No face encoding detected in the frame.")

        # Display the processed frame with rectangles
        cv2.imshow('Face Detection', rgb_frame)

        # Exit the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the OpenCV window
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Use a relative path to the image file
    image_path = "wp1840123.jpg"
    your_face_encoding = load_image_and_encode(image_path)
    detect_face(your_face_encoding)
