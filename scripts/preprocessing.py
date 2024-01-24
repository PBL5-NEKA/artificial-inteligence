import numpy as np
import cv2

CROPPED_IMAGE_SIZE = (128, 128)
HAAR_CASCADE_MODEL_PATH = '../models/haarcascade_frontalface_alt2.xml'


def load_face_cascade(path):
    """
    Load a Haar Cascade classifier for detecting faces.

    Parameters:
    - path (str): Path to the Haar Cascade XML file.

    Returns:
    - cv2.CascadeClassifier: Loaded Haar Cascade classifier.
    """

    face_cascade = cv2.CascadeClassifier(path)
    if face_cascade.empty():
        print("Error loading Haar Cascade classifier.")
    else:
        print("Haar Cascade classifier loaded successfully.")
    return face_cascade


def apply_haar_cascade_on_image(image, face_cascade):
    """
    Apply Haar Cascade face detection on an image.

    Parameters:
    - image (numpy.ndarray): Input image in RGB format.
    - face_cascade (cv2.CascadeClassifier): Loaded Haar Cascade classifier.

    Returns:
    - Tuple: A tuple containing the detected face image and a boolean indicating if the image was miscropped.
    """

    miscropped = False

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.uint8)

    # apply algorithm
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces) > 0:
        # Draw a rectangle around the main face and crop this
        (x, y, w, h) = faces[0]
        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        detected_face = image[y:y + h, x:x + w]
    else:
        miscropped = True
        detected_face = image

    return detected_face, miscropped


def apply_haar_cascade_on_images(images, face_cascade):
    """
    Apply Haar Cascade face detection on a batch of images.

    Parameters:
    - images (numpy.ndarray): Array of input images in RGB format.
    - face_cascade (cv2.CascadeClassifier): Loaded Haar Cascade classifier.

    Returns:
    - Tuple: A tuple containing the count of miscropped images and an array of detected and cropped face images.
    """

    cropped_images = np.random.random((images.shape[0], CROPPED_IMAGE_SIZE[0], CROPPED_IMAGE_SIZE[1], 3))

    miscropped_images = 0

    for idx, image in enumerate(images):
        detected_face, miscropped = apply_haar_cascade_on_image(image, face_cascade)

        cropped_images[idx] = np.array(cv2.resize(detected_face.astype(np.uint8), CROPPED_IMAGE_SIZE))

        if miscropped:
            miscropped_images += 1

    return miscropped_images, cropped_images
