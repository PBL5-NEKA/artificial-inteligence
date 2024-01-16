import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator

NEW_IMAGE_SIZE = (64, 64)
HAAR_CASCADE_MODEL_PATH = '../models/haarcascade_frontalface_alt2.xml'
data_generator = None


# AUTOMATED CROPPING USING HAAR CASCADE

def load_face_cascade(path):
    face_cascade = cv2.CascadeClassifier(path)
    if face_cascade.empty():
        print("Error loading Haar Cascade classifier.")
    else:
        print("Haar Cascade classifier loaded successfully.")

    return face_cascade


def apply_haar_cascade_on_image(image, face_cascade):
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
    cropped_images = np.random.random((images.shape[0], CROPPED_IMAGE_SIZE[0], CROPPED_IMAGE_SIZE[1], 3))

    miscropped_images = 0

    for idx, image in enumerate(images):
        detected_face, miscropped = apply_haar_cascade_on_image(image, face_cascade)

        cropped_images[idx] = np.array(cv2.resize(detected_face.astype(np.uint8), CROPPED_IMAGE_SIZE))

        if miscropped:
            miscropped_images += 1

    return miscropped_images, cropped_images


def apply_and_safe_haar_cascade(images, labels, face_cascade):
    miscropped_images = 0

    for idx, image in enumerate(images):

        # define the output path for the actual image
        if labels[idx] == 0:
            output_path = f'../data/interim/Cropped/Class1/processed_image_{idx}.png'
        else:
            output_path = f'../data/interim/Cropped/Class2/processed_image_{idx}.png'

        detected_face = apply_haar_cascade_on_image(image, face_cascade)

        if detected_face:
            cv2.imwrite(output_path, cv2.resize(cv2.cvtColor(detected_face, cv2.COLOR_RGB2BGR), NEW_IMAGE_SIZE))
        else:
            miscropped_images += 1
            cv2.imwrite(output_path, cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), NEW_IMAGE_SIZE))

    return miscropped_images


# BASIC PREPROCESSING

def init_data_generator():
    data_generator = ImageDataGenerator(
        rotation_range=50,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=15,
        zoom_range=[0.5, 1.5],
        vertical_flip=True,
        # orizontal_flip = True
    )

    return data_generator


def apply_random_data_augmentation(image):
    return data_generator.random_transform(image)
