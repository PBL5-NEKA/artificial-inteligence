import unittest
import cv2
import os
import numpy as np
from scripts.preprocessing import load_face_cascade, apply_haar_cascade_on_image, apply_haar_cascade_on_images

CROPPED_IMAGE_SIZE = (128, 128)


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Set up the face cascade for testing
        self.face_cascade_path = '../models/haarcascade_frontalface_alt2.xml'
        self.face_cascade = load_face_cascade(self.face_cascade_path)

    # --- load_face_cascade test ---

    def test_load_face_cascade_success(self):
        # Provide the correct path to your Haar Cascade XML file
        cascade_path = '../models/haarcascade_frontalface_alt2.xml'
        face_cascade = load_face_cascade(cascade_path)

        self.assertIsInstance(face_cascade, cv2.CascadeClassifier)
        self.assertFalse(face_cascade.empty())
        self.assertEqual(cv2.__version__, '4.9.0')  # Change the version accordingly

    def test_load_face_cascade_failure(self):
        # Provide an incorrect path to simulate a failure
        cascade_path = '../models/haarcascade_frontalface_at2.xml'
        face_cascade = load_face_cascade(cascade_path)

        self.assertIsInstance(face_cascade, cv2.CascadeClassifier)
        self.assertTrue(face_cascade.empty())
        self.assertEqual(cv2.__version__, '4.9.0')  # Change the version accordingly

    # --- apply_haar_cascade_on_image test ---

    def test_apply_haar_cascade_on_image_with_face(self):
        # Load an image with a face for testing
        image_with_face = cv2.imread('img-test/has_face.jpg')
        detected_face, miscropped = apply_haar_cascade_on_image(image_with_face, self.face_cascade)

        self.assertIsInstance(detected_face, np.ndarray)
        self.assertFalse(miscropped)

    def test_apply_haar_cascade_on_image_without_face(self):
        # Load an image without a face for testing
        image_without_face = cv2.imread('img-test/no_face.jpg')
        detected_face, miscropped = apply_haar_cascade_on_image(image_without_face, self.face_cascade)

        self.assertIsInstance(detected_face, np.ndarray)
        self.assertTrue(miscropped)

    # --- apply_haar_cascade_on_images test ---

    def load_images_from_directory(self, directory):
        # Load all images from the specified directory
        image_files = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.jpeg', '.png'))]
        images = [cv2.imread(os.path.join(directory, f)) for f in image_files]
        return np.array(images)

    def test_apply_haar_cascade_on_images(self):
        # Specify the directory containing images for testing
        test_images_directory = 'img-test/images_with_face'
        images = self.load_images_from_directory(test_images_directory)

        miscropped_images, cropped_images = apply_haar_cascade_on_images(images, self.face_cascade)

        self.assertIsInstance(miscropped_images, int)
        self.assertIsInstance(cropped_images, np.ndarray)
        self.assertEqual(cropped_images.shape, (len(images), CROPPED_IMAGE_SIZE[0], CROPPED_IMAGE_SIZE[1], 3))

    def test_apply_haar_cascade_on_images_with_miscropped(self):
        # Specify the directory containing images for testing
        test_images_directory = 'img-test/images_with_no_face'
        images = self.load_images_from_directory(test_images_directory)

        miscropped_images, cropped_images = apply_haar_cascade_on_images(images, self.face_cascade)

        self.assertIsInstance(miscropped_images, int)
        self.assertIsInstance(cropped_images, np.ndarray)
        self.assertEqual(cropped_images.shape, (len(images), CROPPED_IMAGE_SIZE[0], CROPPED_IMAGE_SIZE[1], 3))
        self.assertEqual(miscropped_images, len(images))


if __name__ == '__main__':
    unittest.main()
