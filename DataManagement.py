import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class DatasetHandler:
    def __init__(self, dataset_path, image_size=(256, 256), augment=False):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.augment = augment

    def load_data(self):
        image_paths = []
        mask_paths = []


        for img_name in os.listdir(os.path.join(self.dataset_path, 'images')):
            image_paths.append(os.path.join(self.dataset_path, 'images', img_name))
            mask_paths.append(os.path.join(self.dataset_path, 'masks', img_name))  #  masks following the same naming

        return image_paths, mask_paths

    def preprocess(self, image, mask):
        # Resize/normalize  images\masks
        image = tf.image.resize(image, self.image_size) / 255.0  # Normalize image to [0, 1]
        mask = tf.image.resize(mask, self.image_size) #expecting mask file to be a binary file (0 or 1 depending if background or not)
        return image, mask

    def get_data_generator(self, batch_size=32):
        image_paths, mask_paths = self.load_data()

        def generator():
            for img_path, mask_path in zip(image_paths, mask_paths):
                image = load_img(img_path, target_size=self.image_size)
                image = img_to_array(image)

                mask = load_img(mask_path, color_mode="grayscale", target_size=self.image_size)
                mask = img_to_array(mask)

                yield self.preprocess(image, mask)

        dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.float32))
        dataset = dataset.batch(batch_size).shuffle(buffer_size=100)
        return dataset


