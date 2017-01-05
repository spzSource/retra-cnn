# Need to set up image_dim_ordering parameter to "TF" before in ~/.keras/keras.json config file.

import cv2
import json
import numpy as np
import tensorflow as tf
import keras.backend as keras

from keras.models import Sequential
from keras.layers.core import Lambda
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input


class LocalizationModel(object):

    def __init__(self, classification_model):
        self._model = None
        self.classification_model = classification_model

    @property
    def _last(self):
        return self._model.layers[-1]

    @property
    def _origin(self):
        return self._model.layers[0]

    def _target_output(self, name):
        return [layer for layer in self._origin.layers if layer.name == name][0].output

    def localize(self, source_image, class_number, conv_layer_name, number_of_classes, sensitivity):

        self._model = Sequential([
            self.classification_model,
            Lambda(function=lambda x: self._target_category_loss(x, class_number, number_of_classes),
                   output_shape=lambda input_shape: input_shape)
        ])
        self._model.build()

        output, weights = self._calculate_grad_features_weights(conv_layer_name, source_image)

        cam = self._calculate_class_activation_map(output, weights)
        cam = self._normalize_cam(cam)

        coordinates = self._determine_coordinates(cam, sensitivity)
        return cam, coordinates

    def _determine_coordinates(self, mask, sensitivity):
        max_x, max_y = 0, 0
        min_x, min_y = -1, -1

        for col_index, row in enumerate(mask):
            for row_index, item in enumerate(row):
                if item > sensitivity:
                    if min_x > row_index or min_x < 0:
                        min_x = row_index
                    if max_x < row_index:
                        max_x = row_index
                    if min_y > col_index or min_y < 0:
                        min_y = col_index
                    if max_y < col_index:
                        max_y = col_index

        return (min_x, min_y), (max_x, max_y)

    def _target_category_loss(self, x, category_index, n_classes):
        return tf.mul(x, keras.one_hot([category_index], n_classes))

    def _calculate_grad_features_weights(self, layer_name, source_image):
        conv_output = self._target_output(layer_name)
        loss = keras.sum(self._last.output)

        grads = self._normalize(tf.gradients(loss, conv_output, colocate_gradients_with_ops=True)[0])
        output, grads_val = keras.get_session().run(
            fetches=[conv_output, grads],
            feed_dict={self._origin.input: source_image})

        output, grads_val = output[0, :], grads_val[0, :, :, :]
        weights = np.mean(grads_val, axis=(0, 1))

        return output, weights

    def _normalize(self, x):
        return x / (keras.sqrt(keras.mean(keras.square(x))) + 1e-5)

    def _calculate_class_activation_map(self, output, weights):
        cam = np.ones(output.shape[0: 2], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        return cam

    def _normalize_cam(self, cam):
        cam = cv2.resize(cam, (224, 224))
        cam = np.maximum(cam, 0)
        cam /= np.max(cam)

        return cam

if __name__ == "__main__":

    import argparse
    from PIL import Image

    def load_image(path):
        loaded_image = image.load_img(path=path, target_size=(224, 224))
        img_array = image.img_to_array(loaded_image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    def _combine_with_map(source_image, mask):
        source_image = source_image[0, :]
        source_image -= np.min(source_image)
        source_image = np.minimum(source_image, 255)

        mask = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        mask = np.float32(mask) + np.float32(source_image)
        result = 255 * mask / np.max(mask)

        return np.uint8(result)

    def configure_args():
        parser = argparse.ArgumentParser()

        parser.add_argument("--source_path", action="store", help="Path to the source image to be processed.")
        parser.add_argument("--target_path", action="store", help="Path to the target image as a result of processing.")
        parser.add_argument("--labels", action="store", help="Path to labels of the ImageNet data-set.")
        parser.add_argument("--layer_name", action="store", default="block5_pool",
                            help="The name of the layer to be visualized. Default if 'block5_pool'")

        parsed_args = parser.parse_args()

        return parsed_args

    args = configure_args()
    preprocessed_input = load_image(args.source_path)

    classifier = VGG16(weights="imagenet")
    predicted_class = np.argmax(classifier.predict(preprocessed_input))

    with open(args.labels) as labels_file:
        labels = json.load(labels_file)

    model = LocalizationModel(
        classification_model=classifier)

    cam, cords = model.localize(
        source_image=preprocessed_input,
        class_number=predicted_class,
        conv_layer_name=args.layer_name,
        number_of_classes=len(labels),
        sensitivity=0.3)

    target = _combine_with_map(preprocessed_input, cam)

    cv2.putText(
        img=target,
        text=labels[str(predicted_class)],
        org=(10, 10),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=1,
        color=0)

    cords_start, cords_end = cords
    cv2.rectangle(
        img=target,
        pt1=cords_start,
        pt2=cords_end,
        color=(0, 0, 255),
        thickness=2)

    cv2.imwrite(args.target_path, target)

    img = Image.open(args.target_path)
    img.show()
