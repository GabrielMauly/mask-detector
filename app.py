from imutils.video import VideoStream
from loguru import logger
from time import sleep
import tflite_runtime.interpreter as tflite
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
import numpy as np
import imutils
import cv2


class App:

    def __init__(self, model, label, camera_id, size):
        self.model = model
        self.label = label
        self.camera_id = camera_id
        self.size = size
        self.color_font = (255, 255, 255)
        self.color_mask = (0, 255, 0)
        self.color_n_mask = (0, 0, 255)
        self.line = 1
        self.border = 0.4
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.detector = MTCNN()
        self.mean = 128
        self.std = 128

    def load_labels(self):

        try:
            with open(self.label, 'r') as f:
                return [line.strip() for line in f.readlines()]
        except Exception as e:
            logger.error(e)

    def load_model(self):

        try:
            model = tflite.Interpreter(model_path=self.model)
            model.allocate_tensors()

            return model
        except Exception as e:
            logger.error(e)

    def process_image(self, model, labels, image, input_details, output_details):

        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, axis=0)

        # pre process image
        image_tensor = (image_tensor - self.mean) / self.std

        model.set_tensor(input_details[0]['index'], image_tensor)
        model.invoke()

        output_data = model.get_tensor(output_details[0]['index'])
        output_data = np.squeeze(output_data)

        idx = np.argmax(output_data)
        label = labels[idx]
        score = round(output_data[idx] * 100, 2)

        classification = "{}: {} %".format(label, score)

        return idx, classification

    def inference(self):

        model = self.load_model()
        labels = self.load_labels()

        input_details = model.get_input_details()
        output_details = model.get_output_details()

        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]

        camera = VideoStream(src=self.camera_id).start()

        sleep(0.8)

        while True:

            try:
                frame = camera.read()
                frame = imutils.resize(frame, width=self.size)

                out_frame = frame.copy()

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                faces = self.detector.detect_faces(frame)

                logger.info('Total person: {}'.format(len(faces)))

                for i, face in enumerate(faces):
                    (x, y, w, h) = faces[i]['box']

                    frame = frame[y:y + h, x:x + w, :]

                    image = cv2.resize(frame, (width, height))

                    idx, classification = self.process_image(model, labels, image, input_details, output_details)

                    cv2.putText(out_frame, classification, (x + 6, y - 6), self.font, 0.4, self.color_font, self.line)

                    if idx == 0:
                        cv2.rectangle(out_frame, (x, y), (x + w + 30, y + w + 30), self.color_mask, self.line)
                    else:
                        cv2.rectangle(out_frame, (x, y), (x + w + 30, y + w + 30), self.color_n_mask, self.line)

                cv2.imshow('Real time', out_frame)
                key = cv2.waitKey(1)
                if key == 27:
                    break
            except Exception as e:
                logger.error(e)


if __name__ == "__main__":

    app = App(model='./model.tflite', label='./labels.txt', camera_id=2, size=360)
    app.inference()
