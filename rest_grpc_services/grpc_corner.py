import sys
import threading
import grpc
import numpy
import tensorflow as tf
import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from PIL import Image
import json
IMAGE_PATH = r"../dataset/docdata/20200204_101758.jpg"
vizin = Image.open(IMAGE_PATH)
height = vizin.height
vizin = vizin.resize((256,256))
image = np.asarray(vizin).astype('float32')
image = np.expand_dims(image, axis = 0)
tf.compat.v1.app.flags.DEFINE_string('server', '127.0.0.1:8500',
                            'PredictionService host:port')
FLAGS = tf.compat.v1.app.flags.FLAGS
host, port = FLAGS.server.split(':')

def predict(npimage):
  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'corner'
  # request.model_spec.signature_name = 'corner'
  request.inputs['input_1'].CopyFrom(
        tf.make_tensor_proto(npimage, shape=[1,256,256,3]))
  result = stub.Predict(request, 10.0)
  return result

out = predict(image)
print(str(out))

