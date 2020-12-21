"""Make predictions using a frozen TensorFlow model."""

import numpy as np
import tensorflow as tf

# Model class IDs and corresponding traffic light states
# Note that this is different assignment from styx.msgs.msg.TrafficLight.
# Using constants instead of `enum` as Udacity Workspace doesn't have `enum`
# package installed by default.
TRAFFIC_LIGHT_STATE_GREEN = 1
TRAFFIC_LIGHT_STATE_RED = 2
TRAFFIC_LIGHT_STATE_YELLOW = 3
TRAFFIC_LIGHT_STATE_UNKNOWN = 4


def load_graph(model_file):
  """Loads a TensorFlow graph from file."""

  graph = tf.Graph()
  with graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_file, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
  return graph


def image_to_pred_input(image):
  """Converts a PIL image to Numpy array input for prediction.

  Args:
      image: Numpy array-like RGB image (e.g. PIL Image).

  Returns:
      Numpy array with shape (1, height, width, channels).
  """

  return np.expand_dims(image, 0)


class FrozenModel:
  """Frozen TensorFlow model."""

  OUTPUT_TENSOR_NAMES = [
      'detection_boxes',
      'detection_scores',
      'detection_classes',
      'num_detections',
  ]
  INPUT_TENSOR_NAME = 'image_tensor'

  SCORE_THRESHOLD = 0.5

  @classmethod
  def load_output_tensors(cls, graph):
    """Loads pertinent output tensors for prediction.

    Args:
        graph: TensorFlow graph.

    Returns:
        Dict of {tensor_name: tensor} for each tensor_name in
        OUTPUT_TENSOR_NAMES.
    """

    return {
        tensor_name: graph.get_tensor_by_name(tensor_name + ':0')
        for tensor_name in cls.OUTPUT_TENSOR_NAMES}

  @classmethod
  def load_input_tensor(cls, graph):
    """Returns input tensor for prediction.

    Args:
      graph: TensorFlow graph.

    Returns:
      Tensor representing model input.
    """
    return graph.get_tensor_by_name(cls.INPUT_TENSOR_NAME + ':0')

  def __init__(self, model_file):

    # Load model graph
    self.graph = load_graph(model_file)

    # Load input tensor
    self.input_tensor = self.load_input_tensor(self.graph)

    # Load output tensors
    self.output_tensors =self.load_output_tensors(self.graph)

  def reduce_predictions(self, scores, classes):
      """Reduces predictions to one class."""

      output_class = TRAFFIC_LIGHT_STATE_UNKNOWN
      for score, class_id in zip(scores, classes):
        if score > self.SCORE_THRESHOLD:
          # Return RED immediately if detected
          if class_id == TRAFFIC_LIGHT_STATE_RED:
            return TRAFFIC_LIGHT_STATE_RED

          # Any detected YELLOW light takes priority over any GREEN
          if output_class != TRAFFIC_LIGHT_STATE_YELLOW:
            output_class = class_id

      return output_class

  def predict(self, image):
      """Get model predictions for a single image.

      Args:
          image: Numpy array-like RGB image (e.g. PIL Image).

      Returns:
          int: one of `TRAFFIC_LIGHT_STATE_*`
      """

      np_image = image_to_pred_input(image)
      with tf.Session(graph=self.graph) as sess:
        (scores, classes) = sess.run(
            [
                self.output_tensors['detection_scores'],
                self.output_tensors['detection_classes'],
            ],
            feed_dict={self.input_tensor: np_image})

      # Trim extra dimensions
      scores = np.squeeze(scores)
      classes = np.squeeze(classes)

      # Reduce detections to one class
      return self.reduce_predictions(scores, classes)
