import unittest
import yaml

from light_classification import prediction


class FrozenModelTest(unittest.TestCase):
  """Tests `FronzeModel` class."""

  @classmethod
  def setUpClass(cls):
      # Load ROS config file for simulator
      config_file = 'sim_traffic_light_config.yaml'
      with open(config_file) as f:
        cls.CONFIG = yaml.load(f, Loader=yaml.FullLoader)

  def setUp(self):
      self.model = prediction.FrozenModel(self.CONFIG['tensorflow_model'])

  def test_reduce_predictions_all_red(self):
      """Tests `reduce_predictions` on all red lights."""

      scores = [0.99, 0.49, 0.51]
      classes = [2.0, 2.0, 2.0]
      output_class = self.model.reduce_predictions(scores, classes)
      self.assertEqual(output_class, prediction.TRAFFIC_LIGHT_STATE_RED)

  def test_reduce_predictions_one_yellow(self):
      """Tests `reduce_predictions` on one yellow light."""

      scores = [0.99, 0.49, 0.51]
      classes = [1.0, 1.0, 3.0]
      output_class = self.model.reduce_predictions(scores, classes)
      self.assertEqual(output_class, prediction.TRAFFIC_LIGHT_STATE_YELLOW)


if __name__ == '__main__':
  unittest.main()
