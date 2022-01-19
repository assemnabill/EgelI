# EgelI

E-gel-I is an object detection script aimed to identify a defined set of celebrity faces.
The script is developed by Assem Hussein, Jonas Bartkowski as final project for the
module fundamentals of artificial intelligence at THM university of applied science in Giessen, Germany.

## Add new Pretrained Model

1. Go to [TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
1. Choose a model then copy it's download link
1. Go to configs.py and add it to `pretrained_models_uri` Map
1. Call the Script with `-p <model_name>`
1. Have fun!
