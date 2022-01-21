# EgelI

E-gel-I is an object detection script aimed to identify a defined set of celebrity faces.
The script is developed by Assem Hussein, Jonas Bartkowski as final project for the
module fundamentals of artificial intelligence at THM university of applied science in Giessen, Germany.

## Usage

```
Usage: egeli.py [options]

	 -n, --model-name= 	 set a custom model name

	 -p, --pre-trained=	 the pretrained model to use in training

	 -s, --steps=      	 set the count of steps for training

	 -t, --train=      	 set boolean value to enable training

	 -e, --evaluate=   	 set boolean value to enable evaluation using tensor board

	 -d, --detect=     	 set boolean value to enable detection. Images of faces to be detected should be placed in images/test folder
```

## Add new Pretrained Model

1. Go to [TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
2. Choose a model then copy it's download link
3. Go to configs.py and add it to `pretrained_models_uri` Map
4. Call the Script with `-p <model_name>`
5. Have fun!

## Retrain model from scratch

1. Delete everything in the relevant model folder under `./resources/models/[MODEL_NAME]` except `pipeline.config`.
2. Retrain.
3. Have fun!
