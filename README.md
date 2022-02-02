# EgelI

E-gel-I is an object detection script aimed to identify a defined set of celebrity faces.
The script is developed by Assem Hussein, Jonas Bartkowski as final project for the
module fundamentals of artificial intelligence at THM university of applied science in Giessen, Germany.



## Quick Start

<b>Step 1.</b> Clone this repository and download [egele.zip](https://drive.google.com/file/d/1HB3BQ8FX2TaK-sE6V8RShdniWzM1_OCG/view?usp=sharing)
<br/><br/>
<b>Step 2.</b> Create a new virtual environment

extract egele.zip then run
```shell
python -m venv egele
```

<br/>
<b>Step 3.</b> Activate your virtual environment

```shell
source egele/bin/activate # Linux
.\egele\Scripts\activate # Windows 
```

<br/>

## Usage

```
Usage: egeli.py [options]

	 -n, --model-name=   	 set a custom model name

	 -m, --pre-trained=  	 the pretrained model to use in training.
				 This is set to ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8 by default.

	 -s, --steps=        	 set the count of steps for training

	 -t, --train=        	 set boolean value to enable training
				 This is set to False by default.

	 -e, --evaluate=     	 set boolean value to enable evaluation using tensor board
				 This is set to False by default.

	 -p, --save-plots=   	 set boolean value to save plots after detection
				 This is set to False by default.

	 -d, --detect=       	 set boolean value to enable detection.
				 Images must be at resources/images/test folder.

	 -i, --installation= 	 set boolean value to install object detection api.
				 This is set to False by default.

	 -r, --generation=   	 set boolean value to disable tf record generation. 
	 -c, --checkpoint=   	 set checkpoint to detect from. 
	 -o, --threshold=    	 set detection threshold (minimum score from which a label is drawn).
				 This is set to 0.8 by default.

	 -a, --random=       	 enable random sequencing when detecting from test folder.
				 This is set to False by default.

	 -x, --test-scores==     Generate report on average detection scores of labels in test files.
				 
	 -v, --verbose=       	 Enable more detailed output.
```

## Add new Pretrained Model

1. Go to [TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
2. Choose a model then copy it's download link
3. Go to configs.py and add it to `pretrained_models_uri` Map
4. Call the Script with `-n <model_name>`
5. Have fun!

## Supported Pretrained Models:

1. ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8
2. ssd_resnet152_v1_fpn_640x640_coco17_tpu-8
3. ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
4. efficientdet_d1_coco17_tpu-32
5. efficientdet_d0_coco17_tpu-32

## Retrain model from scratch

1. Delete everything in the relevant model folder under `./resources/models/[MODEL_NAME]` except `pipeline.config`.
2. Retrain.
3. Have fun!
