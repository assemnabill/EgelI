import os

labels = ['OlafScholz', 'AnnalenaBaerbock', 'ChristianLindner', 'JoeBiden', 'KamalaHarris', 'EmmanuelMacron', 'WladimirPutin', 'ElonMusk', 'JeffBezos', 'KarlLauterbach', 'LeonardoDiCaprio', 'OliverWelke', 'JohnOliver', 'JanBoehmermann', 'TrevorNoah', 'JenniferLawrence']


models_repo = {
    'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz',
    'ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz',
    'ssd_resnet152_v1_fpn_640x640_coco17_tpu-8':  'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.tar.gz',
    'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz',
    'efficientdet_d1_coco17_tpu-32': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz',
    'efficientdet_d0_coco17_tpu-32': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz'
}

pipeline_configs = None
# will be set on runtime.
detection_model = None
checkpoint = None
training_enabled = False
evaluation_enabled = False
detection_enabled = False
save_plots = False
install_api = False
generate_records = False
random_detection = False
report_average_test_scores = False
verbose = False

training_steps = 2000
detection_threshold = 0.8
pretrained_model_name = 'ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8'
pretrained_model_url = models_repo[pretrained_model_name]
custom_model_name = f'my_custom_model-{pretrained_model_name}'
IMAGES_PATH = os.path.join('resources', 'images', 'collected images')
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'
paths = {}
files = {
    'PIPELINE_CONFIG': os.path.join('resources', 'models', custom_model_name, 'pipeline.config')
}





