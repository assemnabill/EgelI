import os

#labels = ['OlafScholz', 'AnnalenaBaerbock', 'ChristianLindner', 'JoeBiden', 'KamalaHarris', 'EmmanuelMacron', 'WladimirPutin', 'ElonMusk', 'JeffBezos', 'KarlLauterbach', 'LeonardoDiCaprio', 'OliverWelke', 'JohnOliver', 'JanBöhmermann', 'TrevorNoah', 'JenniferLawrence']
labels = ['AnnalenaBaerbock', 'ChristianLindner', 'ElonMusk']


models_repo = {
    'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz',
    'ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz'
}

pipeline_configs = None
detection_model = None
checkpoint = None
training_enabled = False
evaluation_enabled = False
detection_enabled = False
save_plots = False
install_api = False
generate_records = True
random_detection = False

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





