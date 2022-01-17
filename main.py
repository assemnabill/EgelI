# E-gel-I is an object detection project aimed to identify a set of celebrity faces.
# The Project is developed by Assem Hussein, Jonas Bartkowski as final project for the
# module fundamentals of artificial intelligence at THM university of applied science in Giessen, Germany.


# Import opencv
import this

import cv2

# Import uuid
import uuid

# Import Operating System
import os
import object_detection
# Import time
import time
import wget
from numpy.distutils.system_info import p

# Update Config For Transfer Learning
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

labels = ['Olaf Scholz', 'Annalena Baerbock', 'Christian Lindner',
          'Joe Biden', 'Kamala Harris', 'Emmanuel Macron', 'Wladimir Putin',
          'Elon Musk', 'Jeff Bezos', 'Karl Lauterbach', 'Leonardo DiCaprio',
          'Oliver Welke', 'John Oliver', 'Jan Böhmermann',
          'Trevor Noah', 'Jennifer Lawrence']
number_imgs = 5
IMAGES_PATH = os.path.join('resources', 'images', 'collected images')
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'
paths = {
    'WORKSPACE_PATH': os.path.join('resources'),
    'SCRIPTS_PATH': os.path.join('scripts'),
    'APIMODEL_PATH': os.path.join('models'),
    'ANNOTATION_PATH': os.path.join('resources', 'annotations'),
    'IMAGE_PATH': os.path.join('resources', 'images'),
    'MODEL_PATH': os.path.join('resources', 'models'),
    'PRETRAINED_MODEL_PATH': os.path.join('resources', 'pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('resources', 'models', CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join('resources', 'models', CUSTOM_MODEL_NAME, 'export'),
    'TFJS_PATH': os.path.join('resources', 'models', CUSTOM_MODEL_NAME, 'tfjsexport'),
    'TFLITE_PATH': os.path.join('resources', 'models', CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH': os.path.join('protoc')
}

files = {
    'PIPELINE_CONFIG': os.path.join('resources', 'models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}


def evaluate_model(training_script):
    print('Evaluating the model..')
    command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}" \
        .format(training_script, paths['CHECKPOINT_PATH'], files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'])
    print(command)


def train_model(training_script, steps=2000):
    print('Training the model...')
    command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps={}" \
        .format(training_script, paths['CHECKPOINT_PATH'], files['PIPELINE_CONFIG'], steps)
    print(command)


def config_model():
    print('Copy Model Config to Training Folder..')
    cmd = 'cp {} {}'.format(
        os.path.join(paths["PRETRAINED_MODEL_PATH"], PRETRAINED_MODEL_NAME, "pipeline.config"),
        os.path.join(paths["CHECKPOINT_PATH"]))
    run_cmd(cmd)

    print('Update Config For Transfer Learning..')
    config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    pipeline_config.model.ssd.num_classes = len(labels)
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = os.path \
        .join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path = files['LABELMAP']
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
        os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
    pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
        os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:
        f.write(config_text)


def generate_tf_record():
    if not os.path.exists(files['TF_RECORD_SCRIPT']):
        print('Downloading tf record generator script..')
        cmd = f'git clone https://github.com/nicknochnack/GenerateTFRecord {paths["SCRIPTS_PATH"]}'
        run_cmd(cmd)

    print('Generating tf record..')
    cmd = 'python {} -x {} -l {} -o {}' \
        .format(files["TF_RECORD_SCRIPT"], os.path.join(paths["IMAGE_PATH"], "train"),
                files["LABELMAP"], os.path.join(paths["ANNOTATION_PATH"], "train.record"))
    run_cmd(cmd)
    cmd = 'python {} -x {} -l {} -o {}' \
        .format(files["TF_RECORD_SCRIPT"], os.path.join(paths["IMAGE_PATH"], "test"),
                files["LABELMAP"], os.path.join(paths["ANNOTATION_PATH"], "test.record"))
    run_cmd(cmd)


def create_labels_map():
    print('Creating label map..')
    labels_map = []
    tmp_id = 1
    for label in labels:
        labels_map.append({'name': label, 'id': tmp_id})
        tmp_id += 1

    with open(files['LABELMAP'], 'w') as f:
        for label in labels_map:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')


def download_pretrained_models():
    print('Downloading pretrained model..')
    run_cmd(f'wget {PRETRAINED_MODEL_URL}')
    run_cmd(f'mv {PRETRAINED_MODEL_NAME + ".tar.gz"} {paths["PRETRAINED_MODEL_PATH"]}')
    run_cmd(f'cd {paths["PRETRAINED_MODEL_PATH"]} && tar -zxvf {PRETRAINED_MODEL_NAME + ".tar.gz"}')


def verify_installation():
    print('Verify Installation..')
    verification_script = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders',
                                       'model_builder_tf2_test.py')
    cmd = 'python {}'.format(verification_script)
    status_code = run_cmd(cmd)
    print(f'process exited with status code {status_code}')
    # In case Vf Script fails, run the following
    # !pip uninstall protobuf matplotlib -y
    # !pip install protobuf matplotlib==3.2


def run_cmd(cmd):
    print(f'Running {cmd}')
    return os.system(cmd)


def upgrade_tf():
    print('Upgrading TF..')
    cmd = 'pip install tensorflow --upgrade'
    run_cmd(cmd)


def run():
    # print('Download TF Models, Pretrained Models from Tensorflow Model Zoo and Install TFOD')
    for path in paths.values():
        if not os.path.exists(path):
            if os.name == 'posix':
                run_cmd(f'mkdir -p {path}')
            if os.name == 'nt':
                run_cmd(f'mkdir {path}')

    path = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')
    if not os.path.exists(path):
        print('Found no Object detection Models')

        print('cloning models from TF Repo..')
        cmd = 'git clone {} {}'.format("https://github.com/tensorflow/models", paths["APIMODEL_PATH"])
        run_cmd(cmd)
    else:
        print(f'Object detection Models found in {path}')

    print('Downloading TF Object detection Models..')
    cmd = 'cd {} && protoc {} --python_out=. && cp {} . && python -m pip install .' \
        .format("models/research", "object_detection/protos/*.proto",
                "object_detection/packages/tf2/setup.py")
    run_cmd(cmd)

    verify_installation()

    upgrade_tf()

    download_pretrained_models()

    create_labels_map()

    generate_tf_record()

    config_model()

    training_script = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')

    train_model(training_script)

    evaluate_model(training_script)


if __name__ == '__main__':
    run()
