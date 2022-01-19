import os.path
import tensorflow as tf
import wget
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import configs
from configs import *


def evaluate_model(training_script):
    print('Evaluating the model..')
    command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}" \
        .format(training_script, configs.paths['CHECKPOINT_PATH'], configs.files['PIPELINE_CONFIG'],
                configs.paths['CHECKPOINT_PATH'])
    if configs.evaluation_enabled:
        run_cmd(command)
        # run_cmd(f'cd {os.path.join(configs.paths["CHECKPOINT_PATH"], "train")} && tensorboard --logdir=.')
    else:
        print(command)


def train_model(training_script):
    print('Training the model...')
    command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps={}" \
        .format(training_script, configs.paths['CHECKPOINT_PATH'], configs.files['PIPELINE_CONFIG'], configs.training_steps)
    if configs.training_enabled:
        run_cmd(command)
    else:
        print(command)


def config_model():
    print('Copy Model Config to Training Folder..')
    cmd = 'cp {} {}'.format(
        os.path.join(configs.paths["PRETRAINED_MODEL_PATH"], pretrained_model_name, "pipeline.config"),
        os.path.join(configs.paths["CHECKPOINT_PATH"]))
    run_cmd(cmd)

    print('Update Config For Transfer Learning..')
    configs.pipeline_configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(configs.files['PIPELINE_CONFIG'], "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    pipeline_config.model.ssd.num_classes = len(labels)
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = os.path \
        .join(configs.paths['PRETRAINED_MODEL_PATH'], pretrained_model_name, 'checkpoint', 'ckpt-0')
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path = configs.files['LABELMAP']
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
        os.path.join(configs.paths['ANNOTATION_PATH'], 'train.record')]
    pipeline_config.eval_input_reader[0].label_map_path = configs.files['LABELMAP']
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
        os.path.join(configs.paths['ANNOTATION_PATH'], 'test.record')]

    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(configs.files['PIPELINE_CONFIG'], "wb") as f:
        f.write(config_text)


def generate_tf_record():
    if not os.path.exists(configs.files['TF_RECORD_SCRIPT']):
        print('Downloading tf record generator script..')
        cmd = f'git clone https://github.com/nicknochnack/GenerateTFRecord {configs.paths["SCRIPTS_PATH"]}'
        run_cmd(cmd)

    print('Generating tf record..')
    cmd = 'python {} -x {} -l {} -o {}' \
        .format(configs.files["TF_RECORD_SCRIPT"], os.path.join(configs.paths["IMAGE_PATH"], "train"),
                configs.files["LABELMAP"], os.path.join(configs.paths["ANNOTATION_PATH"], "train.record"))
    run_cmd(cmd)
    cmd = 'python {} -x {} -l {} -o {}' \
        .format(configs.files["TF_RECORD_SCRIPT"], os.path.join(configs.paths["IMAGE_PATH"], "test"),
                configs.files["LABELMAP"], os.path.join(configs.paths["ANNOTATION_PATH"], "test.record"))
    run_cmd(cmd)


def create_labels_map():
    print('Creating label map..')
    labels_map = []
    tmp_id = 1
    for label in labels:
        labels_map.append({'name': label, 'id': tmp_id})
        tmp_id += 1
    if os.path.exists(configs.files['LABELMAP']):
        os.system(f'rm {configs.files["LABELMAP"]}')
    with open(configs.files['LABELMAP'], 'x') as f:
        for label in labels_map:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')


def download_pretrained_models():
    path = os.path.join(configs.paths["PRETRAINED_MODEL_PATH"], configs.pretrained_model_name + ".tar.gz")
    if os.path.exists(path):
        print(f'Found pretrained model at {path}')
        return
    print('Downloading pretrained model..')
    if os.name == 'posix':
        res = run_cmd(f'wget {configs.pretrained_model_url}')
        if res > 0:
            exit(res)
    elif os.name == 'nt':
        res = wget.download(configs.pretrained_model_url)
        if res.__len__() < 1:
            exit(1)
    run_cmd(f'mv {configs.pretrained_model_name + ".tar.gz"} {configs.paths["PRETRAINED_MODEL_PATH"]}')
    run_cmd(f'cd {configs.paths["PRETRAINED_MODEL_PATH"]} && tar -zxvf {configs.pretrained_model_name + ".tar.gz"}')


def verify_installation(installed_protobuf=False):
    print('Verify Installation..')
    verification_script = os.path.join(configs.paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders',
                                       'model_builder_tf2_test.py')
    cmd = 'python {}'.format(verification_script)
    status_code = run_cmd(cmd)
    print(f'process exited with status code {status_code}')

    if (not installed_protobuf) & (status_code > 0):
        run_cmd('pip uninstall protobuf matplotlib -y')
        run_cmd('pip install protobuf matplotlib==3.2')
        verify_installation(True)


def run_cmd(cmd):
    print(f'Running {cmd}')
    return os.system(cmd)


def upgrade_tf():
    print('Upgrading TF..')
    cmd = 'pip install tensorflow --upgrade'
    run_cmd(cmd)


def run():
    for path in configs.paths.values():
        if not os.path.exists(path):
            if os.name == 'posix':
                run_cmd(f'mkdir -p {path}')
            if os.name == 'nt':
                run_cmd(f'mkdir {path}')

    path = os.path.join(configs.paths['APIMODEL_PATH'], 'research', 'object_detection')
    if not os.path.exists(path):
        print('Found no Object detection Models')

        print('cloning models from TF Repo..')
        cmd = 'git clone {} {}'.format("https://github.com/tensorflow/models", configs.paths["APIMODEL_PATH"])
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

    training_script = os.path.join(configs.paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')

    train_model(training_script)

    evaluate_model(training_script)
