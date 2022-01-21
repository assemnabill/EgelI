import os.path
import tensorflow as tf
import wget
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import configs
from object_detection.builders import model_builder

def evaluate_model(training_script):
    command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}" \
        .format(training_script, configs.paths['CHECKPOINT_PATH'], configs.files['PIPELINE_CONFIG'],
                configs.paths['CHECKPOINT_PATH'])
    if configs.evaluation_enabled or configs.training_enabled:
        print('Evaluating the model..')
        run_cmd(command)
        # run_cmd(f'cd {os.path.join(configs.paths["CHECKPOINT_PATH"], "train")} && tensorboard --logdir=.')
    else:
        print('Not evaluating the model..')
        print(command)


def train_model(training_script, steps=None):
    command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps={}" \
        .format(training_script,
                configs.paths['CHECKPOINT_PATH'],
                configs.files['PIPELINE_CONFIG'],
                configs.training_steps if steps is None else steps)
    if configs.training_enabled:
        print('Training the model...')
        run_cmd(command)
    else:
        print('Not training the model...')


def config_model():
    print('Copy Model Config to Training Folder..')
    cmd = 'cp {} {}'.format(
        os.path.join(configs.paths["PRETRAINED_MODEL_PATH"], configs.pretrained_model_name, "pipeline.config"),
        os.path.join(configs.paths["CHECKPOINT_PATH"]))
    run_cmd(cmd)

    print('Update Config For Transfer Learning..')
    configs.pipeline_configs = configs.config_util.get_configs_from_pipeline_file(configs.files['PIPELINE_CONFIG'])
    configs.detection_model = model_builder.build(configs.pipeline_configs['model'], False)
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(configs.files['PIPELINE_CONFIG'], "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    pipeline_config.model.ssd.num_classes = len(configs.labels)
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = os.path \
        .join(configs.paths['PRETRAINED_MODEL_PATH'], configs.pretrained_model_name, 'checkpoint', 'ckpt-0')
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
    for label in configs.labels:
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
    tarPath = os.path.join(configs.paths["PRETRAINED_MODEL_PATH"], configs.pretrained_model_name + ".tar.gz")
    if os.path.exists(tarPath):
        print(f'Found pretrained model at {tarPath}')
    else:
        print('Downloading pretrained model..')
        if os.name == 'posix':
            res = run_cmd(f'wget {configs.pretrained_model_url}')
            if res > 0:
                exit(res)
        elif os.name == 'nt':
            res = wget.download(configs.pretrained_model_url)
            if res.__len__() < 1:
                exit(1)
    folderPath = os.path.join(configs.paths["PRETRAINED_MODEL_PATH"], configs.pretrained_model_name)
    if os.path.isdir(folderPath):
        return
    run_cmd(f'mv {configs.pretrained_model_name + ".tar.gz"} {configs.paths["PRETRAINED_MODEL_PATH"]}')
    run_cmd(f'cd {configs.paths["PRETRAINED_MODEL_PATH"]} && mkdir {configs.pretrained_model_name}')
    run_cmd(f'cd {configs.paths["PRETRAINED_MODEL_PATH"]} && tar -zxvf {configs.pretrained_model_name + ".tar.gz"} -C {configs.pretrained_model_name} --strip-components 1')


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


def set_up():
    # print('Download TF Models, Pretrained Models from Tensorflow Model Zoo and Install TFOD')
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


def load_train_model_from_checkpoint(checkpoint):
    import os
    import tensorflow as tf
    from object_detection.builders import model_builder
    from object_detection.utils import config_util

    # Load pipeline config and build a detection model
    pipeline_configs = config_util.get_configs_from_pipeline_file(configs.files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=pipeline_configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(configs.paths['CHECKPOINT_PATH'], checkpoint)).expect_partial()

    return detection_model


def init():
    set_up()

    #verify_installation()

    #upgrade_tf()


def create_records():
    create_labels_map()
    generate_tf_record()


def check_models():
    download_pretrained_models()
    config_model()

def run():
    #init()

    check_models()
    #create_records()

    training_script = os.path.join(configs.paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')

    train_model(training_script)

    evaluate_model(training_script)
