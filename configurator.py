import os.path
import sys
import tensorflow as tf
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import configs
import object_detection.builders as model_builder
from object_detection.utils import config_util
from egeli import run_cmd


def set_up_object_detection_api():
    print('Setting up object detection api..')
    for path in configs.paths.values():
        if not os.path.exists(path):
            if os.name == 'posix':
                run_cmd(f'mkdir -p {path}')
            if os.name == 'nt':
                run_cmd(f'mkdir {path}')

    path = os.path.join(configs.paths['APIMODEL_PATH'], 'research', 'object_detection')
    if not os.path.exists(path):
        print('Found no Object detection Models \nCloning models from TF Repo..')
        cmd = 'git clone {} {}'.format("https://github.com/tensorflow/models", configs.paths["APIMODEL_PATH"])
        run_cmd(cmd)
    else:
        print(f'Object detection Models found in {path}')

    print('Downloading TF Object detection Models..')
    cmd = 'cd {} && protoc {} --python_out=. && cp {} . && python -m pip install .' \
        .format("models/research", "object_detection/protos/*.proto",
                "object_detection/packages/tf2/setup.py")
    run_cmd(cmd)


def upgrade_tf():
    print('Upgrading TF..')
    cmd = 'pip install tensorflow --upgrade'
    run_cmd(cmd)


def config_transfer_learning_pipeline():
    print('Copy Model Config to Training Folder..')
    run_cmd(f'mkdir {configs.paths["CHECKPOINT_PATH"]}')
    cmd = 'cp {} {}'.format(
        os.path.join(configs.paths["PRETRAINED_MODEL_PATH"], configs.pretrained_model_name, "pipeline.config"),
        os.path.join(configs.paths["CHECKPOINT_PATH"]))
    run_cmd(cmd)
    from object_detection.builders import model_builder
    from object_detection.utils import config_util
    print('Update Config For Transfer Learning..')
    configs.pipeline_configs = config_util.get_configs_from_pipeline_file(configs.files['PIPELINE_CONFIG'])
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


def verify_installation():
    print('Verify Installation..')
    verification_script = os.path.join(configs.paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders',
                                       'model_builder_tf2_test.py')
    cmd = 'python {}'.format(verification_script)
    status_code = run_cmd(cmd)
    print(f'process exited with status code {status_code}')


def download_pretrained_models():
    tar_name = get_model_tar_name()
    if os.path.exists(tar_name):
        print(f'Found pretrained model {tar_name}')
    else:
        print('Downloading pretrained model..')
        if os.name == 'posix':
            run_cmd(f'wget {configs.pretrained_model_url}')
        if os.name == 'nt':
            run_cmd('pip install wget')
            import wget
            res = wget.download(configs.pretrained_model_url, out=configs.paths["PRETRAINED_MODEL_PATH"])
            if res.__len__() < 1:
                exit(1)
    folder_path = os.path.join(configs.paths["PRETRAINED_MODEL_PATH"], configs.pretrained_model_name)
    if os.path.isdir(folder_path):
        return
    run_cmd(f'cd {configs.paths["PRETRAINED_MODEL_PATH"]} && mkdir {configs.pretrained_model_name}')
    run_cmd(f'cd {configs.paths["PRETRAINED_MODEL_PATH"]} '
            f'&& tar -zxvf {tar_name} -C {configs.pretrained_model_name} --strip-components 1')


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


def generate_tf_records():
    if not os.path.exists(configs.files['TF_RECORD_SCRIPT']):
        print('Downloading tf record generator script..')
        cmd = f'git clone https://github.com/nicknochnack/GenerateTFRecord {configs.paths["SCRIPTS_PATH"]}'
        run_cmd(cmd)

    if not os.path.exists(configs.files["LABELMAP"]):
        print("Label map not existing! Creating!")
        create_labels_map()
    else:
        key = input("Already found label map! (" + configs.files["LABELMAP"] + ") Do you want to recreate it? (Y|n)")
        if not str(key).lower() == "n":
            create_labels_map()

    print('Generating tf record..')
    train_images_path = os.path.join(configs.paths["IMAGE_PATH"], "train")
    if not os.path.exists(train_images_path):
        print("No train images folder found! (" + train_images_path + ")! Aborting generation of tf_record!",
              file=sys.stderr)

    train_record_path = os.path.join(configs.paths["ANNOTATION_PATH"], "train.record")
    if os.path.exists(train_record_path):
        key = input("Already found train.record! (" + train_record_path + ") Do you want to recreate it? (Y|n)")
        if not str(key).lower() == "n":
            cmd = 'python {} -x {} -l {} -o {}' \
                .format(configs.files["TF_RECORD_SCRIPT"], train_images_path,
                        configs.files["LABELMAP"], train_record_path)
            run_cmd(cmd)
        else:
            print("Not creating train.record!")

    test_images_path = os.path.join(configs.paths["IMAGE_PATH"], "test")
    if not os.path.exists(test_images_path):
        print("No test images folder found! (" + test_images_path + ")! Aborting generation of tf_record!",
              file=sys.stderr)

    test_record_path = os.path.join(configs.paths["ANNOTATION_PATH"], "test.record")
    if os.path.exists(train_record_path):
        key = input("Already found test.record! (" + test_record_path + ") Do you want to recreate it? (Y|n)")
        if not str(key).lower() == "n":
            cmd = 'python {} -x {} -l {} -o {}' \
                .format(configs.files["TF_RECORD_SCRIPT"], test_images_path,
                        configs.files["LABELMAP"], test_record_path)
            run_cmd(cmd)
        else:
            print("Not creating test.record...")


def get_model_tar_name():
    tar_name = str(configs.models_repo[configs.pretrained_model_name])
    tar_name = tar_name[tar_name.rindex("/") + 1:]
    return tar_name


def init():
    print(tf.__version__)
    if configs.install_api:
        set_up_object_detection_api()
        upgrade_tf()
        verify_installation()
    download_pretrained_models()
    config_transfer_learning_pipeline()
    if configs.generate_records:
        generate_tf_records()
