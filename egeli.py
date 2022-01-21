# E-gel-I is an object detection script aimed to identify a defined set of celebrity faces.
# The script is developed by Assem Hussein, Jonas Bartkowski as final project for the
# module fundamentals of artificial intelligence at THM university of applied science in Giessen, Germany.

import getopt
import sys
import configs
import detector
import trainer
from configs import *


def parse_boolean(value):
    value = value.lower()
    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False
    return False


def usage():
    print('Usage: egeli.py [options]\n')
    print('\t -n, --model-name= \t set a custom model name\n')
    print('\t -p, --pre-trained=\t the pretrained model to use in training\n')
    print('\t -s, --steps=      \t set the count of steps for training\n')
    print('\t -t, --train=      \t set boolean value to enable training\n')
    print('\t -e, --evaluate=   \t set boolean value to enable evaluation using tensor board\n')
    print('\t -d, --detect=     \t set boolean value to enable detection. ')
    print('\t -v, --verify-install=     \t set boolean value to disable installation verification. ')
    print('\t -r, --generate-records=     \t set boolean value to disable tf record generation. '
          'Images of faces to be detected should be placed in images/test folder\n')


def run_cmd(cmd):
    print(f'Running {cmd}')
    return os.system(cmd)


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h:i:n:m:p:s:t:e:d:v:r:",
                                   ["--model-name=", "--pre-trained=", "--steps=", "--train=", "--evaluate=",
                                    "--detect=", "--save-plots=", "--verify-install=", "--generate-records="])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-n", "--model-name"):
            configs.pretrained_model_name = arg
            configs.custom_model_name = f'my_custom_model-{configs.pretrained_model_name}'
            configs.pretrained_model_url = pretrained_models_uri[configs.pretrained_model_name]
        elif opt in ("-m", "--pre-trained"):
            configs.pretrained_model_name = arg
            configs.custom_model_name = f'my_custom_model-{configs.pretrained_model_name}'
            configs.pretrained_model_url = pretrained_models_uri[configs.pretrained_model_name]
        elif opt in ("-s", "--steps"):
            configs.training_steps = arg
        elif opt in ("-t", "--train"):
            configs.training_enabled = parse_boolean(arg)
        elif opt in ("-e", "--evaluate"):
            configs.evaluation_enabled = parse_boolean(arg)
        elif opt in ("-d", "--detect"):
            configs.detection_enabled = parse_boolean(arg)
        elif opt in ("-p", "--save-plots"):
            configs.save_plots = parse_boolean(arg)
        elif opt in ("-v", "--verify-install"):
            configs.verify_installation = parse_boolean(arg)
        elif opt in ("-r", "--generate-records"):
            configs.generate_records = parse_boolean(arg)

    print(f'Model name is  {configs.custom_model_name}')
    print(f'Pretrained model => {configs.pretrained_model_name}')

    configs.paths = {
        'WORKSPACE_PATH': os.path.join('resources'),
        'SCRIPTS_PATH': os.path.join('scripts'),
        'APIMODEL_PATH': os.path.join('models'),
        'ANNOTATION_PATH': os.path.join('resources', 'annotations'),
        'IMAGE_PATH': os.path.join('resources', 'images'),
        'MODEL_PATH': os.path.join('resources', 'models'),
        'PRETRAINED_MODEL_PATH': os.path.join('resources', 'pre-trained-models'),
        'CHECKPOINT_PATH': os.path.join('resources', 'models', configs.custom_model_name),
        'OUTPUT_PATH': os.path.join('resources', 'models', configs.custom_model_name, 'export'),
        'TFJS_PATH': os.path.join('resources', 'models', configs.custom_model_name, 'tfjsexport'),
        'TFLITE_PATH': os.path.join('resources', 'models', configs.custom_model_name, 'tfliteexport'),
        'PROTOC_PATH': os.path.join('protoc')
    }

    configs.files = {
        'PIPELINE_CONFIG': os.path.join('resources', 'models', configs.custom_model_name, 'pipeline.config'),
        'TF_RECORD_SCRIPT': os.path.join(configs.paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
        'LABELMAP': os.path.join(configs.paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
    }

    trainer.run()
    if configs.detection_enabled:
        detector.run()


if __name__ == '__main__':
    main(sys.argv[1:])
