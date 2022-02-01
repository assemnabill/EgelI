# E-gel-I is an object detection script aimed to identify a defined set of celebrity faces.
# The script is developed by Assem Hussein, Jonas Bartkowski as final project for the
# module fundamentals of artificial intelligence at THM university of applied science in Giessen, Germany.

import getopt
import sys
import configs
import configurator
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
    print('\t -n, --model-name=   \t set a custom model name\n')
    print('\t -m, --pre-trained=  \t the pretrained model to use in training.\n'
          '\t\t\t\t\t\t\t This is set to ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8 by default.\n')
    print('\t -s, --steps=        \t set the count of steps for training\n')
    print('\t -t, --train=        \t set boolean value to enable training\n'
          '\t\t\t\t\t\t\t This is set to False by default.\n')
    print('\t -e, --evaluate=     \t set boolean value to enable evaluation using tensor board\n'
          '\t\t\t\t\t\t\t This is set to False by default.\n')
    print('\t -p, --save-plots=   \t set boolean value to save plots after detection\n'
          '\t\t\t\t\t\t\t This is set to False by default.\n')
    print('\t -d, --detect=       \t set boolean value to enable detection.\n'
          '\t\t\t\t\t\t\t Images must be at resources/images/test folder.\n')
    print('\t -i, --installation= \t set boolean value to install object detection api.\n'
          '\t\t\t\t\t\t\t This is set to False by default.\n')
    print('\t -r, --generation=   \t set boolean value to disable tf record generation. ')
    print('\t -c, --checkpoint=   \t set checkpoint to detect from. ')
    print('\t -o, --threshold=    \t set detection threshold (minimum score from which a label is drawn).\n'
          '\t\t\t\t\t\t\t This is set to 0.8 by default.\n')
    print('\t -a, --random=       \t enable random sequencing when detecting from test folder.\n'
          '\t\t\t\t\t\t\t This is set to False by default.\n')
    print('\t -p, --average-test-scores=       \t Generate report on average detection scores of labels in test files.\n'
          '\t\t\t\t\t\t\t This is set to False by default.\n')
    print(
        '\t -v, --verbose=       \t Enable more detailed output.\n'
        '\t\t\t\t\t\t\t This is set to False by default.\n')


def run_cmd(cmd):
    print(f'Running {cmd}')
    return os.system(cmd)


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h:i:n:m:p:s:t:e:d:v:r:c:o:a:p:",
                                   ["--model-name=", "--pre-trained=", "--steps=", "--train=", "--evaluate=",
                                    "--detect=", "--save-plots=", "--installation=", "--generation=",
                                    "--checkpoint=", "--threshold=", "--random=", "--average-test-scores=",
                                    "--verbose="])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-n", "--model-name"):
            configs.custom_model_name = arg
        elif opt in ("-m", "--pre-trained"):
            configs.pretrained_model_name = arg
            configs.custom_model_name = f'my_custom_model-{configs.pretrained_model_name}'
            if configs.pretrained_model_name in models_repo:
                configs.pretrained_model_url = models_repo[configs.pretrained_model_name]
            else:
                print("ERROR. Couldn't find", configs.pretrained_model_name, "in models repo. Valid keys for -m are:", models_repo.keys())
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
        elif opt in ("-i", "--installation"):
            configs.install_api = parse_boolean(arg)
        elif opt in ("-r", "--generation"):
            configs.generate_records = parse_boolean(arg)
        elif opt in ("-c", "--checkpoint"):
            configs.checkpoint = arg
        elif opt in ("-o", "--threshold"):
            configs.detection_threshold = float(arg)
        elif opt in ("-a", "--random"):
            configs.random_detection = parse_boolean(arg)
        elif opt in ("-p", "--average-test-scores"):
            configs.report_average_test_scores = parse_boolean(arg)
        elif opt in ("-v", "--verbose"):
            configs.verbose = parse_boolean(arg)

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

    configurator.init()
    configurator.create_labels_map()

    if configs.training_enabled:
        trainer.run()
    if configs.evaluation_enabled:
        trainer.evaluate_model()
    if configs.detection_enabled:
        detector.detect_and_display_test_images(max_boxes=5)
    if configs.report_average_test_scores:
        detector.calculate_average_score_for_test_labels(verbose=configs.verbose)


if __name__ == '__main__':
    main(sys.argv[1:])
