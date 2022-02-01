import os.path
import configs
import os
from timeit import default_timer as timer


def run_cmd(cmd):
    print(f'Running {cmd}')
    return os.system(cmd)


def evaluate_model(training_script=None):
    if training_script is None:
        training_script = os.path.join(configs.paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
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


def train_model(training_script=None, steps=None):
    if training_script is None:
        training_script = os.path.join(configs.paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
    command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps={} " \
        .format(training_script,
                configs.paths['CHECKPOINT_PATH'],
                configs.files['PIPELINE_CONFIG'],
                configs.training_steps if steps is None else steps)
    if configs.training_enabled:
        print('Training the model...')
        start = timer()
        run_cmd(command)
        end = timer()
        print("Training took", (end - start), "seconds!")
    else:
        print('Not training the model...')


def run():
    train_model()
    evaluate_model()
