import os
import sys

from records_lib import remove_all, extract_valid_models, Logger, get_record_datetime_string


def run_cmd(cmd):
    print(f'Running {cmd}')
    return os.system(cmd)


def main():
    test_scores_reports_dir = os.path.join("", "reports", "tests")
    os.makedirs(test_scores_reports_dir, exist_ok=True)
    ignore = remove_all(extract_valid_models(), ["my_custom_model-ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"])
    models = sorted(remove_all(extract_valid_models(), ignore))
    verbose = False
    models_folder_path = os.path.join("resources", "models")

    for model in models:
        sys.stdout = Logger(os.path.join(test_scores_reports_dir,
                                   "test-report" + model + "-" + get_record_datetime_string() + ".txt"))
        print("\n\nCalculating average test label scores for model", model + "\n\n\n")
        checkpoint_path = os.path.join(models_folder_path, model)
        for file in os.listdir(checkpoint_path):
            if file.endswith(".index"):
                print("\n\n")
                checkpoint = file[:file.rindex(".index")]

                command = "python egeli.py -n {} -x True -r False -c {} -v {} | tee --append {}" \
                    .format(model,
                            checkpoint,
                            verbose,
                            report_file)
                run_cmd(command)


if __name__ == '__main__':
    main()