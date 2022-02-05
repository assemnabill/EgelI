import os

from configurator import run_cmd
from records_writer_lib import remove_all, extract_valid_models, get_record_datetime_string


def main():
    test_scores_reports_dir = os.path.join("", "reports", "tests")
    os.makedirs(test_scores_reports_dir, exist_ok=True)
    ignore = []
    models = sorted(remove_all(extract_valid_models(), ignore))
    verbose = False
    models_folder_path = os.path.join("resources", "models")

    for model in models:
        report_file = os.path.join(test_scores_reports_dir,
                                   "test-report" + model + "-" + get_record_datetime_string() + ".txt")
        print("Calculating average test label scores for model "+model +"\n")
        checkpoint_path = os.path.join(models_folder_path, model)
        for file in os.listdir(checkpoint_path):
            if file.endswith(".index"):
                print("\n")
                checkpoint = file[:file.rindex(".index")]
                print("Running tests for ", checkpoint)
                command = "python egeli.py -n {} -x True -r False -c {} -v {} | tee --append {}" \
                    .format(model,
                            checkpoint,
                            verbose,
                            report_file)
                run_cmd(command)

        print("\n\n")


if __name__ == '__main__':
    main()