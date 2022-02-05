import os


def run_cmd(cmd):
    print(f'Running {cmd}')
    return os.system(cmd)


def main():
    test_scores_reports_dir = os.path.join("reports", "tests")
    os.makedirs(test_scores_reports_dir, exist_ok=True)
    models = os.listdir(os.path.join("resources", "models"))
    verbose = False
    for model in models:
        print("\n\nCalculating average test label scores for model", model + "\n\n\n")
        checkpoint_path = os.path.join("resources", "models", model)
        for file in os.listdir(checkpoint_path):
            if file.endswith(".index"):
                print("\n\n")
                checkpoint = file[:file.rindex(".index")]
                command = "python egeli.py -n {} -x True -r False -c {} -v {} | tee {}" \
                    .format(model,
                            checkpoint,
                            verbose,
                            os.path.join(test_scores_reports_dir, "test-report"+model+".txt"))
                run_cmd(command)


if __name__ == '__main__':
    main()