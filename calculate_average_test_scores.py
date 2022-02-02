import os


def run_cmd(cmd):
    print(f'Running {cmd}')
    return os.system(cmd)


def main():
    #models = ["ssd_mobilenet_v1_640_imp_aa2_8_7k-2", "ssd_mobilenet_v1_640_imp_aa2_8_7k", "my_custom_model-ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"]
    models = ["ssd_mobilenet_v2_320_aa3_10k"]
    verbose = False
    for model in models:
        print("\n\n\nCalculating average test label scores for model", model + "\n\n\n")
        checkpoint_path = os.path.join("resources", "models", model)
        for file in os.listdir(checkpoint_path):
            if file.endswith(".index"):
                print("\n\n")
                checkpoint = file[:file.rindex(".index")]
                command = "python egeli.py -n {} -x True -r False -c {} -v {}" \
                    .format(model, checkpoint, verbose)
                run_cmd(command)


if __name__ == '__main__':
    main()