import random
import cv2
import numpy as np
import configs
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


@tf.function(experimental_relax_shapes=True)
def detect_fn(image):
    image, shapes = configs.detection_model.preprocess(image)
    prediction_dict = configs.detection_model.predict(image, shapes)
    detections = configs.detection_model.postprocess(prediction_dict, shapes)
    return detections


def detect_from_img(image_path):
    category_index = label_map_util.create_category_index_from_labelmap(configs.files['LABELMAP'])
    img = cv2.imread(image_path)
    image_np = np.array(img)
    print(f'detecting from {image_path}')
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=configs.detection_threshold,
        agnostic_mode=False)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')
    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    plt.show(block=True)
    if configs.save_plots:
        plt.savefig(f'{image_path}')


def test():
    images = os.listdir(os.path.join(configs.paths["IMAGE_PATH"], "test"))
    for img in images:
        is_img = img.lower().endswith('jpg') | img.lower().endswith('jpeg') | img.lower().endswith('png')
        if (not is_img) | (img.lower().endswith('.xml')):
            images.remove(img)

    if configs.random_detection:
        random.shuffle(images)

    for img in images:
        image_path = os.path.join(test_images_path, img)
        detect_from_img(image_path)


def detection_model_from_checkpoint(checkpoint):
    from object_detection.builders import model_builder
    from object_detection.utils import config_util
    print('Loading pipeline config...')
    pipeline_config = config_util.get_configs_from_pipeline_file(configs.files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=pipeline_config['model'], is_training=False)
    print('Restoring checkpoint...')
    check_point = tf.compat.v2.train.Checkpoint(model=detection_model)
    check_point.restore(os.path.join(configs.paths['CHECKPOINT_PATH'], checkpoint)).expect_partial()
    return detection_model


def run():
    if configs.checkpoint:
        checkpoint = configs.checkpoint
    else:
        index = os.listdir(configs.paths["CHECKPOINT_PATH"])
        index.reverse()
        checkpoint = index.pop().replace(".index", "")

    configs.detection_model = detection_model_from_checkpoint(checkpoint)
    print("Detection threshold set to", configs.detection_threshold)
    print("Random detection sequence set to", configs.random_detection)
    test()
