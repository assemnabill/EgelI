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

    boxes = detections['detection_boxes']
    classes = detections['detection_classes'] + label_id_offset
    scores = detections['detection_scores']

    return (image_np_with_detections, boxes, classes, scores, category_index)


def detect_and_display_from_img(image_path, verbose=True, model=None, max_boxes=5):
    if model is None:
        model = configs.detection_model
    (image_np_with_detections, boxes, classes, scores, category_index) = setup_data_for_detection(image_path, model)

    if verbose:
        print("Best score is", select_best_class_and_score(category_index, classes, scores))

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=max_boxes,
        min_score_thresh=configs.detection_threshold,
        agnostic_mode=False)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')
    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    plt.show(block=True)

    if configs.save_plots:
        plt.savefig(f'{image_path}')


def detect_and_display_test_images(test_images_path=None, max_boxes=5):
    if test_images_path is None:
        test_images_path = os.path.join(configs.paths["IMAGE_PATH"], "test")
    images = remove_non_images_files(os.listdir(test_images_path))


    if configs.random_detection:
        random.shuffle(images)

    for img in images:
        image_path = os.path.join(test_images_path, img)
        detect_and_display_from_img(image_path, max_boxes=max_boxes)


# Not in use right now.
# Will be used later for first splicing faces in image via RetinaNet, detecting splices faces,
# and then drawing bounding box and label on original image.
def detect_subfaces(path=None):
    if path is None:
        path = os.path.join(configs.paths["IMAGE_PATH"], "test")
    images = remove_non_images_files(os.listdir(path))

    if configs.random_detection:
        random.shuffle(images)

    for img_file in images:
        image_path = os.path.join(path, img_file)
        img = load_image(image_path)
        faces = detect_faces_retinanet(img_file, image_path)
        for (xmin, ymin, xmax, ymax, xtot, ytot) in faces:
            cropped_face = img[ymin:ymax, xmin:xmax]
            detect_and_display_from_img(cropped_face)


def load_detection_model_from_checkpoint(checkpoint):
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
