import random
import statistics
import cv2
import numpy as np
import six
from retinaface import RetinaFace
import configs
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from configurator import remove_non_images_files, load_image


@tf.function(experimental_relax_shapes=True)
def detect_fn(image):
    image, shapes = configs.detection_model.preprocess(image)
    prediction_dict = configs.detection_model.predict(image, shapes)
    detections = configs.detection_model.postprocess(prediction_dict, shapes)
    return detections


def collect_label_from_filepath(image_path):
    return image_path[image_path.rindex(os.path.sep)+1:image_path.rindex("-")]


def calculate_average_score_for_test_labels(verbose=True):

    init(verbose=verbose)
    print("Creating average test scores for labels...")
    test_images_path = os.path.join(configs.paths["IMAGE_PATH"], "test")
    images = remove_non_images_files(os.listdir(test_images_path))

    average_scores_dict = {}
    for img in images:
        image_path = os.path.join(test_images_path, img)
        class_label = collect_label_from_filepath(image_path)
        cls, score = detect_best_score_for_label(image_path, class_label, verbose=verbose)
        if verbose:
            print("Best score for class", cls, "was", score)
        if cls in average_scores_dict:
            label_count, score_sum = average_scores_dict[cls]
            average_scores_dict[cls] = (label_count+1, score_sum + score)
        else:
            average_scores_dict[cls] = (1, score)

    correct_dict = {}
    for (key, val) in average_scores_dict.items():
        label_count, score_sum = val
        correct_dict[key] = score_sum/label_count if score_sum > 0 else 0.0

    print("Finished evaluating label scores for model", configs.custom_model_name, "and checkpoint", configs.checkpoint+"!")
    print(correct_dict)
    median = statistics.median(correct_dict.values())
    print("Median label score is", median)
    return median, average_scores_dict


def detect_classes_from_img(image_path, scores_to_output=-1, verbose=True):
    (image_np_with_detections, boxes, classes, scores, category_index) = setup_data_for_detection(image_path, verbose=verbose)

    scores_dict = {}
    scores_to_output = min(scores_to_output, boxes.shape[0]) if scores_to_output != -1 else boxes.shape[0]

    for i in range(scores_to_output):
        if classes[i] in six.viewkeys(category_index):
            class_name = category_index[classes[i]]['name']
            scores_dict[class_name] = scores[i]

    return scores_dict


def detect_best_score_for_label(image_path, label, verbose=True):
    (image_np_with_detections, boxes, classes, scores, category_index) = setup_data_for_detection(image_path, verbose=verbose)

    for i in range(boxes.shape[0]):
        if classes[i] in six.viewkeys(category_index):
            class_name = category_index[classes[i]]['name']
            if class_name == label:
                return label, scores[i]

    return label, 0.0


def select_best_class_and_score(category_index, classes, scores):
    class_name = category_index[classes[0]]['name']
    score = scores[0]
    return class_name, score


def detect_best_class_and_score_from_img(image_path, verbose=True):
    score_dict = detect_classes_from_img(image_path, scores_to_output=1, verbose=verbose)
    key = list(score_dict.keys())[0]
    return key, score_dict[key]


def setup_data_for_detection(image_path, verbose=True):
    category_index = label_map_util.create_category_index_from_labelmap(configs.files['LABELMAP'])
    img = cv2.imread(image_path)
    image_np = np.array(img)
    if verbose:
        print(f'Detecting from {image_path}')
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


def detect_and_display_from_img(image_path, verbose=True, max_boxes=5):
    (image_np_with_detections, boxes, classes, scores, category_index) = setup_data_for_detection(image_path)

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


def detect(max_boxes=5, verbose=False):
    init(verbose=verbose)
    print("Detection threshold set to", configs.detection_threshold)
    print("Random detection sequence set to", configs.random_detection)
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


def locate_faces_opencv(img, scale_factor=1.3, min_neighbors=10):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(os.path.join("", "resources", "haarcascade_frontalface_default.xml"))
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
    return faces


def locate_faces_retina(img, img_path):
    (ytot, xtot, depth) = img.shape
    resp = RetinaFace.detect_faces(img_path, align=True)
    faces = []
    for key, value in resp.items():
        fa = value["facial_area"]
        faces.add(fa[0], fa[1], fa[2]-fa[0], fa[3]-fa[1], xtot, ytot)

    return faces


def detect_faces_retinanet(img, img_path, images_to_locate=-1, do_show_image=False, min_faces_to_show=-1, scale_factor=1.3, min_neighbors=10):
    bounding_boxes = []
    # Draw rectangle around the faces
    faces = locate_faces_retina(img, img_path)
    i = 0
    while i < len(faces) and (i < images_to_locate or images_to_locate == -1):
        face = faces[i]
        if do_show_image and (min_faces_to_show == -1 or len(faces) >= min_faces_to_show):
            paint_face_on_image(face, img)
        bounding_boxes.append(get_bounding_box(face, img))
        i = i + 1
    if do_show_image and (min_faces_to_show == -1 or len(faces) >= min_faces_to_show):
        show_image(img)
    return bounding_boxes


def paint_face_on_image(face, img):
    (x, y, w, h) = face
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)


def get_bounding_box(face, img):
    (ytot, xtot, depth) = img.shape
    (x, y, w, h) = face
    return x, y, x + w, y + h, xtot, ytot


def show_image(img, wait=True, win_name='img', wait_time=None):
    cv2.namedWindow(win_name, )
    cv2.imshow(win_name, img)
    if wait:
        return chr(cv2.waitKey(wait_time))
    return None


def init(verbose=False):
    if verbose:
        print("Loading checkpoint " + configs.checkpoint + "...")
    if configs.checkpoint:
        checkpoint = configs.checkpoint
    else:
        index = os.listdir(configs.paths["CHECKPOINT_PATH"])
        index.reverse()
        checkpoint = index.pop().replace(".index", "")

    configs.detection_model = load_detection_model_from_checkpoint(checkpoint)
