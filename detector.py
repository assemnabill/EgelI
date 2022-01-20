import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import cv2
import numpy as np
# from matplotlib import pyplot as plt
import configs


@tf.function
def detect_fn(image):
    image, shapes = configs.detection_model.preprocess(image)
    prediction_dict = configs.detection_model.predict(image, shapes)
    detections = configs.detection_model.postprocess(prediction_dict, shapes)
    return detections


# Images should be in the test folder
def detect_from_img(image_path, threshold=0.8):
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
        min_score_thresh=threshold,
        agnostic_mode=False)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')
    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    plt.show()
    if configs.save_plots:
        plt.savefig(f'{image_path}')


def test_all_images(base_dir=".\\resources\\images\\test", random_sequence=False, threshold=0.6):
    for subdir, dirs, files in os.walk(base_dir):
        if subdir == base_dir:
            for img in images:
                if img.endswith('xml'):
                    images.remove(img)
            i = 0
            while i < len(files):
                index = randrange(0, len(files)) if random_sequence else i
                file = files[index]
                try:
                    detect_from_img(os.path.join(subdir, file), threshold=0.6)
                    i = i + 1
                except:
                    pass


def run():
    print('Restore checkpoint...')
    ckpt = tf.compat.v2.train.Checkpoint(configs.detection_model)
    index = os.listdir(configs.paths["CHECKPOINT_PATH"])
    index.reverse()
    checkpoint = index.pop().replace(".index", "")
    ckpt.restore(os.path.join(configs.paths['CHECKPOINT_PATH'], checkpoint)).expect_partial()

    images = os.listdir(os.path.join(configs.paths["IMAGE_PATH"], "test"))

    for img in images:
        if img.endswith('xml'):
            images.remove(img)

    for img in images:
        image_path = os.path.join(configs.paths['IMAGE_PATH'], 'test', img)
        detect_from_img(image_path)
