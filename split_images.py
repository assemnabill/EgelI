import math
import os
import re
import shutil
from optparse import OptionParser
from configs import remove_non_images_files
import cv2
from retinaface import RetinaFace

folder_name_placeholder = "[FOLDER_NAME]"
file_name_placeholder = "[FILE_NAME]"
path_placeholder = "[PATH]"
width_placeholder = "[WIDTH]"
height_placeholder = "[HEIGHT]"
label_placeholder = "[LABEL]"
xmin_placeholder = "[XMIN]"
ymin_placeholder = "[YMIN]"
xmax_placeholder = "[XMAX]"
ymax_placeholder = "[YMAX]"

collected_images_folder = os.path.join("", "resources", "images", "collected images")
faces_folder = os.path.join("", "resources", "images", "FACES")
test_folder = os.path.join("", "resources", "images", "test")
train_folder = os.path.join("", "resources", "images", "train")
default_xml_path = os.path.join("", "resources", "EXAMPLE.xml")


def read_xml_content():
    with open(default_xml_path, 'r') as f:
        content = f.read()
    return content


default_xml_content = read_xml_content()


def make_xml(folder_name, file_name, label, xml_path, image_path, width, height, xmin, ymin, xmax, ymax):
    xml_content = default_xml_content
    xml_content = xml_content.replace(folder_name_placeholder, folder_name)
    xml_content = xml_content.replace(path_placeholder, os.path.realpath(image_path))
    xml_content = xml_content.replace(file_name_placeholder, file_name)
    xml_content = xml_content.replace(width_placeholder, str(width))
    xml_content = xml_content.replace(height_placeholder, str(height))
    xml_content = xml_content.replace(label_placeholder, label)
    xml_content = xml_content.replace(xmin_placeholder, str(xmin))
    xml_content = xml_content.replace(ymin_placeholder, str(ymin))
    xml_content = xml_content.replace(xmax_placeholder, str(xmax))
    xml_content = xml_content.replace(ymax_placeholder, str(ymax))
    xml_content = xml_content.replace(ymax_placeholder, str(ymax))

    with open(xml_path, 'w') as f:
        f.write(xml_content)


def make_copy_in_relevant_folder(testing_reached, label, org_file, i):
    if testing_reached:
        file_name = os.path.join(train_folder, label + "-" + str(i) + ".jpg")
    else:
        file_name = os.path.join(test_folder, label + "-" + str(i) + ".jpg")


def load_image(img_path):
    abspath = os.path.abspath(img_path)
    return cv2.imread(abspath)


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


def detect_faces_opencv(img, images_to_locate=-1, do_show_image=False, min_faces_to_show=-1, scale_factor=1.3, min_neighbors=10):
    bounding_boxes = []
    # Draw rectangle around the faces
    faces = locate_faces_opencv(img, scale_factor=scale_factor, min_neighbors=min_neighbors)
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


def make_label(subdir):
    return subdir[subdir.rindex("\\") + 1:]


def is_image(file_name):
    return file_name.lower().endswith('jpg') | file_name.lower().endswith('jpeg') | file_name.lower().endswith('png')


def iterate_images_folder(root_dir, file_lambda=lambda subdir, file:(), exclude_folders=[]):
    for subdir, dirs, files in os.walk(root_dir):
        if subdir != root_dir:
            label = make_label(subdir)
            if label not in exclude_folders:
                print("Processing", subdir, "...")
                for file in files:
                    if is_image(file):
                        print("Processing", os.path.join(subdir, file), "...")
                        file_lambda(subdir, file)
            else:
                print("Skipping", subdir, "...")


def iterate_collected_images(file_lambda=lambda subdir, file:(), exclude_folders=[]):
    iterate_images_folder(collected_images_folder, file_lambda, exclude_folders=exclude_folders)


def make_org_file_path(subdir, file):
    return os.path.join(subdir, file)


def show_collected_faces(exclude_folders=[]):
    iterate_collected_images(lambda subdir, file: detect_faces_opencv(load_image(make_org_file_path(subdir, file)), do_show_image=True, min_faces_to_show=3), exclude_folders=exclude_folders)


def write_subimages_faces(img, org_file_name, label, do_show_image=False):
    label_faces_folder = os.path.abspath(faces_folder, label)
    os.makedirs(label_faces_folder, exist_ok=True)

    #crop images and save
    face_boxes = detect_faces_opencv(img)
    i = 0
    for (xmin, ymin, xmax, ymax, xtot, ytot) in face_boxes:
        cropped_image = img[ymin:ymax, xmin:xmax]
        if do_show_image:
            show_image(cropped_image)
        file_base = org_file_name[:str(org_file_name).rindex(".")]
        final_file_path = os.path.join(label_faces_folder, file_base + ("-"+str(i) if i > 0 else "") + ".jpg")
        cv2.imwrite(final_file_path, cropped_image)
        i=i+1


def create_face_files(exclude_folders=[], do_show_images=False):
    iterate_collected_images(lambda subdir, file: write_subimages_faces(load_image(make_org_file_path(subdir, file)), file, make_label(subdir), do_show_image=do_show_images),
                             exclude_folders=exclude_folders)


def split_and_annotate(training_percentage=0.8, max_number_images=-1, exclude_folders=[], generate_xml=True,
                       do_show_image=False):
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(train_folder, exist_ok=True)

    for subdir, dirs, files in os.walk(collected_images_folder):
        print("Processing", subdir, "...")
        if subdir != collected_images_folder:
            label = subdir[subdir.rindex("\\") + 1:]
            if label not in exclude_folders:
                num_images = min(max_number_images, len(files)) if max_number_images > 0 else len(files)
                testing_start_index = math.floor(num_images * training_percentage)
                i = 0
                while i < num_images:
                    org_file_name = files[i]
                    print("Processing image", str(i) + "/" + str(num_images),
                          "(" + os.path.join(subdir, org_file_name) + ")")
                    testing_reached = i < testing_start_index
                    org_file_path = os.path.join(subdir, org_file_name)
                    tar_image_file_name_base = label + "-" + str(i)
                    file_base = os.path.join(train_folder, label + "-" + str(i)) if testing_reached else os.path.join(
                        test_folder, tar_image_file_name_base)
                    shutil.copy(org_file_path, file_base + ".jpg")
                    if generate_xml:
                        img = load_image(org_file_path)
                        detected_face = detect_faces(img, images_to_locate=1, do_show_image=do_show_image)[0]
                        if detected_face:
                            (xmin, ymin, xmax, ymax, xtot, ytot) = detected_face
                            make_xml(label, tar_image_file_name_base + ".jpg", label,
                                     file_base + ".xml",
                                     file_base + ".jpg",
                                     xtot, ytot, xmin, ymin, xmax, ymax)

                    i = i + 1

def annotate_extracted_faces(root_dir=faces_folder, training_percentage=0.8, max_number_images=-1, exclude_folders=[], generate_xml=True,
                       do_show_image=False):
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(train_folder, exist_ok=True)

    for subdir, dirs, files in os.walk(root_dir):
        print("Processing", subdir, "...")
        if subdir != root_dir:
            label = subdir[subdir.rindex("\\") + 1:]
            if label not in exclude_folders:
                num_images = min(max_number_images, len(files)) if max_number_images > 0 else len(files)
                testing_start_index = math.floor(num_images * training_percentage)
                i = 0
                while i < num_images:
                    org_file_name = files[i]
                    print("Processing image", str(i) + "/" + str(num_images),
                          "(" + os.path.join(subdir, org_file_name) + ")")
                    testing_reached = i < testing_start_index
                    org_file_path = os.path.join(subdir, org_file_name)
                    tar_image_file_name_base = label + "-" + str(i)
                    file_base = os.path.join(train_folder, label + "-" + str(i)) if testing_reached else os.path.join(
                        test_folder, tar_image_file_name_base)
                    shutil.copy(org_file_path, file_base + ".jpg")
                    if generate_xml:
                        img = load_image(org_file_path)
                        if do_show_image:
                            show_image(img)
                        (ytot, xtot, depth) = img.shape
                        #(x, y, w, h) = face
                        make_xml(label, tar_image_file_name_base + ".jpg", label,
                                 file_base + ".xml",
                                 file_base + ".jpg",
                                 xtot, ytot, 0, 0, xtot, ytot)

                    i = i + 1


def usage():
    print('Usage: split_images.py [options]\n')
    print('\t -n, --model-name= \t set a custom model name\n'
          'Images of faces to be split and annotated should be placed in \'resources/images/collected images\' folder.\n')


def run_cmd(cmd):
    print(f'Running {cmd}')
    return os.system(cmd)


def main():
    parser = OptionParser()
    parser.add_option("-a", "--auto_annotate", action="store_true", default=False)
    parser.add_option("-p", "--training_percentage", action="store", default=0.8, type="float")
    parser.add_option("-m", "--max_images", action="store", default=-1, type="int")
    parser.add_option("-x", "--exclude_folders", action="store", default=[], type="string")
    parser.add_option("-s", "--show_images", action="store_true", default=False)
    parser.add_option("-f", "--face_files", action="store_true", default=False)

    options, args = parser.parse_args()

    excluded_folders = options.exclude_folders
    if excluded_folders:
        excluded_folders = re.split(",\\s*", excluded_folders)

    if options.face_files:
        if options.auto_annotate:
            annotate_extracted_faces()
        else:
            create_face_files(exclude_folders=excluded_folders, do_show_images=options.show_images)
        if options.show_images:
            show_collected_faces(exclude_folders=excluded_folders)
    else:
        split_and_annotate(generate_xml=options.auto_annotate,
                           max_number_images=options.max_images,
                           exclude_folders=excluded_folders,
                           training_percentage=options.training_percentage,
                           manual_confirmation=options.show_images,
                           max_number_faces=-1)




if __name__ == '__main__':
    main()
