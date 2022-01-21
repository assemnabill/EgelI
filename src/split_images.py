import math
import os
import re
import shutil
from optparse import OptionParser


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

base_folder = os.path.join("..", "resources", "images", "collected images")
test_folder = os.path.join("..", "resources", "images", "test")
train_folder = os.path.join("..", "resources", "images", "train")
default_xml_path = os.path.join("..", "resources", "EXAMPLE.xml")


def read_xml_content():
    with open(default_xml_path, 'r') as f:
        content = f.read()
    return content


default_xml_content = read_xml_content()

def make_xml(folder_name, file_name, xml_path, image_path, width, height, label, xmin, ymin, xmax, ymax):
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


def detect_face(img_path, show_image=False):
    import cv2

    # Load the cascade
    face_cascade = cv2.CascadeClassifier(os.path.join("..", "resources", "haarcascade_frontalface_default.xml'"))
    # Read the input image
    img = cv2.imread(img_path)
    (ytot, xtot, depth) = img.shape
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 10)
    # Draw rectangle around the faces
    detected_face = len(faces) >= 1
    if detected_face:
        (x, y, w, h) = faces[0]
        #for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # Display the output
    if show_image:
        cv2.imshow('img', img)
        cv2.waitKey()

    if (detected_face):
        return x, y, x+w, y+h, xtot, ytot
    else: return False


def split_images(training_percentage=0.8, max_number_images=-1, exclude_folders=[], generate_xml = True):
    #1. folder selections
    #1.1 select testing percentage
    #1.2 select image amount to be maximally split
    #2. iterate over folders
        #2.1 for each folder:
            #2.2 number_images = max(max_number_images, number_images_in_folder)
            #2.3 while i < number_images
                #2.4 if (i < number_images * testing_percentage) copy [with name [folder]-i.jpg] to ../test/
                #2.5 else (i < number_images * testing_percentage) copy [with name [folder]-i.jpg] to ../test/

    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(train_folder, exist_ok=True)

    for subdir, dirs, files in os.walk(base_folder):
        print("Processing", subdir, "...")
        if subdir != base_folder:
            label = subdir[subdir.rindex("\\") + 1:]
            if not label in exclude_folders:
                num_images = min(max_number_images, len(files)) if max_number_images > 0 else len(files)
                testing_start_index = math.floor(num_images * training_percentage)
                i = 0
                while i < num_images:
                    org_file_name = files[i]
                    print("Processing image",str(i)+"/"+str(num_images), "("+os.path.join(subdir, org_file_name)+")")
                    testing_reached = i < testing_start_index
                    org_file_path = os.path.join(subdir, org_file_name)
                    tar_image_file_name_base = label + "-" + str(i)
                    file_base = os.path.join(train_folder, label + "-" + str(i)) if testing_reached else os.path.join(test_folder, tar_image_file_name_base)
                    shutil.copy(org_file_path, file_base + ".jpg")
                    if generate_xml:
                        detected_face = detect_face(org_file_path)
                        if (detected_face):
                            (xmin, ymin, xmax, ymax, xtot, ytot) = detected_face
                            make_xml(label, tar_image_file_name_base+".jpg", file_base+".xml", file_base + ".jpg", xtot, ytot, label, xmin, ymin, xmax, ymax)

                    i=i+1


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

    options, args = parser.parse_args()

    excluded_folders = options.exclude_folders
    if excluded_folders:
        excluded_folders = re.split(",\\s*", excluded_folders)

    split_images(generate_xml=options.auto_annotate,
                 max_number_images=options.max_images,
                 exclude_folders=excluded_folders,
                 training_percentage=options.training_percentage)


if __name__ == '__main__':
    main()