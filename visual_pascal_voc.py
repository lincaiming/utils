import os
import numpy as np
import xml.etree.ElementTree as ET
import cv2
# from pascal_voc_writer import Writer
import argparse
import shutil

from jinja2 import Environment, PackageLoader, FileSystemLoader


class Writer:
    def __init__(self, path, width, height, depth=3, database='Unknown', segmented=0):
        environment = Environment(loader=FileSystemLoader('./templates', followlinks=True))
        self.annotation_template = environment.get_template('annotation.xml')

        abspath = os.path.abspath(path)

        self.template_parameters = {
            'path': abspath,
            'filename': os.path.basename(abspath),
            'folder': os.path.basename(os.path.dirname(abspath)),
            'width': width,
            'height': height,
            'depth': depth,
            'database': database,
            'segmented': segmented,
            'objects': []
        }

    def addObject(self, name, xmin, ymin, xmax, ymax, pose='Unspecified', truncated=0, difficult=0):
        self.template_parameters['objects'].append({
            'name': name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'pose': pose,
            'truncated': truncated,
            'difficult': difficult,
        })

    def save(self, annotation_path):
        with open(annotation_path, 'w') as file:
            content = self.annotation_template.render(**self.template_parameters)
            file.write(content)


def parse_args():
    """Parse arguments of command line"""
    parser = argparse.ArgumentParser(
        description='Convert xxx XML annotations to PASCAL VOC format'
    )

    parser.add_argument(
        '--data_dir', metavar='DIRECTORY', required=True,
        help='directory which contains original images'
    )

    parser.add_argument(
        '--output_dir', metavar='DIRECTORY', required=True,
        help='directory for output annotations in PASCAL VOC format'
    )

    return parser.parse_args()


def process_voc_xml(data_dir, output_dir):

    xml_dir = os.path.join(data_dir, "Annotations")

    for item in os.listdir(xml_dir):
        anno_file = os.path.join(data_dir, "Annotations", item)
        jpeg_file = os.path.join(data_dir, "JPEGImages", os.path.splitext(item)[0] + ".jpg")
        if not os.path.exists(jpeg_file):
            print(jpeg_file)
        tree = ET.parse(anno_file)

        image_name = tree.find("filename").text
        width = int(tree.findall("./size/width")[0].text)
        height = int(tree.findall("./size/height")[0].text)
        depth = int(tree.findall("./size/depth")[0].text)
        image_path = os.path.join(data_dir, "JPEGImages", image_name)

        writer = Writer(image_path, width, height, depth=depth)
        img = cv2.imread(jpeg_file)
        cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
        cnt = 0          
        for obj in tree.findall('object'):
            cls = obj.find("name").text
            os.makedirs(os.path.join(output_dir, cls), exist_ok=True)
            bbox = obj.find("bndbox")
            xmin = min(max(int(float(bbox.find("xmin").text)), 0), width - 1)
            ymin = min(max(int(float(bbox.find("ymin").text)), 0), height - 1)
            xmax = min(max(int(float(bbox.find("xmax").text)), 0), width - 1)
            ymax = min(max(int(float(bbox.find("ymax").text)), 0), height - 1)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), thickness=2)
            cv2.putText(img, cls, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0),
                       thickness=2)
            # roi_img = img[xmin:xmax, ymin:ymax, :]
            # cv2.imwrite(os.path.join(output_dir, cls, cls+"_"+str(cnt) + ".jpg"), roi_img)
            # writer.addObject(cls, xmin, ymin, xmax, ymax)
            cnt = cnt + 1

        cv2.imshow(image_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        anno_name = os.path.basename(os.path.splitext(image_name)[0] + '.xml')
        anno_dir = os.path.dirname(os.path.join(output_dir, "Annotations", image_name))
        # os.makedirs(anno_dir, exist_ok=True)
        # jpeg_dir = os.path.dirname(os.path.join(output_dir, "JPEGImages", image_name))
        # os.makedirs(jpeg_dir, exist_ok=True)
        # shutil.copyfile(jpeg_file, os.path.join(output_dir, "JPEGImages", image_name))
        # writer.save(os.path.join(anno_dir, anno_name))


def main():
    args = parse_args()
    process_voc_xml(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()