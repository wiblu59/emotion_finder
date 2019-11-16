#!/usr/bin/env python3

import io
import cv2
import tempfile
import argparse

from PIL import Image, ImageDraw
from google.cloud import vision
from google.cloud.vision import types


def find_faces(image_to_analyse, max_results):
    """ Find faces on a given picture file
    :param image_to_analyse: an opened file containing a picture, ideally with faces
    :param max_results: the maximum number of faces to find
    :return: an array of faces found
    """
    client = vision.ImageAnnotatorClient()
    content = image_to_analyse.read()
    image = types.Image(content=content)
    return client.face_detection(image=image, max_results=max_results).face_annotations


def highlight_faces(image_path, faces, output_filename):
    """Draws a polygon around the faces, then saves to output_filename.
    :param image_path: a file containing the image with the faces.
    :param faces: a list of faces found in the file. This should be in the format
          returned by the Vision API.
    :param output_filename: the name of the image file to be created, where the
          faces have polygons drawn around them.
    :return: the name of file created with the polygon
    """
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    for index, face in enumerate(faces):
        box = [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
        draw.line(box + [box[0]], width=5, fill='#00ff00')
        draw.text((face.bounding_poly.vertices[0].x,
                   face.bounding_poly.vertices[0].y - 30),
                  str(format(face.detection_confidence, '.3f') + '% | ' + str(index)),
                  fill='#FF0000')
    img.save(output_filename)


def emotion_finder(image_path):
    """Given the path to a picture file, print if someone found on the picture is angry | happy | surprised | sorrowed
    :param image_path:Path to a picture file to analyze
    :return:Print the emotions found
    """
    likelihood_name = ('Incoonu', 'Pas du tout', 'Un petit peu', 'Probablement', 'Tout à fait', 'Carrément !')
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, 'rb') as image:
        content = image.read()
    final_image = vision.types.Image(content=content)
    face_property = client.face_detection(image=final_image)
    for i, face in enumerate(face_property.face_annotations):
        print('Visage {}'.format(i))
        print('Colère: {}'.format(likelihood_name[face.anger_likelihood]))
        print('Joie: {}'.format(likelihood_name[face.joy_likelihood]))
        print('Surprise: {}'.format(likelihood_name[face.surprise_likelihood]))
        print('Tristesse: {}'.format(likelihood_name[face.sorrow_likelihood]))


def video_capture(store_path):
    """Show a live view from webcam and capture a picture when pressing 'P'
    :param store_path: a path to files captured
    :return: a path to the captured picture
    """
    print("Video capture is starting...")
    cam = cv2.VideoCapture(0)
    print("Video capture in process in a new window")
    frame = -1
    while True:
        ret_val, img_data = cam.read()
        img_data = cv2.flip(img_data, 1)
        cv2.imshow('Ma webcam', img_data)
        frame += 1
        key = 0
        key = cv2.waitKey(1)
        if (key == ord('p')) or (key == ord('P')):
            cv2.imwrite('{0}/capture{1}.jpg'.format(store_path, frame), img_data)
            break
        elif key == 27:
            break
    path = '{0}/capture{1}.jpg'.format(store_path, frame)
    cv2.destroyAllWindows()
    print("Video capture ended")
    return path


def main():
    parser = argparse.ArgumentParser(description=">> Detect some emotions on human faces")
    parser.add_argument("-p", "--path", help="give a path to a file to analyze", type=str)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp_directory:
        if args.path:
            img_path = args.path
        else:
            img_path = video_capture(tmp_directory)
        with open(img_path, 'rb') as image_to_analyse:
            faces = find_faces(image_to_analyse, max_results=12)
        print('{} visage{} trouvé{}'.format(len(faces), '' if len(faces) == 1 else 's', '' if len(faces) == 1 else 's'))
        highlight_faces(img_path, faces, output_filename='found-faces.jpg')
        emotion_finder(img_path)


if __name__ == '__main__':
    main()
