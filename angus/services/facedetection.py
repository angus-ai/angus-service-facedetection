#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import logging
import os

import angus.service

LOGGER = logging.getLogger(__name__)


def compute(resource, data):
    img = str(data['image'].path)
    img = cv2.imread(img)

    home = os.path.dirname(os.path.realpath(__file__))
    classifiers = home + "/resources/classifiers/"
    classifier = classifiers + 'haarcascade_frontalface_alt.xml'
    face_cascade = cv2.CascadeClassifier(classifier)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height = img.shape[0]
    width = img.shape[1]

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    result = {
        "input_size": [width, height],
        "nb_faces": len(faces),
        "faces": [{
            "roi": [int(x), int(y), int(w), int(h)],
            "roi_fonfidence": 0,
        } for (x, y, w, h) in faces]
    }
    resource.update(result)


def main():
    port = os.environ.get('PORT', 9999)
    logging.basicConfig(level=logging.DEBUG)

    service = angus.service.Service(
        'face_detection', 1,
        port,
        compute,
        resource_storage=dict(), threads=4
    )
    service.start()

if __name__ == '__main__':
    main()
