from google.cloud import vision
from google.protobuf.json_format import MessageToJson
import json


def request_ocr(img_contents):
    client = vision.ImageAnnotatorClient()
    image = vision.types.Image(content=img_contents)
    response = client.document_text_detection(image=image)
    return response


def get_json_ocr_from_image(image_path):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    text_annotations = request_ocr(image_bytes)
    text_annotations = MessageToJson(text_annotations)
    text_annotations = json.loads(text_annotations)
    return text_annotations
