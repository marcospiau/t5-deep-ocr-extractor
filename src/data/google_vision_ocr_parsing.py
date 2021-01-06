from typing import Dict


def extract_full_text_annotation(x: Dict[str, str]) -> str:
    """Extract full text annotation from a single OCR output from Google Cloud
    Vision API

    Args:
        x (Dict[str, str]): whole OCR output.

    Returns:
        str: string corresponding to the full text portion of OCR output.
    """
    raw_full_text = x["fullTextAnnotation"]["text"]
    return raw_full_text


def extract_text_and_vertices(x: Dict[str, str]):
    """Extracts all annotations and bounding box vertices from a single OCR
        output from Google Cloud Vision API.

    The first element is the full OCR. It's equivalent to the output of
    `extract_full_text_annotation` for the same OCR output.

    Args:
        x (Dict[str, str]): whole OCR output.

    Returns:
        list where each item is a tuple where the first element is the text and
        the second are the 4 vertices of the corresponding bounding box.
    """
    blocks = []
    for annotation in x["textAnnotations"]:
        text = annotation['description']
        vertices = [
            tuple(x.values()) for x in annotation['boundingPoly']['vertices']
        ]
        blocks.append((text, vertices))
    return blocks
