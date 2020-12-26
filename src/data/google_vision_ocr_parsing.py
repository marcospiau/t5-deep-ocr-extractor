from typing import Dict


def extract_full_text_annotation(x: Dict[str, str]) -> str:
    """Extract full text annotation from a single OCR output.

    Args:
        x (Dict[str, str]): whole OCR output.

    Returns:
        str: string corresponding to the full text portion of OCR output.
    """
    raw_full_text = x["fullTextAnnotation"]["text"]
    return raw_full_text
