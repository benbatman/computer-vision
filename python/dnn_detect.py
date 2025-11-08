import sys
import logging
import os
from typing import List, Tuple, Dict, Any

import cv2 as cv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SHOW_LOW_CONFIDENCE = True
SCORE_THRESHOLD = 0.4
# Adjust based on expected object sizes in image
BOX_TO_IMAGE_RATIO_THRESHOLD = 0.45
LOW_CONFIDENCE_THRESHOLD = 0.2
LOW_CONFIDENCE_COLOR = (0, 0, 255)
# class:color mapping
DEFAULT_COLOR = (0, 255, 0)
# we'll just do one class rn
class_colors = {1: (255, 0, 0)}


def load_labels(path_to_pbtxt):
    """Load label map from .pbtxt file"""
    labels = {}
    with open(path_to_pbtxt, "r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if "id:" in lines[i]:
                class_id = int(lines[i].strip().split("id:")[1])
                class_name = (
                    lines[i + 1].strip().split("display_name:")[1].strip().strip('"')
                )
                labels[class_id] = class_name
    return labels


def load_tf_model(model_path: str, config_path: str):
    """Load TensorFlow DNN model"""
    cvNet = cv.dnn.readNetFromTensorflow(
        model_path,
        config_path,
    )
    return cvNet


def process_image(image_path: str) -> Tuple[cv.Mat, List[Dict[str, Any]]]:
    valid_detections = []
    img = cv.imread(image_path)
    rows = img.shape[0]
    cols = img.shape[1]
    blob_image = cv.dnn.blobFromImage(
        img,
        scalefactor=1.0 / 127.5,  # Scale to [-1, 1]
        size=(320, 320),
        mean=(127.5, 127.5, 127.5),  # Mean subtraction
        swapRB=True,
        crop=False,
    )
    resized_h, resized_w = blob_image.shape[2], blob_image.shape[3]
    print(f"Original image size: {cols}x{rows}")
    print(f"Resized blob size: {resized_w}x{resized_h}")
    cvNet.setInput(blob_image)
    cvOut = cvNet.forward()

    for detection in cvOut[0, 0, :, :]:
        class_id = int(detection[1])
        score = float(detection[2])
        class_name = str(labels.get(class_id, "Unknown"))
        print(f"Detected class ID: {class_id} | Score: {score} | Name: {class_name}")

        # Coords are normalized to [0, 1], multiply to get pixels
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows

        # Box dims
        box_width = right - left
        box_height = bottom - top
        box_area = box_width * box_height
        image_area = rows * cols

        # Filter out boxes that are too large (likely false positives)
        if box_area > BOX_TO_IMAGE_RATIO_THRESHOLD * image_area:
            continue

        valid_detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "score": score,
                "box": (left, top, right, bottom),
            }
        )
    logger.info(f"Number of valid detections: {len(valid_detections)}")
    return img, valid_detections


def draw_detection(
    img: cv.Mat, detection: Dict[str, Any], low_confidence_ids: List[int]
) -> None:
    """
    Draw detection box and label on image

    Args:
        img (cv.Mat): Image to draw on
        detection (Dict[str, Any]): Detection info
        low_confidence_ids (List[int]): List of class IDs to SHOW that are considered low confidence
    """
    class_id = int(detection["class_id"])
    class_name = detection["class_name"]
    score = detection["score"]
    left, top, right, bottom = detection["box"]

    # some models use 0 for background
    if score > SCORE_THRESHOLD and class_id != 0:
        cv.rectangle(
            img,
            (int(left), int(top)),
            (int(right), int(bottom)),
            class_colors.get(class_id, DEFAULT_COLOR),
            thickness=2,
        )
        cv.putText(
            img,
            f"{class_name}: {int(score * 100)}%",
            (int(left), int(top) - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            class_colors.get(class_id, DEFAULT_COLOR),
            1,
            cv.LINE_AA,  # smooth line
        )
    if (
        score > LOW_CONFIDENCE_THRESHOLD
        and score <= SCORE_THRESHOLD
        and SHOW_LOW_CONFIDENCE
        and class_id in low_confidence_ids
    ):
        cv.rectangle(
            img,
            (int(left), int(top)),
            (int(right), int(bottom)),
            LOW_CONFIDENCE_COLOR,
            thickness=2,
        )
        cv.putText(
            img,
            f"{class_name}: {int(score * 100)}%",
            (int(left), int(top) - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            LOW_CONFIDENCE_COLOR,
            1,
            cv.LINE_AA,  # smooth line
        )


def nms(
    img: cv.Mat,
    valid_detections: List[Dict[str, Any]],
    low_confidence_ids: List[int] = [],
) -> None:
    """
    Apply Non-Maximum Suppression (NMS) to filter overlapping boxes
    """
    if not valid_detections:
        return []

    boxes = [
        [d["box"][0], d["box"][1], d["box"][2], d["box"][3]] for d in valid_detections
    ]
    scores = [d["score"] for d in valid_detections]

    # NMS, remove overlapping boxes
    indices = cv.dnn.NMSBoxes(boxes, scores, SCORE_THRESHOLD, 0.4)

    # Draw!
    if len(indices) > 0:
        indices = (
            indices.flatten()
            if hasattr(indices, "flatten")
            else [i[0] for i in indices]
        )
        for i in indices:
            draw_detection(img, valid_detections[i], low_confidence_ids)


def save_and_show_image(img, output_path="output.jpg"):
    cv.imwrite(output_path, img)
    logger.info(f"Output image saved to {output_path}")
    cv.imshow("Detections", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.info("Usage: python dnn_detect.py <image_path>")
        logger.info("Example: python dnn_detect.py input.jpg")
        sys.exit(1)

    low_confidence_ids = [1]  # Show low confidence for person class only

    labels = load_labels("label_map.pbtxt")
    logger.info(f"Loaded {len(labels)} labels.")
    cvNet = load_tf_model(
        "models/ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb",
        "models/ssd_mobilenet_v3_large_coco_2020_01_14/ssd_mobilenet_v3.pbtxt",
    )
    output_img, valid_detections = process_image(sys.argv[1])
    nms(output_img, valid_detections, low_confidence_ids)
    output_path = os.path.splitext(sys.argv[1])[0] + "_output.jpg"
    save_and_show_image(output_img, output_path)
    logger.info("Processing complete.")
