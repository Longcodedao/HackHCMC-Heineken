# Import yolo library
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torchvision import datasets, transforms, models
from transformers import CLIPProcessor, CLIPModel, Blip2Processor, Blip2Model
import json
from torch.utils.data import DataLoader, Dataset
import os
import argparse


parser = argparse.ArgumentParser(description="YOLO and CLIP detection script")
parser.add_argument('--image_dir', type=str, required=True, help='Path to the directory containing images')
args = parser.parse_args()


PATH_MODEL_YOLO = './weights/best.pt'


# Initialize YOLOv9
model_YOLO = YOLO(PATH_MODEL_YOLO)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize CLIP
processor_CLIP = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model_CLIP = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model_CLIP.to(device)


list_labels_Scene = ["The bar scene", "The pub scene", "The restaurant scene",
                    "The grocery store scene", "The supermarket scene",
                    "The party scene", "The celebration scene",
                    "The gathering scene"]

list_labels_Environment = ["Outside", "Inside"]


# Detect Emotion
def infer_image_emotion(image, processor, model):
    list_labels = ["A person is happy",
                    "A person is Angry",
                    "A person is Enjoyable",
                    "A person is Relaxed",
                    "A person is Neutral",
                    "Cannot idenitify the emotion"]
    image = Image.fromarray(image)

    inputs = processor(
        text = list_labels,
        images = image,
        return_tensors = "pt",
        padding=True
    )

    model.to(device)

    pixel_values = inputs["pixel_values"].to(device)
    labels = inputs["input_ids"].to(device)

    output = model(pixel_values = pixel_values, input_ids = labels)

    logits_per_image = output['logits_per_image']
    logits_per_text = output['logits_per_text']

    pred = torch.argmax(logits_per_image)
    final_result = list_labels[pred]

    return final_result



def infer_image_context(image, list_labels, processor,
                   model, device = "cuda", number_images = 20, display = True):


    image = Image.fromarray(image)

    inputs = processor(
        text = list_labels,
        images = image,
        return_tensors = "pt",
        padding = True
    )

    model.to(device)

    pixel_values = inputs["pixel_values"].to(device)
    labels = inputs["input_ids"].to(device)

    output = model(pixel_values = pixel_values, input_ids = labels)

    logits_per_image = output['logits_per_image']
    logits_per_text = output['logits_per_text']

    pred = torch.argmax(logits_per_image)
    final_result = list_labels[pred]

    return final_result


def crop_person(image, results):
    # crop beers
    persons = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get the bounding box coordinates
            if int(box.cls[0]) == 35:
                x1, y1, x2, y2 = box.xyxy[0]

                # Crop the image
                cropped_image = image[int(y1):int(y2), int(x1):int(x2), :]
                persons.append(cropped_image)
    return persons

def crop_beer(image, results):
    # crop beers
    beers = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get the bounding box coordinates
            if int(box.cls[0])%5 == 0 and int(box.cls[0]) != 35:
                x1, y1, x2, y2 = box.xyxy[0]

                # Crop the image
                cropped_image = image[int(y1):int(y2), int(x1):int(x2), :]
                cropped_image = cv2.resize(cropped_image, (224, 224))
                beers.append(cropped_image)



def draw_bouding_box(img_array, results):
    # draw bouding box
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get the bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get the class label and confidence score
            class_id = int(box.cls[0])
            class_name = model_YOLO.names[class_id]
            confidence = float(box.conf[0])

            # Draw the bounding box and label on the frame
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_array, f'{class_name} ({confidence:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)



def frequency_class(results):
    # count frequency class
    frequency = {}
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get the class label and confidence score
            class_id = int(box.cls[0])
            class_name = model_YOLO.names[class_id]
            if class_name in frequency:
                frequency[class_name] += 1
            else:
                frequency[class_name] = 1
    return frequency


def extract_json(data):
    with open("output.json", "w") as file:
        json.dump(data, file, indent=len(data))


def detect_all(image_path, idx):
    # Open the image file with Pillow
    # img = Image.open(PATH_IMAGE)  # Replace with your image path
    img = cv2.imread(image_path)
    # Convert the image to RGB format (Matplotlib expects RGB images
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the image data to a NumPy array (because Matplotlib needs this as input)
    img_array = np.array(img)

    # Use Matplotlib to display the image
    plt.imshow(img_array)
    plt.show()

    objects = []

    results = model_YOLO.predict(image_path)

    persons = crop_person(img_array, results)
    print(len(persons))

    for person in persons:
        result = infer_image_emotion(person, processor_CLIP, model_CLIP)
        fig, ax = plt.subplots()
        ax.imshow(person)
        ax.set_title(result)
        plt.show()

    # img = img_array.copy()

    draw_bouding_box(img, results)
    plt.imshow(img)

    freq = frequency_class(results)

    context = infer_image_context(img_array, list_labels_Scene, processor_CLIP, model_CLIP)
    # print(context)

    env = infer_image_context(img_array, list_labels_Environment, processor_CLIP, model_CLIP)
    # print(env)

    class_names = model_YOLO.names
    # print(class_names)

    object_image = {'image_id': image_path, "objects": freq, "context": context,
                    "environment": env }

    objects.append(object_image)
    return objects


def dectection(path_dir):
    objects = []
    for idx, filename in enumerate(os.listdir(path_dir)):
        obj = detect_all(os.path.join(path_dir, filename), idx)[0]
        objects.append(obj)
    return objects


if __name__ == '__main__':
    obj = dectection(args.image_dir)
    extract_json(obj)