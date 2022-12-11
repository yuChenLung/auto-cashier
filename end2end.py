import os
import sys
import random
import numpy as np
import skimage.io
#import matplotlib
import matplotlib.pyplot as plt
import colorsys
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
from skimage.measure import find_contours
from mrcnn.config import Config
import time

#import keras
from keras.models import load_model
#from keras import models
#from keras.preprocessing.image import ImageDataGenerator
#from keras.models import Model
from keras.preprocessing import image
#from PIL import Image

ROOT_DIR = os.path.abspath("E:/Documents/ML/Mask_RCNN-master")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
#import coco

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# %matplotlib inline only works in iPython notebooks
#Use the following
from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
LOG_DIR = os.path.join(ROOT_DIR, "end2end/seglogs")
SEG_DIR = os.path.join(ROOT_DIR, "end2end/segimages")
#SEG_DIR = os.path.abspath("C:/ML_data/notouchtest")
IMAGE_DIR = os.path.abspath("E:/Pictures/ML_data/foodTray123119")

#Set up shape model configuration
class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# Create seg model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=LOG_DIR, config=config)
model_path = os.path.join(ROOT_DIR, "mask_rcnn_shapes_0010.h5")
model.load_weights(model_path, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

cmodel = load_model('mobilenet192sct.h5')
total_class_count = []

def classify(croppedimg):
    skimage.io.imsave('inferencetemp.png', croppedimg)
    #img = Image.fromarray(img_path*255)
    #img = img.resize((150,150))
    img_path = 'E:/Documents/ML/Mask_RCNN-master/inferencetemp.png'
    img = image.load_img(img_path, target_size=(192, 192))
    #img = img.astype(np.float32)

    img_tensor = image.img_to_array(img)


    #img_tensor = img_path[:,:,3]

    img_tensor = np.expand_dims(img_tensor, axis=0)

    # Remember that the model was trained on inputs
    # that were preprocessed in the following way:
    img_tensor /= 255.


    img_class = cmodel.predict_classes(img_tensor)
    #prediction = img_class[0]
    classname = img_class[0]

    if classname == 0:
        class_label="Apple"
    if classname == 1:
        class_label="Bagel"
    if classname == 2:
        class_label="Banana"
    if classname == 3:
        class_label="Muffin"

    total_class_count.append(class_label)
    print("Class: ",class_label)

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_anti_mask(image, mask, color, alpha=1):
    """Apply the given mask to the image.

    mark pixel which has mask 0 with color c
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 0,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_segs(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("/n*** No instances to display *** /n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = True
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        masked_image = image.astype(np.uint32).copy()

        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            #ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
            '''
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")
'''
        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_anti_mask(masked_image, mask, [1,1,1]) # [1,1,1]: white background color





        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)

        #crop
        cropped = masked_image.astype(np.uint8).copy()
        l = r['rois'][i]
        cropped = cropped[l[0]:l[2],l[1]:l[3],:]

        #ax.imshow(masked_image.astype(np.uint8))
        plt.imsave(os.path.join(SEG_DIR, str(random.randint(0, 10000))+'.png'), cropped)

        #CLASSIFY
        classify(cropped)


#Load images
from skimage.io import imread_collection
startTime = time.time()

col = imread_collection(IMAGE_DIR + '/*.jpg')
for i, img in enumerate(col):
    img = col[i]
    # Run detection
    results = model.detect([img], verbose=1)

    # Visualize results
    r = results[0]
    display_segs(img, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])

print ("Total " + str(len(total_class_count)) + " Objects")

numApple = 0
numBagel = 0
numBanana = 0
numMuffin = 0

for item in total_class_count:
    if item == "Apple":
        numApple += 1
    elif item == "Bagel":
        numBagel += 1
    elif item == "Banana":
        numBanana += 1
    elif item == "Muffin":
        numMuffin += 1

print ("Apples: " + str(numApple) + ", Price: $1.00")
print ("Bagels: " + str(numBagel) + ", Price: $0.80")
print ("Bananas: " + str(numBanana) + ", Price: $0.50")
print ("Muffins: " + str(numMuffin) + ", Price: $1.00")

totalCost = float(numApple) + float(numBagel) * 0.80 + float(numBanana) * 0.50 + float(numMuffin)

print ("Total Cost: $" + str(totalCost) + "0")

print ("---%s seconds processing time ---" % (time.time()-startTime))




#START CLASSIFICATION
#UTILIZE SEG_DIR
