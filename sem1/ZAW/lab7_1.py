#from google.colab import drive
import numpy as np
import cv2
#from google.colab.patches import cv2_imshow
import imutils
import os
from os.path import join

#drive.mount('/content/gdrive')
DATASET_DIR = r"C:\Users\IslandBoyXD\Desktop\magisterskie\rok1\ZAW\lab7\sequences"

SIGMA = 17
SEARCH_REGION_SCALE = 2
LR = 0.125
NUM_PRETRAIN = 128
VISUALIZE = True

def load_gt(gt_file):

    with open(gt_file, 'r') as file:
        lines = file.readlines()

    lines = [line.split(',') for line in lines]
    lines = [[int(float(coord)) for coord in line] for line in lines]
    # returns in x1y1wh format
    return lines


def show_sequence(sequence_dir):

    imgdir = join(sequence_dir, 'color')
    imgnames = os.listdir(imgdir)                  
    imgnames.sort()
    gt_boxes = load_gt(join(sequence_dir, 'groundtruth.txt'))

    for imgname, gt in zip(imgnames, gt_boxes):
        img = cv2.imread(join(imgdir, imgname))
        position = [int(x) for x in gt]
        cv2.rectangle(img, (position[0], position[1]), (position[0]+position[2], position[1]+position[3]), (255, 0, 0), 2)
        cv2.imshow('demo', img)
        if cv2.waitKey(0) == ord('q'):
            break



def crop_search_window(bbox, frame):

    xmin, ymin, width, height = bbox
    xmax = xmin + width
    ymax = ymin + height

    #----TODO (1)
    x_offset = (width * SEARCH_REGION_SCALE - width) / 2
    y_offset = (height * SEARCH_REGION_SCALE - height) / 2
    xmin = xmin - x_offset
    xmax = xmax + x_offset
    ymin = ymin - y_offset
    ymax = ymax + y_offset
    #----TODO (1)

    #----TODO (2)
    x_pad = max(0, int(-min(0, xmin)))
    y_pad = max(0, int(-min(0, ymin)))
    xmax_pad = max(0, int(max(xmax, frame.shape[1]) - frame.shape[1]))
    ymax_pad = max(0, int(max(ymax, frame.shape[0]) - frame.shape[0]))

    frame = cv2.copyMakeBorder(
        frame,
        y_pad, ymax_pad,
        x_pad, xmax_pad,
        cv2.BORDER_REFLECT
    )

    # Korekta współrzędnych po dodaniu paddingu
    xmin += x_pad
    xmax += x_pad
    ymin += y_pad
    ymax += y_pad
    #----TODO (2)

    window = frame[int(ymin) : int(ymax), int(xmin) : int(xmax), :]
    # cv2.imshow('search window', window.astype(np.uint8))
    window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)

    return window


def get_gauss_response(gt_box):

    width = gt_box[2] * SEARCH_REGION_SCALE
    height = gt_box[3] * SEARCH_REGION_SCALE
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))

    center_x = width // 2
    center_y = height // 2
    dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * SIGMA)
    response = np.exp(-dist)

    return response

def pre_process(img):

    height, width = img.shape
    img = img.astype(np.float32)

    #---- TODO (3)
    img = np.log(img +1)

    img = (img - np.mean(img)) / np.std(img)

    #---- TODO(3)

    #2d Hanning window
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)
    window = mask_col * mask_row
    img = img * window

    return img

def random_warp(img):

    #---TODO (4)

    angle = np.random.uniform(-15,15)

    img_rot = imutils.rotate_bound(img, angle)

    img_resized = cv2.resize(img_rot, (img.shape[1], img.shape[0]))
    return img_resized

def initialize(init_frame, init_gt):

    g = get_gauss_response(init_gt)
    G = np.fft.fft2(g)
    Ai, Bi = pre_training(init_gt, init_frame, G)

    return Ai, Bi, G


def pre_training(init_gt, init_frame, G):

    template = crop_search_window(init_gt, init_frame)
    fi = pre_process(template)
    
    Ai = G * np.conjugate(np.fft.fft2(fi))                # (1a)
    Bi = np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))  # (1b)

    for _ in range(NUM_PRETRAIN):
        fi = pre_process(random_warp(template))

        Ai = Ai + G * np.conjugate(np.fft.fft2(fi))               # (1a)
        Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi)) # (1b)

    return Ai, Bi

def track(image, position, Ai, Bi, G):

    response = predict(image, position, Ai/Bi)
    new_position = update_position(response, position)
    newAi, newBi = update(image, new_position, Ai, Bi, G)

    return new_position, newAi, newBi

def predict(frame, position, H):

    #----TODO (5)

    patch = crop_search_window(position, frame)

    patch_processed = pre_process(patch)

    F = np.fft.fft2(patch_processed)
    response_fft = H * F
    gi = np.fft.ifft2(response_fft)

    return np.real(gi)


def update(frame, position, Ai, Bi, G):

    #----TODO (5)

    patch = crop_search_window(position, frame)
    patch_processed = pre_process(patch)
    F = np.fft.fft2(patch_processed)

    Ai = (1 - LR) * Ai + LR * G * np.conj(F)
    Bi = (1 - LR) * Bi + LR * F * np.conj(F)

    return Ai, Bi


def update_position(spatial_response, position):

    #----TODO (6)

    x, y, w, h = position

    max_val = np.max(spatial_response)

    max_pos = np.where(spatial_response == max_val)           

    dy = int(np.mean(max_pos[0]))
    dx = int(np.mean(max_pos[1]))

    center_y = spatial_response.shape[0] // 2
    center_x = spatial_response.shape[1] // 2

    shift_y = dy - center_y
    shift_x = dx - center_x

    new_x = x + shift_x
    new_y = y + shift_y

    new_position = [new_x, new_y, w, h]

    return new_position


def bbox_iou(box1, box2):

    # Transform from center and width to exact coordinates
    b1_x1, b1_x2 = box1[0], box1[0] + box1[2]
    b1_y1, b1_y2 = box1[1], box1[1] + box1[3]
    b2_x1, b2_x2 = box2[0], box2[0] + box2[2]
    b2_y1, b2_y2 = box2[1], box2[1] + box2[3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1, a_min=0, a_max=None) * np.clip(inter_rect_y2 - inter_rect_y1, a_min=0, a_max=None)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def test_sequence(DATASET_DIR, sequence):

    seqdir = join(DATASET_DIR, sequence)
    imgdir = join(seqdir, 'color')
    imgnames = os.listdir(imgdir)                  
    imgnames.sort()

    print('init frame:', join(imgdir, imgnames[0]))
    init_img = cv2.imread(join(imgdir, imgnames[0]))
    gt_boxes = load_gt(join(seqdir, 'groundtruth.txt'))
    position = gt_boxes[0]
    Ai, Bi, G = initialize(init_img, position)

    if VISUALIZE:
        cv2.rectangle(init_img, (position[0], position[1]), (position[0]+position[2], position[1]+position[3]), (255, 0, 0), 2)
        cv2.imshow('demo', init_img)
        cv2.waitKey(0)
        

    results = []
    for imgname in imgnames[1:]:
        img = cv2.imread(join(imgdir, imgname))
        position, Ai, Bi = track(img, position, Ai, Bi, G)
        results.append(position.copy())

        if VISUALIZE:
            position = [round(x) for x in position]
            cv2.rectangle(img, (position[0], position[1]), (position[0]+position[2], position[1]+position[3]), (255, 0, 0), 2)
            cv2.imshow('demo', img)
            cv2.waitKey(10)


    return results, gt_boxes

DATASET_DIR = '/home/plorenc/Desktop/AiR_ISS/AVS/sequences'
# sequences = os.listdir(DATASET_DIR)
sequences = ['sunshade']
ious_per_sequence = {}
for sequence in sequences:

    results, gt_boxes = test_sequence(DATASET_DIR, sequence)
    ious = []
    for res_box, gt_box in zip(results, gt_boxes[1:]):
        iou = bbox_iou(res_box, gt_box)
        ious.append(iou)

    ious_per_sequence[sequence] = np.mean(ious)
    print(sequence, ':', np.mean(ious))

print('Mean IoU:', np.mean(list(ious_per_sequence.values())))