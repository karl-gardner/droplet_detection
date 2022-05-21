#imports
import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y
  
# Print and return images and labels in a list format
def drop_labels(label_path = None, set = None):
  if label_path == None:
    print("Pass a file as an argument")
    pass
  else:
    images = 0
    classes = []
    for f in os.listdir(label_path):
      images += 1
      with open(label_path + '/' + f) as read_file:
          lines = read_file.readlines()
          for line in lines:
            line = line.split()
            classes.append(int(line[0]))

    zero_cell = classes.count(0)
    one_cell = classes.count(1)
    two_cell = classes.count(2)
    three_cell = classes.count(3)
    
    print(set + " Set:")
    print("drop_0cell: " + str(zero_cell))
    print("drop_1cell: " + str(one_cell))
    print("drop_2cell: " + str(two_cell))
    print("drop_3cell: " + str(three_cell))
    print("combined: " + str(zero_cell + one_cell + two_cell + three_cell))
    print("images: " + str(images) + '\n')
    
    return [zero_cell, one_cell, two_cell, three_cell, images]

  
# Print and return images and labels in a list format
def cell_labels(label_path = None, set = None):
  if label_path == None:
    print("Pass a file as an argument")
    pass
  else:
    images = 0
    classes = []
    for f in os.listdir(label_path):
      images += 1
      with open(label_path + '/' + f) as read_file:
          lines = read_file.readlines()
          for line in lines:
            line = line.split()
            classes.append(int(line[0]))

      cell = classes.count(0)
    
    print(set + " Set:")
    print("cells: " + str(cell))
    print("images: " + str(images) + '\n')
    
    return [cell, images]


def save_cropped(datasets, counter_tot):
  if counter_tot  == 0:
    global set_index 
    set_index = 0
    shutil.rmtree("/cropped_drops", ignore_errors=True)
    os.mkdir("/cropped_drops")
  
  dataset = datasets[set_index]
  counter_set = 0
  for j, im_file in enumerate(os.listdir(f"../{dataset}/images")):
    if j % 5 == 0:
      f_label = im_file[0:-4]+".txt"
      with open(f"../{dataset}/labels/{f_label}") as f:
        lines = f.readlines()
        rows = len(lines)
        boxes = []
        for line in lines:
          line = line.split()
          if int(line[0]) == 0:
            continue
          x = float(line[1])
          y = float(line[2])
          mean_wh = (float(line[3])+float(line[4]))/2
          if x + mean_wh/2 > 1:
            x = 1 - mean_wh/2
          if y + mean_wh/2 > 1:
            y = 1 - mean_wh/2
          if x-mean_wh/2 < 0:
            x = mean_wh/2
          if y-mean_wh/2 < 0:
            y = mean_wh/2
          boxes.append([x,y,mean_wh,mean_wh])
      boxes = xywhn2xyxy(np.array(boxes), w=544, h=544)
      im = cv2.imread(f"../{dataset}/images/{im_file}")
      for i in range(boxes.shape[0]):
        # May not be square by one pixel... make square
        if int(boxes[i,3])-int(boxes[i,1]) < int(boxes[i,2])-int(boxes[i,0]):
          boxes[i,3] += 1
        if int(boxes[i,3])-int(boxes[i,1]) > int(boxes[i,2])-int(boxes[i,0]):
          boxes[i,2] += 1
        cropped = im[int(boxes[i,1]):int(boxes[i,3]),int(boxes[i,0]):int(boxes[i,2]),:]
        cv2.imwrite(f"/cropped_drops/im_{counter_tot}.png",cropped)
        counter_tot += 1
        counter_set += 1
  print(f"number of images saved from {dataset} set: {counter_set}")
  if set_index < len(sets)-1:
    set_index += 1
    counter_tot = save_cropped(datasets, counter_tot)
  return counter_tot



def save_map(results_path):
  map5 = []
  map595 = []
  with open(results_path, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(spamreader)
    for row in spamreader:
      for i, elem in enumerate(row):
        if i == 6:
          map5.append(float(elem.strip()))
        if i == 7:
          map595.append(float(elem.strip()))
  fig, axs = plt.subplots(1, 2, figsize=(18,6))
  axs[0].plot(map5, color="blue", marker='.', linewidth=2, markersize=12)
  axs[0].set_ylim(0, 1)
  axs[0].tick_params(axis='both', which='major', labelsize=29)
  axs[0].set_xlabel("Epochs", fontsize=35, fontfamily="Arial")
  axs[0].set_ylabel("mAP @ IOU 0.5", fontsize=35, fontfamily="Arial")
  axs[1].plot(map595, color="blue", marker='.', linewidth=2, markersize=12)
  axs[1].set_ylim(0, 1)
  axs[1].tick_params(axis='both', which='major', labelsize=29)
  axs[1].set_ylabel("mAP\n@ IOU 0.5:0.95", fontsize=35, fontfamily="Arial")
  axs[1].set_xlabel("Epochs", fontsize=35, fontfamily="Arial")
  fig.suptitle("YOLOv3",fontsize=45,fontfamily="Arial", y=1.05)
  fig.tight_layout(pad=4)

  fig.savefig("/mAP_yolov3.png", dpi=500, bbox_inches='tight')

  
def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), box_thick=1, fontsize = 1, tf = 1, filled = True):
  # Add one xyxy box to image with label
  p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  cv2.rectangle(image, p1, p2, color, thickness=box_thick, lineType=cv2.LINE_AA)
  if label:
      w, h = cv2.getTextSize(label, 0, fontScale=fontsize, thickness=tf)[0]  # text width, height
      outside = p1[1] - h - 3 >= 0  # label fits outside box
      p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
      if filled:
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
      cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, fontsize, txt_color,
                  thickness=tf, lineType=cv2.LINE_AA)
  return image
      
    
    
def save_labels(images_path, model, gt_colors=[(0, 255, 0)], pred_colors=[(0,0,255)], pred_labels = ''):
  os.mkdir('/label_results')
  os.mkdir('/label_results/inputs')
  
  # Save images with annotated ground truth labels
  if model == 'droplet':
    gt_labels = ['drop_0cell', 'drop_1cell', 'drop_2cell', 'drop_3cell']
    gt_box_thick = 1
    pred_colors = gt_colors
    text_color = (0, 0, 0)
    box_thickness = 1
    font_size = .55
    font_thickness = 1
  
  if model == 'cell':
    gt_box_thick = 4
    gt_labels = ['']
    text_color = (255, 255, 0)
    box_thickness = 3
    font_size = 1.2
    font_thickness = 4
  
  f = os.listdir(images_path)
  im_file = images_path + '/' + f[0]
  all_gt_boxes = []
  all_gt_classes = []
  all_pred_boxes = []
  all_pred_classes = []
  all_conf = []

  # Try block for ground truth labels
  try:
    assert(len(os.listdir(images_path + '/../labels')) != 0)
  except:
    print("no ground truth labels for these images")
  else:
    os.mkdir('/label_results/gts')
    for f in sorted(os.listdir(images_path)):
      im_file = images_path + '/' + f
      input_im = cv2.imread(im_file)
      gt_file = images_path + '/../labels/' + f[0:-4] + '.txt'
      with open(gt_file) as lab:
        lines = lab.readlines()
        rows = len(lines)
        boxes = np.zeros((rows,4))
        gt_classes = []
        for i, line in enumerate(lines):
          line = line.split()
          gt_classes.append(int(line[0]))
          boxes[i,0] = float(line[1])
          boxes[i,1] = float(line[2])
          boxes[i,2] = float(line[3])
          boxes[i,3] = float(line[4])
      gt_boxes = xywhn2xyxy(boxes, w=544, h=544)
      all_gt_boxes.append(gt_boxes)
      all_gt_classes.append(gt_classes)
      
      gt_im = np.copy(input_im)
      for i in range(gt_boxes.shape[0]):
        b = gt_boxes[i,:]
        gt_im = box_label(gt_im, b, label=gt_labels[gt_classes[i]], color=gt_colors[gt_classes[i]], txt_color=(0,0,0), box_thick=gt_box_thick, fontsize=0.55, tf=1)
#       gt_im = box_label(gt_im, b, label='', color=gt_colors[gt_classes[i]], txt_color=(0,0,0), box_thick=2, fontsize=0.55, tf=1)

      # Now save ground truth images and input images
      cv2.imwrite('/label_results/inputs/' + f[:-4] + '.png', input_im)
      cv2.imwrite('/label_results/gts/' + f[:-4] + '.png', gt_im)
      
  # If block for predicted labels
  try:
    assert(len(os.listdir('runs/detect/exp/labels')) != 0)
  except:
    print("no predections for these images or the first prediction for the set has no detections")
  else:
    os.mkdir('/label_results/preds')
    for f in sorted(os.listdir(images_path)):
      im_file = images_path + '/' + f
      input_im = cv2.imread(im_file)
      pred_file = 'runs/detect/exp/labels/' + f[0:-4] + '.txt'
      try:
        lab = open(pred_file)
      except:
        cv2.imwrite('/label_results/inputs/' + f[:-4] + '.png', input_im)
        cv2.imwrite('/label_results/preds/' + f[:-4] + '.png',input_im)
        continue
      else:
        lines = lab.readlines()
        rows = len(lines)
        boxes = np.zeros((rows,4))
        pred_classes = []
        conf = []
        for i, line in enumerate(lines):
          line = line.split()
          pred_classes.append(int(line[0]))
          boxes[i,0] = float(line[1])
          boxes[i,1] = float(line[2])
          boxes[i,2] = float(line[3])
          boxes[i,3] = float(line[4])
          conf.append(float(line[5]))
        lab.close()
        pred_boxes = xywhn2xyxy(boxes, w=544, h=544)
        all_pred_boxes.append(pred_boxes)
        all_pred_classes.append(pred_classes)
        all_conf.append(conf)

        pred_im = np.copy(input_im)
        for i in range(pred_boxes.shape[0]):
          b = pred_boxes[i,:]
          pred_im = box_label(pred_im, b, label=pred_labels + ('%.2f' % conf[i]), color=pred_colors[pred_classes[i]],
                         txt_color=text_color, box_thick=box_thickness, fontsize=font_size, tf =font_thickness)

        cv2.imwrite('/label_results/inputs/' + f[:-4] + '.png', input_im)
        cv2.imwrite('/label_results/preds/' + f[:-4] + '.png',pred_im)
  
  # If block for ground truth and predicted labels
  try:
    assert(len(os.listdir(images_path + '/../labels')) != 0)
    assert(len(os.listdir('runs/detect/exp/labels')) != 0)
  except:
    pass
  else:
    os.mkdir('/label_results/gt_preds')
    for j, f in enumerate(sorted(os.listdir(images_path))):
      im_file = images_path + '/' + f
      gt_pred_im = cv2.imread(im_file)
      for i in range(all_gt_boxes[j].shape[0]):
        gt_b = all_gt_boxes[j][i,:]
        gt_pred_im = box_label(gt_pred_im, gt_b, label=gt_labels[all_gt_classes[j][i]], color=gt_colors[all_gt_classes[j][i]], 
                               txt_color=(0,0,0), box_thick=gt_box_thick, fontsize=0.55, tf=1)

      for i in range(all_pred_boxes[j].shape[0]):
        pred_b = all_pred_boxes[j][i,:]
        gt_pred_im = box_label(gt_pred_im, pred_b, '%.2f' % all_conf[j][i], color=pred_colors[all_pred_classes[j][i]],
                     txt_color=(255,255,0), box_thick=3, fontsize=font_size, tf =font_thickness)
      cv2.imwrite('/label_results/gt_preds/' + f[:-4] + '.png',gt_pred_im)
        
