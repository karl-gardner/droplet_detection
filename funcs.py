import os
import cv2
import numpy as np
import copy

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
  
  
  
  
  
  
  
def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
  # Add one xyxy box to image with label
  p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  cv2.rectangle(image, p1, p2, color, thickness=1, lineType=cv2.LINE_AA)
  if label:
      tf = max(1, 1)  # font thickness
      w, h = cv2.getTextSize(label, 0, fontScale=1 / 3, thickness=tf)[0]  # text width, height
      outside = p1[1] - h - 3 >= 0  # label fits outside box
      p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
      cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
      cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, 1 / 3, txt_color,
                  thickness=tf, lineType=cv2.LINE_AA)
  return image
      
def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y
      
      
def save_results(images_path):
  os.mkdir('/test_results')
  os.mkdir('/test_results/gt_vs_pred')
  os.mkdir('/test_results/inputs')
  os.mkdir('/test_results/gts')
  os.mkdir('/test_results/preds')
  
  for count, f in enumerate(os.listdir(images_path)):
    im_file = images_path+'/'+f
    pred_file = "runs/detect/exp/"+f
    label_file = images_path + '/../labels/' + f[0:-4] + '.txt'
    
    input_im = cv2.imread(im_file)
    cv2.imwrite('/test_results/inputs/' + f[:-4] + '.png', input_im)
    
    # Save boxes with ground thruth labels in numpy array
    with open(label_file) as lab:
      lines = lab.readlines()
      rows = len(lines)
      boxes = np.zeros((rows,4))
      classes = []
      for i, line in enumerate(lines):
        line = line.split()
        classes.append(int(line[0]))
        boxes[i,0] = float(line[1])
        boxes[i,1] = float(line[2])
        boxes[i,2] = float(line[3])
        boxes[i,3] = float(line[4])
    gt_boxes = xywhn2xyxy(boxes, w=544, h=544)
    
    # Save boxes with predicted labels in numpy array
    with open(label_file) as lab:
      lines = lab.readlines()
      rows = len(lines)
      boxes = np.zeros((rows,4))
      classes = []
      for i, line in enumerate(lines):
        line = line.split()
        classes.append(int(line[0]))
        boxes[i,0] = float(line[1])
        boxes[i,1] = float(line[2])
        boxes[i,2] = float(line[3])
        boxes[i,3] = float(line[4])
    pred_boxes = xywhn2xyxy(boxes, w=544, h=544)
    
    im = copy.deepcopy(input_im)
    for i in range(gt_boxes.shape[0]):
      lab = "cell"
      col = (0,0,255)
      b = gt_boxes[i,:]
      im = box_label(im, b, lab, col)
    cv2.imwrite('/test_results/gts/' + f[:-4] + '.png',im)
    
    for i in range(gt_boxes.shape[0]):
      lab = "cell"
      col = (0,0,255)
      b = pred_boxes[i,:]
      im = box_label(im, b, lab, col)
    cv2.imwrite('/test_results/gt_vs_pred/' + f[:-4] + '.png',im)

