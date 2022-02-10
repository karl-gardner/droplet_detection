import os
import cv2
import numpy as np

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
  
  
  
  
  
  
  
def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), box_thick=1, fontsize = 1, tf = 1):
  # Add one xyxy box to image with label
  p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  cv2.rectangle(image, p1, p2, color, thickness=box_thick, lineType=cv2.LINE_AA)
  if label:
      w, h = cv2.getTextSize(label, 0, fontScale=fontsize, thickness=tf)[0]  # text width, height
      outside = p1[1] - h - 3 >= 0  # label fits outside box
      p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
      cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
      cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, fontsize, txt_color,
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
      
      
def save_results(images_path, yolo, model):
  os.mkdir('/test_results')
  os.mkdir('/test_results/gt_vs_pred')
  os.mkdir('/test_results/inputs')
  os.mkdir('/test_results/gts')
  os.mkdir('/test_results/preds')
  
  for count, f in enumerate(os.listdir(images_path)):
    im_file = images_path+'/'+f
    pred_file = yolo + '/runs/detect/exp/labels/' + f[0:-4] + '.txt'
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
    with open(pred_file) as lab:
      lines = lab.readlines()
      rows = len(lines)
      boxes = np.zeros((rows,4))
      classes = []
      conf = []
      for i, line in enumerate(lines):
        line = line.split()
        classes.append(int(line[0]))
        boxes[i,0] = float(line[1])
        boxes[i,1] = float(line[2])
        boxes[i,2] = float(line[3])
        boxes[i,3] = float(line[4])
        conf.append(float(line[5]))
    pred_boxes = xywhn2xyxy(boxes, w=544, h=544)
    
    if model == 'cell':
      # Save images with annotated ground truth labels
      im = np.copy(input_im)
      for i in range(gt_boxes.shape[0]):
        col = (0, 255, 0)
        b = gt_boxes[i,:]
        im = box_label(im, b, color=col, box_thick=4)
      cv2.imwrite('/test_results/gts/' + f[:-4] + '.png',im)

      # Save images with annotated predicted labels
      for i in range(pred_boxes.shape[0]):
        lab = "cell %.2f" % conf[i]
        col = (0, 0, 255)
        b = pred_boxes[i,:]
        im = box_label(im, b, lab, col, box_thick=3, fontsize=1.2, tf=4)
      cv2.imwrite('/test_results/gt_vs_pred/' + f[:-4] + '.png',im)
      
    if model == 'droplet':
      # Save images with annotated ground truth labels
      labels = ['drop_0cell', 'drop_1cell', 'drop_2cell', 'drop_3cell']
      colors = [(0,0,255), (0,255,255), (255,0,127), (255,0,255)]
      im = np.copy(input_im)
      for i in range(gt_boxes.shape[0]):
        b = gt_boxes[i,:]
        im = box_label(im, b, label=labels[classes[i]], color=colors[classes[i]], txt_color=(0,0,0), box_thick=1, fontsize=0.5, tf=2)
      cv2.imwrite('/test_results/gts/' + f[:-4] + '.png',im)
      
      # Save images with annotated predicted labels
      im = np.copy(input_im)
      for i in range(pred_boxes.shape[0]):
        b = pred_boxes[i,:]
        im = box_label(im, b, labels[classes[i]] + ' %.2f' % conf[i], color=colors[classes[i]], txt_color=(0,0,0), box_thick=1, fontsize=0.5, tf = 2)
      cv2.imwrite('/test_results/preds/' + f[:-4] + '.png',im)

