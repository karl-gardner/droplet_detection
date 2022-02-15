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
      
    
    
def save_labels(images_path, model, yolo='yolov3'):
  os.mkdir('/label_results')
  os.mkdir('/label_results/inputs')
  
  # Save images with annotated ground truth labels
  if model == 'droplet':
    gt_labels = ['drop_0cell', 'drop_1cell', 'drop_2cell', 'drop_3cell']
    gt_box_thick = 1
    pred_labels = ['', '', '', '']
    gt_colors = pred_colors = [(0,0,255), (0,255,255), (0,255, 0), (255,0,255)]
    text_color = (0, 0, 0)
    box_thickness = 1
    font_size = .55
    font_thickness = 1
  
  if model == 'cell':
    gt_labels = ['']
    gt_box_thick = 4
    pred_labels = ['cell']
    gt_colors = [(0, 255, 0)]
    pred_colors = [(0, 0, 255)]
    text_color = (255, 255, 255)
    box_thickness = 3
    font_size = 1.2
    font_thickness = 4
  
  f = os.listdir(images_path)
  im_file = images_path + '/' + f[0]
  pred_file = yolo + '/runs/detect/exp/labels/' + f[0][0:-4] + '.txt'
  gt_file = images_path + '/../labels/' + f[0][0:-4] + '.txt'
  all_gt_boxes = []
  all_gt_classes = []
  all_pred_boxes = []
  all_pred_classes[]

  try:
    lab = open(gt_file)
  except:
    print("no ground truth labels for these images")
  else:
    lab.close()
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
      
      gt_im = np.copy(input_im)
      for i in range(gt_boxes.shape[0]):
        b = gt_boxes[i,:]
        gt_im = box_label(gt_im, b, label=gt_labels[gt_classes[i]], color=gt_colors[gt_classes[i]], txt_color=(0,0,0), box_thick=gt_box_thick, fontsize=0.55, tf=1)
#       gt_im = box_label(gt_im, b, label='', color=gt_colors[gt_classes[i]], txt_color=(0,0,0), box_thick=2, fontsize=0.55, tf=1)

      # Now save ground truth images and input images
      cv2.imwrite('/label_results/inputs/' + f[:-4] + '.png', input_im)
      cv2.imwrite('/label_results/gts/' + f[:-4] + '.png', gt_im)
      
  # Try block for predicted labels
  try:
    lab = open(pred_file)
  except:
    print("no predections for these images or the first prediction for the set has no detections")
  else:
    lab.close()
    os.mkdir('/label_results/preds')
    for f in sorted(os.listdir(images_path)):
      im_file = images_path + '/' + f
      input_im = cv2.imread(im_file)
      pred_file = yolo + '/runs/detect/exp/labels/' + f[0:-4] + '.txt'
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

        pred_im = np.copy(input_im)
        # Now save (1) predicted labels
        for i in range(pred_boxes.shape[0]):
          b = pred_boxes[i,:]
          pred_im = box_label(pred_im, b, pred_labels[pred_classes[i]] + ' %.2f' % conf[i], color=pred_colors[pred_classes[i]],
                         txt_color=text_color, box_thick=box_thickness, fontsize=font_size, tf =font_thickness)

        cv2.imwrite('/label_results/inputs/' + f[:-4] + '.png', input_im)
        cv2.imwrite('/label_results/preds/' + f[:-4] + '.png',pred_im)
        
  try:
    gt_pred_im = gt_im
  except:
    pass
  else:
    os.mkdir('/label_results/gt_pred')
    for j, f in enumerate(sorted(os.listdir(images_path))):
      im_file = images_path + '/' + f
      gt_pred_im = cv2.imread(im_file)
      for i in range(all_gt_boxes[j].shape[0]):
        gt_b = all_gt_boxes[j][i,:]
        print("i: ",i)
        print("gt_labels: ",gt_labels)
        print("gt_colors: ",gt_colors)
        print("gt_classes: ",gt_classes)
        gt_pred_im = box_label(gt_pred_im, gt_b, label=gt_labels[all_gt_classes[j][i]], color=gt_colors[all_gt_classes[j][i]], 
                               txt_color=(0,0,0), box_thick=gt_box_thick, fontsize=0.55, tf=1)
      
      for i in range(all_pred_boxes[j].shape[0]):
        pred_b = all_pred_boxes[j][i,:]
        gt_pred_im = box_label(gt_pred_im, pred_b, pred_labels[all_pred_classes[j][i]] + ' %.2f' % conf[i], color=pred_colors[all_pred_classes[j][i]],
                     txt_color=text_color, box_thick=1, fontsize=font_size, tf =font_thickness)
      cv2.imwrite('/label_results/gt_preds/' + f[:-4] + '.png',gt_pred_im)
        
        
        
        
        
        
#         if gt_im:
#           gt_im = box_label(gt_im, b, pred_labels[pred_classes[i]] + ' %.2f' % conf[i], color=pred_colors[pred_classes[i]],
#                          txt_color=text_color, box_thick=1, fontsize=0.55, tf =font_thickness)
# #         if model == 'cell':
# #           input_im = box_label(input_im, b, pred_labels[pred_classes[i]] + ' %.2f' % conf[i], color=pred_colors[pred_classes[i]], box_thick=3, fontsize=1.2, tf=4)
# #           if gt_im:
# #             gt_im = box_label(gt_im, b, pred_labels[pred_classes[i]] + ' %.2f' % conf[i], color=pred_colors[pred_classes[i]], box_thick=3, fontsize=1.2, tf=4)
      
#       # Now save predicted labels
#       os.mkdir('/label_results/preds')
#       cv2.imwrite('/label_results/preds/' + f[:-4] + '.png',input_im)
      
#       if gt_im:
#         # Now save predicted with ground truth labels
#         if count == 0:
#           os.mkdir('/label_results/gt_vs_pred')
#         cv2.imwrite('/label_results/gt_vs_pred/' + f[:-4] + '.png', gt_im)
        
      
