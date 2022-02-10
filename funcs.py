import os

# Print and return images and labels
def count_droplet_labels(label_path = None, set = None):
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
    print("images: " + str(images))
    
    return (zero_cell, one_cell, two_cell, three_cell)
