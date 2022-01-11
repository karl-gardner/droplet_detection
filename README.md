# dropletdetection
Real time detection of cell encapsulation for Poisson distribution analysis and cell visualization with YOLOv3 and YOLOv5

![experimental_workflow](https://user-images.githubusercontent.com/91646805/148269422-758ea029-7165-4259-98b7-d89a26e66361.png)


hello

Directions to Run Droplet Detection Models:

1) First create a folder in your google drive account called droplet_classification (This step is important in order to keep the directories in check)
2) Use this link: https://drive.google.com/drive/folders/1Oo68HSdU-jzcBAEr0yeRuzuSxoprEP_D?usp=sharing to access the shared google drive folder
3) At the top there will be a dropdown arrow after the folder location (Shared with me > data_files): click on this dropdown arrow
4) Click on the "Add shortcut to Drive" button then navigate to inside your droplet_classification folder and click the blue "Add Shortcut" button.  This will add a shortcut to the shared google drive folder in your droplet_classification folder.
5) Open the yolov3 colab notebook from the colab badge provided, then click "Save a copy in Drive" under File > Save a copy in Drive.  Do the same for the provided yolov5 colab notebook.
6) This will save the two notebooks in the "Colab Notebooks" folder in your google drive.  Move these two notebooks to the droplet_classification folder and rename them yolov3.ipynb and yolov5.ipynb respectively in order for the directories to be correct.  The final droplet_classification folder should look like this:<img width="720" alt="image" src="https://user-images.githubusercontent.com/91646805/148874654-890a5d94-f9e9-4273-bcd8-318df44feca4.png">

7) Click the link here: https://universe.roboflow.com/karl-gardner-kmk9u/pc3dropletdetection2/6 and you will see two datasets (No_Augmentation and final_dataset).  Start with the final_dataset and click on "Download" in the upper right corner.  Then, click "Sign in with Github" and follow the prompts to allow roboflow to sign in with github.  Or you may create an account with roboflow.  Then, the download link will bring you to a pop up that says Export.  For the "Format" click on the YOLO v5 PyTorch and "show download code" on the bottom.  You will then see a link that you can use to enter in the colab notebook.  The final page should look like this but with your own link under the red stripe: ![image](https://user-images.githubusercontent.com/91646805/149022115-300412ca-9951-4356-8f06-73b8f4172b20.png) Then copy this link into the section of both notebooks that says "Curl droplet data from roboflow > Data with Augmentation for Training > [ROBOFLOW-API-KEY]"

8) Repeat step 7 for the droplet dataset with no augmentations (No_Augmentation).  Then, do the same thing with the cell dataset for the yolov3.ipynb with the link provided: https://universe.roboflow.com/karl-gardner-kmk9u/cropped_drops2/1

9) You can now use both notebooks to perform more testing or contribute to the project.  You can find the code written for many of the figures in the final paper: DOI Website
