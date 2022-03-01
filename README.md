Author: Karl Gardner<br>PhD Candidate, Department of Chemical Engineering ([Dr. Wei Li](https://www.depts.ttu.edu/che/research/li-lab/)), Texas Tech University
<br><br>
![ttu_chemical](https://user-images.githubusercontent.com/91646805/154190573-53e361f6-7c60-4062-b56b-7cbd11d39fc4.jpg)

# Droplet Detection Model

<div align="center">
  <a href="https://colab.research.google.com/github/karl-gardner/droplet_detection/blob/master/yolov3.ipynb">
  <img src="https://img.shields.io/badge/-YOLOv3-blue?logo=googlecolab&logoWidth=18&labelColor=grey" /></a>
  
  <a href="https://colab.research.google.com/github/karl-gardner/droplet_detection/blob/master/yolov5.ipynb">
  <img src="https://img.shields.io/badge/-YOLOv5-blue?logo=googlecolab&logoWidth=18&labelColor=grey" /></a><br>

  
  <a href="https://www.depts.ttu.edu/che/">
  <img src="https://user-images.githubusercontent.com/91646805/156217188-9ce740df-8856-435b-ab1c-9c1500a13a43.svg" /></a><br>
  
Artificial Intelligence (AI) is becoming increasingly useful in numerous microfluidic platforms for biomedical applications. Although other researchers have introduced machine learning models for droplet generation and classification of mixing in droplets, automatic detection of droplets with cell encapsulation has not been explored, which hinders the implication of AI in droplet microfluidics for single cell-based applications.  You Only Look Once (YOLO), an influential class of object detectors, has had several improvements since the original publication by Joseph Redmon in 2016. This paper investigates the application of both YOLOv3 and YOLOv5 object detectors in the development of an automated droplet and cell detector. In the procedure, a droplet bounding box is predicted, then cropped from the original image for the individual cells to be detected through a separate model for full examination. The system includes a production set for additional performance analysis with Poisson statistics while providing a full experimental workflow with both droplet and cell models. The droplet generator employed contains an expansion chamber allowing for visualization of cell-encapsulated droplets with a digital camera. The training set is collected and preprocessed before labeling and applying image augmentations, allowing for a generalizable object detector. The mean average precision (mAP) is utilized as a validation and test set metric to investigate the model performance while individual predictions are explored to gather a deeper understanding of the results. Furthermore, it is demonstrated that droplet enumeration from the YOLOv3 model is consistent with hand counted ratios and the Poisson distribution, confirming that the platform can be used in real-time experiments for cell encapsulation optimization.
</div>

![workflow](https://user-images.githubusercontent.com/91646805/156113705-35f9fa1e-6913-4ecb-bc1b-c4979e2d7acf.jpg)


<details>
<summary>Instructions (click to expand)</summary>
<br>

1) First create a folder in your google drive account called droplet_classification (This step is important in order to keep the directories in check)
2) Use this link: https://drive.google.com/drive/folders/1Oo68HSdU-jzcBAEr0yeRuzuSxoprEP_D?usp=sharing to access the shared google drive folder
3) At the top there will be a dropdown arrow after the folder location (Shared with me > data_files): click on this dropdown arrow
4) Click on the "Add shortcut to Drive" button then navigate to inside your droplet_classification folder and click the blue "Add Shortcut" button.  This will add a shortcut to the shared google drive folder in your droplet_classification folder.
5) Open the yolov3 colab notebook from the colab badge provided, then click "Save a copy in Drive" under File > Save a copy in Drive.  Do the same for the provided yolov5 colab notebook.
6) This will save the two notebooks in the "Colab Notebooks" folder in your google drive.  Move these two notebooks to the droplet_classification folder and rename them yolov3.ipynb and yolov5.ipynb respectively in order for the directories to be correct.  The final droplet_classification folder should look like this:<img width="720" alt="image" src="https://user-images.githubusercontent.com/91646805/148874654-890a5d94-f9e9-4273-bcd8-318df44feca4.png">

7) Click the link here for the droplet model dataset: https://universe.roboflow.com/karl-gardner-kmk9u/pc3dropletdetection2/6 and you will see two datasets (No_Augmentation and final_dataset).  Start with the final_dataset and click on "Download" in the upper right corner.  Then, click "Sign in with Github" and follow the prompts to allow roboflow to sign in with github.  Or you may create a different account with roboflow.  Then, the download link will bring you to a pop up that says Export.  For the "Format" click on the YOLO v5 PyTorch and "show download code" on the bottom.  You will then see a link that you can use to enter in the colab notebook.  The final page should look like this but with your own link under the red stripe: <img width="925" alt="image" src="https://user-images.githubusercontent.com/91646805/149068681-5d5529b4-7d6f-41f5-8710-98f04c780654.png"> Then copy this link into the section of both notebooks (yolov3.ipynb and yolov5.ipynb) that says "Curl droplet data from roboflow > Data with Augmentation for Training > [ROBOFLOW-API-KEY]": ![image](https://user-images.githubusercontent.com/91646805/151044698-1d03e6c8-7d2b-401c-b632-b00d1fbe6821.png)  Copy your download link inside of the double quaotations as in the red box in the image provided.

8) Repeat step 7 for the droplet dataset with no augmentations (No_Augmentation): ![image](https://user-images.githubusercontent.com/91646805/151045660-a4fb9e26-a108-4369-aba9-63be2bb9efc1.png)

9) Repeat steps 7 and 8 with the cell dataset in the link provided: https://universe.roboflow.com/karl-gardner-kmk9u/cropped_drops2/1.  This dataset only needs to be copied into the yolov3.ipynb notebook since it is not used in the yolov5.ipynb notebook.
10) You can now use both notebooks to perform more testing or contribute to the project.  You can find the code written for many of the figures in the final paper: DOI Website
</details>

<details>
<summary>Contributions (click to expand)</summary><br>

 **Publication Authors:**<br>Karl Gardner, Md Mezbah Uddin, Linh Tran, Thanh Pham, Siva Vanapalli, and Wei Li<br><br>
 
 **Publication Acknowledgements:**<br>WL acknowledge support from National Science Foundation (CBET, Grant No. 1935792) and National Institute of Health (IMAT, Grant No. 1R21CA240185-01).
</details>
