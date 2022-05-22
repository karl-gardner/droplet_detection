<h4 align="center">Author: Karl Gardner<br>PhD Candidate, Department of Chemical Engineering, Texas Tech University</h4>

<div align="center">
  <a href="https://www.depts.ttu.edu/che/research/li-lab/">
  <img src="https://user-images.githubusercontent.com/91646805/154190573-53e361f6-7c60-4062-b56b-7cbd11d39fc4.jpg"/></a><br><br>
  
  <a href="https://www.depts.ttu.edu/che/research/li-lab/">
  <img src="https://user-images.githubusercontent.com/91646805/156635015-0cdcb0bb-0482-4693-b096-04f2a78f6b8e.svg" height="32"/></a>
  
  <a href="https://vanapallilab.wixsite.com/microfluidics">
  <img src="https://user-images.githubusercontent.com/91646805/156635010-a1049d8a-a72e-4ed5-89ec-2ace11169d85.svg" height="32"/></a>
  
  <a href="https://www.depts.ttu.edu/che/">
  <img src="https://user-images.githubusercontent.com/91646805/156641068-be8f0336-89b5-43e9-aa64-39481ce37c94.svg" height="32"/></a>
  
  <a href="https://roboflow.com/">
  <img src="https://user-images.githubusercontent.com/91646805/156641388-c609a6aa-8fce-47f0-a111-abfde9c5da05.svg" height="32"/></a><br>
  
  <a href="https://colab.research.google.com/github/karl-gardner/droplet_detection/blob/master/yolov3.ipynb">
  <img src="https://user-images.githubusercontent.com/91646805/156640198-51f0ef4c-21c1-4d0f-aebd-861561dede95.svg" height="32"/></a>
  
  <a href="https://colab.research.google.com/github/karl-gardner/droplet_detection/blob/master/yolov5.ipynb">
  <img src="https://user-images.githubusercontent.com/91646805/156640073-0a7ad496-7691-4e1c-822c-b78f3e7d070b.svg" height="32"/></a>
  
  <a href="https://github.com/ultralytics">
  <img src="https://user-images.githubusercontent.com/91646805/156641066-fbc3635b-f373-4cb7-b141-9bcaad21beff.svg" height="32"/></a>


# Droplet Detection Model
Encapsulation of cells inside microfluidic droplets is central to several applications involving cellular analysis. Although, theoretically the encapsulation statistics are expected to follow Poisson distribution, experimentally this may not be achieved due to lack of full control of the experimental variables and conditions. Therefore, there is a need to automatically detect droplets and enumerate cell counts within droplets so that this can be used as process control feedback to adjust experimental conditions. In this study, we use a deep learning object detector called You Only Look Once (YOLO), an influential class of object detectors with several benefits over traditional methods. This paper investigates the application of both YOLOv3 and YOLOv5 object detectors in the development of an automated droplet and cell detector. Experimental data was obtained from a microfluidic flow focusing device with a dispersed phase of cancer cells. The microfluidic device contained an expansion chamber downstream of the droplet generator, allowing for visualization and recording of cell-encapsulated droplet images. In the procedure, a droplet bounding box is predicted, then cropped from the original image for the individual cells to be detected through a separate model for further examination. The system includes a production set for additional performance analysis with Poisson statistics while providing an experimental workflow with both droplet and cell models. The training set is collected and preprocessed before labeling and applying image augmentations, allowing for a generalizable object detector. Precision and recall were utilized as a validation and test set metric, resulting in a high mean average precision (mAP) metric for an accurate droplet detector. To examine model limitations, the predictions were compared to ground truth labels, illustrating that the YOLO predictions closely matched with the droplet and cell labels. Furthermore, it is demonstrated that droplet enumeration from the YOLOv3 model is consistent with hand counted ratios and the Poisson distribution, confirming that the platform can be used in real-time experiments for encapsulation optimization.
</div>

![project_workflow](https://user-images.githubusercontent.com/91646805/169369625-ad3ce7ff-9b38-4820-abef-8962b6750bf6.jpg)

<details>
<summary>Instructions (click to expand)</summary>
<br>

1) First create a folder in your google drive account called droplet_classification (This step is important in order to keep the directories in check)
2) Use this link <a href="https://drive.google.com/drive/folders/1Oo68HSdU-jzcBAEr0yeRuzuSxoprEP_D?usp=sharing">
  <img src="https://user-images.githubusercontent.com/91646805/156700933-5cc77dba-5df1-40c0-94c8-7459abb6402b.svg" height="18"/></a> to access the shared google drive folder
3) At the top there will be a dropdown arrow after the folder location (Shared with me > data_files): click on this dropdown arrow
4) Click on the "Add shortcut to Drive" button then navigate to inside your droplet_classification folder and click the blue "Add Shortcut" button.  This will add a shortcut to the shared google drive folder in your droplet_classification folder.
5) Open the yolov3 colab notebook from the colab badge provided, then click "Save a copy in Drive" under File > Save a copy in Drive.  Do the same for the provided yolov5 colab notebook.
6) This will save the two notebooks in the "Colab Notebooks" folder in your google drive.  Move these two notebooks to the droplet_classification folder and rename them yolov3.ipynb and yolov5.ipynb respectively in order for the directories to be correct.  The final droplet_classification folder should look like this:<img width="720" alt="image" src="https://user-images.githubusercontent.com/91646805/148874654-890a5d94-f9e9-4273-bcd8-318df44feca4.png">

7) Find the droplet model dataset here: <a href="https://universe.roboflow.com/karl-gardner-kmk9u/pc3dropletdetection2/5">
  <img src="https://user-images.githubusercontent.com/91646805/156698861-29c0ae55-eff3-4bfe-9dcc-fe06e5a1c6cd.svg" height="18"/></a> and you will see two datasets (No_Augmentation and final_dataset).  Start with the final_dataset and click on "Download" in the upper right corner.  Then, click "Sign in with Github" and follow the prompts to allow roboflow to sign in with github.  Or you may create a different account with roboflow.  Then, the download link will bring you to a pop up that says Export.  For the "Format" click on the YOLO v5 PyTorch and "show download code" on the bottom.  You will then see a link that you can use to enter in the colab notebook.  The final page should look like this but with your own link under the red stripe: <img width="925" alt="image" src="https://user-images.githubusercontent.com/91646805/149068681-5d5529b4-7d6f-41f5-8710-98f04c780654.png"> Then copy this link into the section of both notebooks (yolov3.ipynb and yolov5.ipynb) that says "Curl droplet data from roboflow > Data with Augmentation for Training > [ROBOFLOW-API-KEY]": ![image](https://user-images.githubusercontent.com/91646805/151044698-1d03e6c8-7d2b-401c-b632-b00d1fbe6821.png)  Copy your download link inside of the double quaotations as in the red box in the image provided.

8) Repeat step 7 for the droplet dataset with no augmentations (No_Augmentation): ![image](https://user-images.githubusercontent.com/91646805/151045660-a4fb9e26-a108-4369-aba9-63be2bb9efc1.png)

9) Repeat steps 7 and 8 with the cell dataset <a href="https://universe.roboflow.com/karl-gardner-kmk9u/cropped_drops2/1">
  <img src="https://user-images.githubusercontent.com/91646805/156698862-6591ba12-a90f-4495-8736-cab83f5cd237.svg" height="18"/></a>  This dataset only needs to be copied into the yolov3.ipynb notebook since it is not used in the yolov5.ipynb notebook.
10) You can now use both notebooks to perform more testing or contribute to the project.  You can find the code written for many of the figures in the final paper: DOI Website
</details>

<details>
<summary>Testing (click to expand)</summary><br>
Nearly all figures and tables from the paper are outlined in yolov3.ipynb and yolov5.ipynb colab notebooks. For example Table 2 displays the annotation summary for cell and droplet models before augmentations. This can be shown in section 2. of the colab notebook:
![table_2](https://user-images.githubusercontent.com/91646805/169673685-cf782a19-ebbf-4f5e-82ba-0f7d42a37402.jpg)
You may run this for example by first uncommenting section 1.1 labeled "Data with No Augmentation (No_Augmentation)" then uncommenting section 2. labeled: "For droplet model". Then the following output will be printed:
![table_2](https://user-images.githubusercontent.com/91646805/169673813-3c9c0321-fec8-4ae4-a092-fdf8be4f3464.jpg)
![table_2](https://user-images.githubusercontent.com/91646805/169673815-b5ea0589-038f-4a4c-8d84-2fdd2b69ba58.png)
![table_2_droplet](https://user-images.githubusercontent.com/91646805/169673816-8f4f5a29-fe0d-43aa-bd8d-3f9e9f994a8f.png)

</details>

<details open>
<summary>Contributions</summary><br>

 **Publication Authors:**<br>Karl Gardner, Md Mezbah Uddin, Linh Tran, Thanh Pham, Siva Vanapalli, and Wei Li<br><br>
 
 **Publication Acknowledgements:**<br>WL acknowledge support from National Science Foundation (CBET, Grant No. 1935792) and National Institute of Health (IMAT, Grant No. 1R21CA240185-01).
</details>
