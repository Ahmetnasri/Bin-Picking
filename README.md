# Bin-Picking

- **libraries**

    - **python 3.7**

    - **opencv-python 3.4.2.17**

    - **pytorch 1.7.1**

    - **torchvision 0.8.2**



Set up the environment using command:

    conda env create -f environment.yml


Download the weights from this link:
[https://drive.google.com/file/d/1ZnVUZWofISRliAbAJ3eGeOkRRJ83Ux9V/view?usp=sharing]

Put the weights files in a folder and copy the path of yolact_resnet50_Intel_RealSense_283_3974_interrupt.pth to the 130 line in the eval.py script

Be sure that you pluged the Intel Realsense Camera and just run the demo.py script and make your inference. 

        python demo.py

## Take a closer look
https://github.com/Ahmetnasri/Bin-Picking/assets/63724301/b3e6939c-d993-4067-bf7f-568facfac273
