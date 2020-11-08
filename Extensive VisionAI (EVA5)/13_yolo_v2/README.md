# EVA5 S13 _YoloV3
________
#### YoloV3 Training on Colab with custom dataset (object detection for hat, mask and boots dataset)

#### Links:

    Colab - https://colab.research.google.com/drive/1dZtOHVrMIE-oocGbbhXOEfmy14iTQ1ZE?usp=sharing

    Dataset - https://drive.google.com/drive/folders/12L5hphqJSu2C9azQJjfoH3Zm3a-sITpV?usp=sharing

    Weights - https://drive.google.com/file/d/1CYS2jfjgGaLvsxQFjSRHc1VgxG3PCTfs/view?usp=sharing

    github - https://github.com/vvshankar78/DeepLearning/tree/master/Extensive%20VisionAI%20(EVA5)/13_yolo_v2



#### Steps:
The objective of this exercise is divided primarily into 3 parts.
1. OpenCV Yolo - Try out the OpenCV Yolo on a custom image containing the objects in coco.
2. NEW ANNOTATION TOOL- Use the new annotation tool(https://github.com/miki998/YoloV3_Annotation_Tool) to annotate the custom dataset. Since this tool does not work well on colab, it was run on local jupyter notebook. The Annotated dataset was shared with Zoheb for preparation of larger dataset. (https://drive.google.com/drive/folders/1OKDbct5RT1QLbVQz72BbhiSLyWEhwsy0?usp=sharing). The output folder structure and file names of this tool is in the same format as accepted by Yolov3.

3. YOLO V3 for Custom Dataset :

    a. A reference colab file provided where in initial weights were provided and the steps as described in https://colab.research.google.com/drive/1LbKkQf4hbIuiUHunLlvY-cc0d_sNcAgS

    b. DATASET Description :

    Once the coco dataset for trained sucessfully, the same model is used to retrain for the full custom dataset as provided by theschoolofai. The images were cleaned up and little more prep files were generated and stored in the following location          is stored in shape file, and the final dataset folder structure is stored in the following link- https://drive.google.com/drive/folders/12L5hphqJSu2C9azQJjfoH3Zm3a-sITpV?usp=sharing

     - The folder structure of the dataset is described below.      

        data
          --YoloV3_Dataset - Images and labels folder - this is created by annotation tool and shared by TSAI.

            --images/
              --img001.jpg
              --img002.jpg
              --...
            --labels/
              --img001.txt
              --img002.txt
              --...

            train.data -- Contains number of classes, location of train and validation dataset annotation names and classes names. -
                ./data/YoloV3_Dataset/images/img_001.jpg
                ./data/YoloV3_Dataset/images/img_002.jpg
                ./data/YoloV3_Dataset/images/img_003.jpg
                ./data/YoloV3_Dataset/images/img_004.jpg

            The train.txt file will look something like this -
            classes=4
            train=data/YoloV3_Dataset/train.txt
            valid=data/YoloV3_Dataset/test.txt
            names=data/YoloV3_Dataset/classes.txt

            train.shapes -- contains the dimension of each of test and train image. The sample file will look something like this.
                900 568
                899 600
                900 600
                899 600

            classes.names #your class names
                hardhat
                vest
                mask
                boots


4. For our custom dataset we have 4 classes, VOLOv3's output vector has 27 dimensions ( (4+1+4)*3). This is updated in the following files -
Now we have 4 class, so we would need to change it's architecture.
This is updated in the config file - ./YoloV3/cfg/yolov3-hat.cfg.

            a. Search for filters=255 and change it to filters=27.
            b. Search for classes=80 and update it to classes=4.

            c. keep the following parameters -
                    burn_in=100
                    max_batches = 5000
                    policy=steps
                    steps=4000,4500



5. Now we are good to train the model -
    update the template colab shared by TSAI and change the train command to following.

        !python train.py --data data/YoloV3_Dataset/train.data --batch 16 --cache --cfg cfg/yolov3-hat.cfg --epochs 200 --weights='weights/last.pt'


6. Training for 300 epochs is a challenge.. so its is a good idea to save the weights and continue the run by using last.pt weights file. This automatically continues the epoch from previous run.

7. REsults:
    Create a set of images or video images and store it in the following location  -
                    data\samples

    Run the following command to generate annotated output images generated from images in samples folder.

        !python detect.py --conf-thres 0.3 --output train_out

    The results are stored in following directory

                data\train_out\

![image](https://github.com/vvshankar78/DeepLearning/blob/master/Extensive%20VisionAI%20(EVA5)/13_yolo_v2/train_out%20(1)%20-%20images/img-15.jpg)

8. The youtube video is downloaded and the video is converted into frames using ffpeg tool (https://en.wikibooks.org/wiki/FFMPEG_An_Intermediate_Guide/image_sequence).


            Convert video to images
            ffmpeg -i "sample.avi" frames/img%3d.jpg

            Convert images to video
            ffmpeg -framerate 24 -i ./train_out/img%3d.jpg -r 24 -y ./out-frames/out_video.mp4

            Location - C:\python\video-converter

The output of the youtube video annotated is shown in the following link -

https://www.youtube.com/watch?v=oLSnbfJb6x8&feature=youtu.be
