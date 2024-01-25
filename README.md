# FDSMinds-Pixta-hackathon-2024

This brach uses many models, each model do a specific task. Facedetector use yolov8n model to detect face in input image then crop bboxes areas then send the cropped image to classify models. 6 different models with backbone resnet will classify the detected dace by age, race, emotion, gender, skintone and masked.

## How to run

Install dependencies
```bash
pip install -r requirements.txt
```

Download model that I will send via drive and the directory, download the public_text.zip then unzip in the same directory.
```
    models/
    |-------classify/
            |-------classify_model/
                    |-------age/
                            |-------best.pt
                            |-------last.pt
                    |-------gender/
                            |-------best.pt
                            |-------last.pt
            .
            .
            .
    |-------detect/
            |-------yolov8n/
                    |-------lastest/
                            |-------best.pt
                            |-------last.pt
            .
            .
    public_test/
    |-------img1.jpg
    |-------img2.jpg
    |-------img3.jpg
            .
            .
            .
  ```

To get the answer.csv, run the **convert2answer.ipynb** notebook

