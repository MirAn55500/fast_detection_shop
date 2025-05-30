# Dataset Preparation Guide for Custom YOLO Training

## Required Dataset Format

1. **Directory Structure**:
   ```
   custom_dataset/
   ├── images/
   │   ├── img1.jpg
   │   ├── img2.jpg
   │   └── ...
   └── labels/
       ├── img1.txt
       ├── img2.txt
       └── ...
   ```

2. **Label Format**:
   Each `.txt` file corresponds to an image and contains one line per object in the following format:
   ```
   <class_id> <x_center> <y_center> <width> <height>
   ```
   Where:
   - `class_id`: Integer class ID (0 for person, 1 for pallet, 2 for dome)
   - `x_center`, `y_center`: Normalized center coordinates of the bounding box (0.0 to 1.0)
   - `width`, `height`: Normalized width and height of the bounding box (0.0 to 1.0)

   Example:
   ```
   0 0.507 0.626 0.945 0.747
   1 0.253 0.154 0.235 0.124
   ```

3. **Image Requirements**:
   - Supported formats: JPG, JPEG, PNG
   - Recommended resolution: at least 416×416 pixels
   - Varied lighting conditions, angles, and backgrounds for better generalization

## Annotation Tools

You can use the following tools for annotation:

1. **LabelImg**: https://github.com/tzutalin/labelImg
   - Set format to YOLO
   - Define classes in a `classes.txt` file

2. **CVAT**: https://github.com/opencv/cvat
   - Web-based annotation tool
   - Supports YOLO format export

3. **Roboflow**: https://roboflow.com
   - Online platform with free tier
   - Supports annotation and export to YOLO format

## Best Practices

1. **Dataset Size**:
   - Minimum 50 images per class
   - Recommended: 500+ images per class for good performance

2. **Data Augmentation**:
   - The training script includes basic augmentation
   - For more advanced augmentation, consider using Roboflow or Albumentations

3. **Class Balance**:
   - Try to have a similar number of images for each class
   - Ensure each class has sufficient representation

4. **Annotation Quality**:
   - Be consistent in bounding box placement
   - Include the entire object in the bounding box
   - For overlapping objects, annotate each one separately 