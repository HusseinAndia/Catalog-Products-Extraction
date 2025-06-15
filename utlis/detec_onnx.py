import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from typing import List, Tuple
import cv2
import onnxruntime as rt


class Detection():
    """
    Detection model class is for handling ONNX inference and visualization.

        Attributes:
        input_image (str): Path to the input image file.
        classes (List[str]): Classes that can be detected by the model.

    """
    def __init__(self, img_path: str, classes: List[str]):
        self.img_path = img_path
        self.classes = classes

    def crop_save(self, boxes: List[List], class_ids:List[float], wanted_classes: List[int], save_path: str) -> None:
        """
        Draw bounding boxes and labels on the input image based on the detected objects.
    
        Args:
            boxes (List[float]): Detected bounding boxes coordinates [x1, y1, x2, y2].
            class_ids (int): Class IDs for the detected object.
            wanted_classes (List[int]): Classes that can be detected by the model.
            save_path (str): The save directory path.
        """
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
            
        cv_image = cv2.imread(self.img_path)
        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        for i in range(len(boxes)):
            if class_ids[i] in wanted_classes:
                x1, y1, x2, y2 = boxes[i]
                cropped_image = img[y1:y2, x1:x2]
                output_name = os.path.basename(self.img_path).split('.')[0]
                cv2.imwrite(f"{save_path}/{output_name}_{self.classes[int(class_ids[i])]}.jpg", cropped_image)

    def preprocess(self, new_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the input image before performing inference.

        This method reads the input image, converts its color space, normalizes pixel values 
        and prepares the image data for model input.

        Returns:
            (np.ndarray): Preprocessed image data ready for inference with shape (1, 3, height, width).
        """

        # Read the image and extract it's dimensions
        cv_image = cv2.imread(self.img_path)
        img_height,img_width = cv_image.shape[:2]

        # Converts the image's color space, resize it and reshape it's dimensions
        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img, new_size) 
        cv_input_tensor = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        cv_input_tensor = np.expand_dims(cv_input_tensor, axis=0)

        dims = np.array([img_height, img_width, input_height, input_width]) 

        return cv_input_tensor, dims
        

    def postprocess(self, dim: np.ndarray, output: List[np.ndarray])-> Tuple[List[List], List[float], List[float]]:
        """
        Perform post-processing on the model's output to extract and visualize detections.

        Args:
            dim (np.ndarray): The dimensions of the input image and the model input.
            output (List[np.ndarray]): The output arrays from the model.
        
        Returns:
            (Tuple[List, List, List]): The bounding box coordinates, scores and class_ids reulted from the detections model.
        """
        
        # Get the number of rows in the outputs array
        rows = output[0].shape[0]
        
        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []
        
        # Calculate the scaling factors for the bounding box coordinates
        gain = [dim[1] / dim[3], dim[0] / dim[2]]
        
        # Iterate over each row in the outputs array
        for i in range(rows):
            if (outputs[0][i][0] != 0) and (outputs[0][i][1] != 0):
            # Extract the class scores from the current row
                classes_scores = outputs[0][i][4]
                class_id = outputs[0][i][-1]
        
                # Extract the bounding box coordinates from the current row
                x1, y1, x2, y2 = outputs[0][i][0], outputs[0][i][1], outputs[0][i][2], outputs[0][i][3]
        
                # Calculate the scaled coordinates of the bounding box
                x1, x2 = x1 * gain[0], x2 * gain[0]
                y1, y2 = y1 * gain[1], y2 * gain[1]
                
                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(classes_scores)
                boxes.append([int(x1), int(y1), int(x2), int(y2)])

        return boxes, scores, class_ids
        

def draw_detections(self,boxes: List[list], scores: List[float], class_ids: List[int], view_img: bool=True, save: bool=False, save_path: str="") -> None:
        """
        Draw bounding boxes and labels on the input image based on the detected objects.
    
        Args:
            boxes (List[float]): Detected bounding boxes coordinates [x, y, width, height].
            scores (float): Confidence scores of the detection.
            class_ids (int): Class IDs for the detected object.
            view_img (bool): To show the result image after drawing the detected bounding boxes.
            save (bool): To save the result image.
            save_path (str): The path to the save directory.
        """
    
        cv_image = cv2.imread(self.img_path)
        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
        for i in range(len(boxes)):
            # Extract the coordinates of the bounding box
            x1, y1, x2, y2 = boxes[i]
           
            if len(self.classes) > 1:
                # Retrieve the color for the class ID
                color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
                color = color_palette[int(class_ids[i])]
            else:
                color = (0, 0, 255)
                
            # Draw the bounding box on the image
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
            # Create the label text with class name and score
            label = f"{self.classes[int(class_ids[i])]}: {scores[i]:.2f}"
        
            # Calculate the dimensions of the label text
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Calculate the position of the label text
            label_x = int(x1)
            label_y = int(y1 - 10) if y1 - 10 > label_height else int(y1 + 10)
            
            # Draw a filled rectangle as the background for the label text
            cv2.rectangle(
                img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
            )
        
            # Draw the label text on the image
            cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
        if view_img:
            cv2.imshow(img)
            cv2.waitKey(0)

        if save:
            cv2.imwrite(save_path, img)

    