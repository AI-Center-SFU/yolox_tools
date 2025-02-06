import os
import cv2
import torch
import numpy as np

# Import necessary modules from YOLOX
from yolox.exp import get_exp
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess, vis
from yolox.data.datasets import COCO_CLASSES


class YOLOXInferencer:
    """
    YOLOX Inferencer for performing object detection inference using YOLOX models.

    This class loads an experiment configuration, builds a YOLOX model, loads its weights from a
    checkpoint, and provides methods to perform inference and visualize the detection results.

    Parameters
    ----------
    exp_file : str
        Path to the experiment configuration file (e.g. "YOLOX/exps/default/yolox_m.py").
    model_name : str
        Model name as specified in the configuration (e.g. "yolox_m").
    ckpt_path : str
        Path to the checkpoint file containing the weights.
    num_classes : int, optional
        New number of classes if you need to change it, by default None. If None, the number of classes
        will be taken from the configuration.
    test_size : tuple, optional
        Size of the test image (width, height), by default (640, 640).
    conf_thre : float, optional
        Confidence threshold for detection, by default 0.3.
    nmsthre : float, optional
        Threshold for non-max suppression (NMS), by default 0.45.
    device : str, optional
        Device for inference ("cpu" or "cuda"), by default "cuda".
    fp16 : bool, optional
        Whether to use FP16 precision, by default False.
    decoder : callable, optional
        Decoder function for model outputs (if used), by default None.
    legacy : bool, optional
        Whether to use legacy mode in pre-processing, by default False.

    Examples
    --------
    >>> # Example usage:
    >>> exp_file = "YOLOX/exps/default/yolox_m.py"  # Path to experiment config file
    >>> model_name = "yolox_m"  # Name of the model as specified in the config
    >>> ckpt_path = "path/to/best_ckpt.pth"  # Path to the checkpoint file
    >>> # If you do not wish to change the number of classes, set num_classes to None.
    >>> inferencer = YOLOXInferencer(exp_file=exp_file, model_name=model_name, ckpt_path=ckpt_path)
    >>>
    >>> # Perform inference on an image (provide either the image path or a numpy array in BGR format)
    >>> image_path = "path/to/image.jpg"
    >>> predictions = inferencer.predict(image_path)
    >>> print("Predictions:", predictions)
    >>>
    >>> # Visualize the detections using the visualize method.
    >>> # For example, in a Jupyter Notebook, you can display the result using matplotlib:
    >>> import matplotlib.pyplot as plt
    >>> img = cv2.imread(image_path)
    >>> img_with_dets = inferencer.visualize(predictions, img)
    >>> # Convert BGR to RGB for displaying with matplotlib
    >>> img_rgb = cv2.cvtColor(img_with_dets, cv2.COLOR_BGR2RGB)
    >>> plt.figure(figsize=(10, 10))
    >>> plt.imshow(img_rgb)
    >>> plt.axis("off")
    >>> plt.show()
    """

    def __init__(
        self,
        exp_file: str,
        model_name: str,
        ckpt_path: str,
        num_classes: int = None,
        test_size: tuple = (640, 640),
        conf_thre: float = 0.3,
        nmsthre: float = 0.45,
        device: str = "cuda",
        fp16: bool = False,
        decoder=None,
        legacy: bool = False,
    ):
        self.device = device
        self.fp16 = fp16

        # 1. Load the experiment configuration
        exp = get_exp(exp_file, model_name)

        # 2. Update the number of classes and other inference parameters if needed
        if num_classes is not None:
            exp.num_classes = num_classes
        exp.test_size = test_size
        exp.test_conf = conf_thre
        exp.nmsthre = nmsthre

        self.exp = exp

        # 3. Create the model according to the configuration
        model = exp.get_model()

        # 4. Load the checkpoint
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])

        # Transfer the model to the specified device and set it to evaluation mode
        model.to(device)
        if fp16:
            model.half()
        model.eval()

        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.test_size = test_size
        self.conf_thre = conf_thre
        self.nmsthre = nmsthre

        # Initialize pre-processing (transform)
        self.preproc = ValTransform(legacy=legacy)

    def predict(self, img):
        """
        Performs inference on an image using the model.

        Parameters
        ----------
        img : str or numpy.ndarray
            A path to an image or an image in numpy array format (BGR).

        Returns
        -------
        numpy.ndarray
            Array of detected objects. Each detected object has the format:
            [x1, y1, x2, y2, object_confidence, class_confidence, class_id].
        """
        # Load the image if a path is provided
        if isinstance(img, str):
            if not os.path.exists(img):
                raise FileNotFoundError(f"Image not found: {img}")
            img_raw = cv2.imread(img)
            if img_raw is None:
                raise ValueError(f"Failed to load image: {img}")
        else:
            img_raw = img

        height, width = img_raw.shape[:2]
        # Calculate the scaling ratio to maintain the aspect ratio
        ratio = min(self.test_size[0] / height, self.test_size[1] / width)

        # Pre-processing: resizing, normalization, etc.
        img_preproc, _ = self.preproc(img_raw, None, self.test_size)
        img_tensor = torch.from_numpy(img_preproc).unsqueeze(0).float()
        if self.device != "cpu":
            img_tensor = img_tensor.to(self.device)
            if self.fp16:
                img_tensor = img_tensor.half()

        with torch.no_grad():
            outputs = self.model(img_tensor)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs,
                self.num_classes,
                self.conf_thre,
                self.nmsthre,
                class_agnostic=True
            )

        if outputs is None or len(outputs) == 0:
            return np.array([])

        predictions = outputs[0].cpu().numpy()
        # Scale the coordinates back to the original image size
        predictions[:, :4] /= ratio
        return predictions

    def visualize(self, predictions, img, cls_conf: float = 0.35):
        """
        Visualizes the detections on the image.

        Parameters
        ----------
        predictions : numpy.ndarray
            Array of detected objects (result from the `predict` method).
        img : numpy.ndarray
            The original image in numpy array format (BGR).
        cls_conf : float, optional
            Confidence threshold for visualization, by default 0.35.

        Returns
        -------
        numpy.ndarray
            The image with detections overlaid.
        """
        if predictions.size == 0:
            return img

        return vis(
            img,
            predictions[:, :4],
            predictions[:, 4] * predictions[:, 5],
            predictions[:, 6],
            cls_conf,
            COCO_CLASSES,
        )
