# RetinaFaceNet

Repository is intended to wrap the code from 
https://github.com/biubug6/Pytorch_Retinaface to be easily
integrated with any existing codebase.

## Weights
* Original source: [weights](https://drive.google.com/drive/folders/1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1)
* Mirror: [weights](https://drive.google.com/open?id=1Cfq2f9jhFCN-thB_pM40-Duzcullufs1)


## Implementation details
The implementation only covers RetinaFaceNet model based on ResNet-50.


## Installation
```bash
/path/to/repository_root$ pip install .
```

## API overview

### Importing
In order to import the inference wrapper (as well as other important
library elements) one need to create the following import statement:
```python
from retina_face_net import RetinaFaceNet, \
    RetinaFaceNetPrediction, BoundingBox, Point
```

### Initialization
To initialize the **RetinaFaceNet** object the easiest way is to use:

```python
from retina_face_net import RetinaFaceNet

retina_face_net = RetinaFaceNet.initialize(
    weights_path="/path/to/pre_fetched/weights",
    use_gpu=False,
    confidence_threshold=0.3,
    top_k=20,
    nms_threshold=0.4
)
```

where:
* **confidence_threshold** points minimum confidence of valid predictions
* **top_k** points number of predictions that will be taken into account 
  as an input to nms algorithm
* **nms_threshold** is a IoU threshold used in Non-Maximum Suppression algorithm

### Usage
```python
from typing import List

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from retina_face_net import RetinaFaceNet, RetinaFaceNetPrediction


def draw_predictions(image: np.ndarray,
                     predictions: List[RetinaFaceNetPrediction]
                     ) -> np.ndarray:
    image = image.copy()
    for prediction in predictions:
        bbox = prediction.bbox
        cv.rectangle(
            img=image,
            pt1=bbox.left_top.compact_form,
            pt2=bbox.right_bottom.compact_form,
            color=(0, 255, 0),
            thickness=5
        )
    return image

retina_face_net = RetinaFaceNet.initialize(
    weights_path="/path/to/pre_fetched/weights"
)

image = cv.imread("/path/to/input_image.jpg")
inference_results = retina_face_net.infer(image=image)
output = draw_predictions(
    image=image,
    predictions=inference_results
)


plt.imshow(output[:,:,::-1])
```