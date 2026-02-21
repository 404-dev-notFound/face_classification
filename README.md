# Face Classifier and Sorter

This project uses the `face_recognition` library to identify individuals in images and sort those images into person-specific folders.

## Setup

1.  **System Dependencies**: `dlib` requires C++ compilation. Install these first:
    ```bash
    sudo apt update
    sudo apt install -y cmake python3-dev libopenblas-dev liblapack-dev libx11-dev
    ```
    *Note: For GPU acceleration (CNN model), ensure you have NVIDIA CUDA and cuDNN installed.*

2.  **Python Requirements**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Known Faces**: Create a folder (e.g., `known_people`) and place images of people you want to recognize. 
    *   Example: `known_people/obama.jpg` or a folder `known_people/obama/` containing multiple images.
3.  **Unknown Faces**: Place the images you want to classify in another folder (e.g., `input_images`).

## How to Run

Run the script using Python:

```bash
python classify_faces.py --known-dir path/to/known_people --unknown-dir path/to/input_images
```

### Options

*   `--known-dir`: (Required) Path to known images.
*   `--unknown-dir`: (Required) Path to images to be classified.
*   `--output-dir`: (Default: `output`) Where sorted folders will be created.
*   `--use-cnn`: (Flag) Use the CNN model for face detection. This is more accurate but slower on CPU. If you have a GPU with CUDA support, this is much faster.
*   `--tolerance`: (Default: `0.6`) How strict the matching should be. Lower is stricter.

## Example

```bash
python classify_faces.py --known-dir ./known --unknown-dir ./unknown --use-cnn
```

This will create an `output` folder with subfolders named after the people found in the images. Any images where no face was found or no match was made will be placed in `output/unknown`.


## Thanks

* Thanks to [ageitgey](https://github.com/ageitgey/face_recognition) for making face_recognition project. 