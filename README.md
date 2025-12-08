# Face Detection & Recognition from Video using MTCNN & ResNet

This project demonstrates a full pipeline for detecting faces from a video, extracting frames, building embeddings using Inception-Resnet-V1, and finally recognizing identities in new images.

## Requirements

pip install facenet-pytorch
pip install torchvision
pip install opencv-python
pip install matplotlib
pip install pillow
~/.cache/torch/checkpoints/20180402-114759-vggface2.pt
InceptionResnetV1(pretrained="vggface2")


## Project Objectives

Extract frames from a video

Detect faces using MTCNN

Compute facial feature embeddings using FaceNet

Build a mini face-database from known identities

Recognize faces in new images

Visualize bounding boxes and predicted names


| Library                   | Purpose                   |
| ------------------------- | ------------------------- |
| `OpenCV`                  | Extract frames from video |
| `MTCNN` (facenet-pytorch) | Detect faces + landmarks  |
| `InceptionResnetV1`       | Compute face embeddings   |
| `Torchvision`             | Dataset & image loading   |
| `PIL` / `Matplotlib`      | Visualization             |
| `PyTorch`                 | Core computation          |


project/
│── data_video/
│   ├── pat_morgan.mp4
│   └── extracted_frames/   <-- auto-generated frames
│
│── data_images/
│   ├── morgan/
│   └── patrick/
│
│── embedded.pt             <-- saved embeddings
│── multi_faces/
│── README.md
│── notebook.ipynb          <-- Your code


## Face Detection with MTCNN

mtcnn = MTCNN(device="cpu", keep_all=True, min_face_size=60)

### MTCNN returns:

Bounding boxes

Probabilities

Landmarks (eyes, nose, lips corners)

## Improvements Suggestions
Can be extended into:

✔ Live webcam face recognition
✔ Streamlit/Gtk GUI app
✔ Multiple identities
✔ Persist identification using SQLite
✔ Real-time video tracking

