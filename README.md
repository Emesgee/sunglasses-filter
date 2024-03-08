OpenCV Face Detection with Sunglasses Overlay
This repository contains code for detecting faces in images and overlaying sunglasses on the detected faces using OpenCV's Deep Neural Network (DNN) module. The program detects faces in images using a pre-trained DNN model and overlays sunglasses on the detected faces.

Dependencies
OpenCV (4.x)
C++11 or higher
CMake (for building)
Setup
Clone the repository:
bash
Copy code
git clone https://github.com/Emesgee/sunglasses-filter.git
cd sunglasses-filter
Compile the code using CMake:
bash
Copy code
mkdir build
cd build
cmake ..
make
Ensure you have the necessary model and image files in the appropriate directories:

Pre-trained model files (opencv_face_detector.pbtxt and opencv_face_detector_uint8.pb) should be located in data/models/.
Sunglasses image file (sunglass.png) should be in the root directory.
Background image file (optional, desert-or-ocean.jpg or desert.jpg) should be in the root directory.
Usage
Run the compiled executable with an image file as input:

bash
Copy code
./face_detection_with_sunglasses <image_file>
Example
bash
Copy code
./face_detection_with_sunglasses input_image.jpg
This will display the input image with sunglasses overlay on detected faces.

Notes
This code is based on OpenCV's DNN module and requires OpenCV 4.x or higher.
Ensure that the pre-trained model files (opencv_face_detector.pbtxt and opencv_face_detector_uint8.pb) are properly configured in the data/models/ directory.
The sunglasses image (sunglass.png) should be appropriate for overlaying and should be placed in the root directory.
Optionally, you can use a background image (desert-or-ocean.jpg or desert.jpg) for the background behind the sunglasses. Place it in the root directory.
Credits
This code is proudly developed by Mohammad Ghadban.
