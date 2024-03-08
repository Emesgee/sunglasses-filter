#include <iostream> 
#include <string> 
#include <vector> 
#include <stdlib.h> 
#include <opencv2/core.hpp> 
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp> 
#include <opencv2/dnn.hpp> // Deep Neural Network (DNN) module of OpenCV

// defining namespaces
using namespace cv;
using namespace std;
using namespace cv::dnn;

// global vairables (Might delete this one) 
Mat faceROI;
Mat sunglass;
Mat outputImage;

const size_t inWidth = 300; // input image width
const size_t inHeight = 300; // input image Height
const double inScaleFactor = 1.0; // scale factor for input image 
const double confidenceThreshold = .999; // confidence threshold for face detection 
const cv::Scalar meanVal(104.0, 177.0, 123.0); // mean pixels value for normalization


// path to pre-trained dnn model 
string MODEL_PATH = "../../data/models/";

// tensorflow config file
const std::string tensorflowConfigFile = MODEL_PATH + "opencv_face_detector.pbtxt";
// tensorflow weight file
const std::string tensorflowWeightFile = MODEL_PATH + "opencv_face_detector_uint8.pb";

// detect faces function with OpenCV DNN
void detectFaceOpenCVDNN(Net net, Mat& frameOpenCVDNN)
{
    //  get frame height and width
    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;

    // preprocess input Image 
    cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight),
        meanVal, false, false);
    // set input for neural network
    net.setInput(inputBlob, "data");

    // forward pass to get detection
    cv::Mat detection = net.forward("detection_out");
    // get detection matrix
    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    // collect the coordinates of all detected faces
    vector<Rect> facesRects;

    // iterate over each detected face
    for (int i = 0; i < detectionMat.rows; i++)
    {
        // get confidence of the detection
        float confidence = detectionMat.at<float>(i, 2);

        // check if confidence is above threshold
        if (confidence > confidenceThreshold)
        {
            // calculate coordinates if above threshold
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

            // append detected face rectangle to the list
            facesRects.push_back(Rect(x1, y1, x2 - x1, y2 - y1));

            // Inside the loop where faces are detected
            cout << "Detected Face " << i + 1 << " - (x1,y1): (" << x1 << "," << y1 << "), (x2,y2): (" << x2 << "," << y2 << ")" << endl;
        }
    }
    for (const auto& rect : facesRects)
    {
        // Draw green rectangle around detected face
        //rectangle(frameOpenCVDNN, rect, Scalar(0, 255, 0), 2); // green rectangle

        // Calculate dimensions for the new red rectangle to locate the eye region
        int thresh = 3;
        int NewHeight = rect.height / thresh; // Dividing the height by four

        Point topLeft(rect.x - 6, static_cast<int>(rect.y + NewHeight * 0.8));

        // Define the new rectangle
        Rect eyeRegionRect(topLeft, Size((rect.width + 8), NewHeight));

        // Draw red rectangle for the eye region
        //rectangle(frameOpenCVDNN, eyeRegionRect, Scalar(0, 0, 255), 2); // red rectangle


        //-------------------------------------------------------------------------------------------------------------------------------


        // extracting face(eye) region as faceROI
        Mat faceROI = frameOpenCVDNN(eyeRegionRect).clone();

        // Load the sunglass image and reize
        Mat sunglass = imread("sunglass.png", IMREAD_UNCHANGED);
        resize(sunglass, sunglass, faceROI.size());

        blur(sunglass, sunglass, Size(2, 2));

 
        // extract sunglass alpha channel 
        vector<Mat>channels;
        split(sunglass, channels);
        Mat alphaChannel = channels[3];

        // Load the desert image and reize
        Mat desert = imread("desert-or-ocean.jpg"); // or Mat desert = imread("desert.jpg");
        resize(desert, desert, faceROI.size());
        cvtColor(desert, desert, COLOR_BGR2HSV);

        //extract desert alpha channel 
        vector<Mat>desertChannels;
        split(desert, desertChannels);
        desert = desertChannels[2];

        //blur(desert, desert, Size(2, 2));

        // convert Mat to float datatypes
        sunglass.convertTo(sunglass, CV_32FC3);
        faceROI.convertTo(faceROI, CV_32FC3);
        desert.convertTo(desert, CV_32FC3);

        // normalize the alpha mask to keep intensity between 0 and 1
        alphaChannel.convertTo(alphaChannel, CV_32FC3, 1.0 / 255);

        // Convert alphaChannel to a 4-channel image
        Mat alphaChannel4Channels;
        cvtColor(alphaChannel, alphaChannel4Channels, COLOR_GRAY2BGRA);

        // Adjust the alpha channel values of the sunglass to make it more transparent
        double transparencyFactor = 0.8; // Adjust this value to control transparency
        alphaChannel4Channels *= transparencyFactor;

        // Convert desert to BGRA
        cvtColor(desert, desert, COLOR_BGR2BGRA);
        
        // Multiply the background with (1 - alpha)
        multiply(Scalar::all(1.0) - alphaChannel4Channels, desert, desert);

        // extracting desert outputimage 
        Mat desertOuputImage;
        add(sunglass, desert, desertOuputImage);

        // converting desert outputimage to BGR
        Mat desertOuputImageBGR;
        cvtColor(desertOuputImage, desertOuputImageBGR, COLOR_BGRA2BGR);

        // converting desert outputimage to BGRA
        Mat desertOuputImageBGRA;
        cvtColor(desertOuputImageBGR, desertOuputImageBGRA, COLOR_BGR2BGRA);
        sunglass = desertOuputImageBGRA.clone();

        
        // ---------------------------------------------------------------------------

       //Multiply the sunglass with the alpha matte
        multiply(alphaChannel4Channels, sunglass, sunglass);
  
        // Convert faceROI to BGRA
        cvtColor(faceROI, faceROI, COLOR_BGR2BGRA);

        // Multiply the background with (1 - alpha)
        multiply(Scalar::all(1.0) - alphaChannel4Channels, faceROI, faceROI);

        // storage for output image
        Mat outputImage = Mat::zeros(sunglass.size(), sunglass.type()); // move down maybe
        // Combine the sunglass region with the background region to form the final output
        add(sunglass, faceROI, outputImage);

        // Convert the output image to 8-bit unsigned integer (if necessary)
        outputImage.convertTo(outputImage, CV_8UC4);

        // converting output image to BGR
        cvtColor(outputImage, outputImage, COLOR_BGRA2BGR);

        // Copy the sunglasses onto the face image at the specified location
        outputImage.copyTo(frameOpenCVDNN(eyeRegionRect));
    }
}

////VIDEO int main() BELOW!

int main() {
    cout << "Code Proudly by: Mohammad Ghadban :) \n Cheers!" << endl;

    // Load pre-trained model
    Net net = cv::dnn::readNetFromTensorflow(tensorflowWeightFile, tensorflowConfigFile);

    // Load the sunglass image
    Mat musk = imread("musk.jpg", 1);


    // Check if the sunglass image is loaded successfully
    if (musk.empty()) {
        cerr << "Failed to load sunglass image!" << endl;
        return -1;
    }
    detectFaceOpenCVDNN(net, musk);
    imshow("musk", musk);

    // wait for a key press
    waitKey(0);

    // close the window
    destroyAllWindows();

    return 0;
}
//int main() {  
   // // Create a VideoCapture object to access the camera
   // VideoCapture cap(0); // 0 for the default camera, you can change it if you have multiple cameras
   //
   // // Check if the camera is opened successfully
   // if (!cap.isOpened()) {
   //     cout << "Error: Unable to open the camera." << endl;
   //     return -1;
   // }
   //
   // // Load pre-trained model
   // Net net = cv::dnn::readNetFromTensorflow(tensorflowWeightFile, tensorflowConfigFile);
   //
   // // Create a window to display the camera feed
   // namedWindow("Camera Feed", WINDOW_NORMAL);
   //
   // // Main loop to continuously read frames from the camera
   // while (true) {
   //     // Read a frame from the camera
   //     Mat frame;
   //     cap.read(frame);
   //
   //     // Check if the frame is empty
   //     if (frame.empty()) {
   //         cout << "Error: Unable to read frame from the camera." << endl;
   //         break;
   //     }
   //     
   //     // Display the frame in the window
   //     detectFaceOpenCVDNN(net, frame);
   //     imshow("Camera Feed", frame);
   //
   //
   //     // Check for key press to exit the loop
   //     if (waitKey(1) == 27) // ASCII code for 'ESC'
   //         break;
   // }
   //
   // // Release the VideoCapture object and close the window
   // cap.release();
   //
   // // close the window
   // destroyAllWindows();
   //
   // return 0;
//}

