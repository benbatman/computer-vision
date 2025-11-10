#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <regex>
#include <unordered_map>
#include <cmath>

const float CONFIDENCE_THRESHOLD = 0.4f;
const float LOW_CONFIDENCE_THRESHOLD = 0.2f;
bool SHOW_LOW_CONFIDENCE = true;
const float BOX_TO_IMAGE_RATIO_THRESHOLD = 0.45f;
const float NMS_THRESHOLD = 0.4f;
const int LINE_WIDTH = 5;

// colors
cv::Scalar LOW_CONFIDENCE_COLOR = cv::Scalar(0, 0, 255); // Red
cv::Scalar DEFAULT_COLOR = cv::Scalar(0, 255, 0);        // Green

// Struct for detection info
struct Detection
{
    int class_id;
    std::string class_name;
    float score;
    cv::Rect box; // bbox
};

static std::unordered_map<int, std::string>
load_tf_label_map(const std::string &path_to_pbtxt)
{
    std::unordered_map<int, std::string> id2name;
    std::ifstream file(path_to_pbtxt);
    if (!file.is_open())
    {
        std::cerr << "Could not open the label map file: " << path_to_pbtxt << std::endl;
        return id2name;
    }
    std::string line;
    std::string name;
    int id = -1;
    std::regex re_name(R"(display_name:\s*\"([^\"]+)\")");
    std::regex re_id(R"(id:\s*([0-9]+))");
    while (std::getline(file, line))
    {
        std::smatch match;
        if (std::regex_search(line, match, re_id))
        {
            id = std::stoi(match[1]);
        }
        else if (std::regex_search(line, match, re_name))
        {
            name = match[1];
        }
        if (id != -1 && !name.empty())
        {
            id2name[id] = name;
            id = -1;
            name.clear();
        }
    }
    return id2name;
}

// SSD Mobilenet v3 TF params: 320x320, mean=127.5, scale=1/127.5, swapRB
cv::dnn::Net load_tf_model(const std::string &modelPath, const std::string &configPath)
{
    cv::dnn::Net net = cv::dnn::readNetFromTensorflow(modelPath, configPath);
    if (net.empty())
    {
        std::cerr << "Failed to load the model from: " << modelPath << " and " << configPath << std::endl;
    }
    return net;
}

std::pair<cv::Mat, std::vector<Detection>> process_frame(
    const cv::Mat &frame,
    cv::dnn::Net &cvNet,
    const std::unordered_map<int, std::string> &labels)
{
    std::vector<Detection> valid_detections{};
    if (frame.empty())
    {
        std::cerr << "Input frame is empty." << std::endl;
        return {frame, valid_detections};
    }

    int rows = frame.rows;
    int cols = frame.cols;

    cv::Mat blob = cv::dnn::blobFromImage(
        frame, 1.0 / 127.5, cv::Size(320, 320), cv::Scalar(127.5, 127.5, 127.5), true, false);

    std::cout << "Original Image Size: " << cols << "x" << rows << "\n";
    std::cout << "Blob Size: " << blob.size[3] << "x" << blob.size[2] << "\n";

    cvNet.setInput(blob);
    cv::Mat detectionMat = cvNet.forward();

    const float *data = (float *)detectionMat.ptr<float>(0, 0);
    int numDetections = detectionMat.size[2];

    for (int i = 0; i < numDetections; ++i)
    {
        int baseIdx = i * 7;
        int class_id = static_cast<int>(data[baseIdx + 1]);
        float score = data[baseIdx + 2];
        // .at() throws an exception if key not found
        auto it = labels.find(class_id);
        std::string class_name = (it != labels.end()) ? it->second : "Unknown";

        float left = data[baseIdx + 3] * cols;
        float top = data[baseIdx + 4] * rows;
        float right = data[baseIdx + 5] * cols;
        float bottom = data[baseIdx + 6] * rows;

        float box_width = right - left;
        float box_height = bottom - top;
        float box_area = box_width * box_height;
        float frame_area = static_cast<float>(cols * rows);

        // Large boxes probably false positives or background detection
        if (box_area > BOX_TO_IMAGE_RATIO_THRESHOLD * frame_area)
        {
            std::cout << "Skipping detection " << i << " due to large box-to-image ratio." << std::endl;
            continue;
        }

        Detection det;
        det.class_id = class_id;
        det.class_name = class_name;
        det.score = score;
        det.box = cv::Rect(cv::Point(static_cast<int>(left), static_cast<int>(top)),
                           cv::Point(static_cast<int>(right), static_cast<int>(bottom)));

        valid_detections.push_back(det);

        std::cout << "Detected class ID: " << class_id << " | Score: " << score << " | Name: " << class_name << "\n";
    }

    std::cout << "Total valid detections: " << valid_detections.size() << std::endl;
    return {frame, valid_detections};
}

void draw_detection(cv::Mat &img,
                    const Detection &detection,
                    const std::vector<int> &low_confidence_ids)
{
    int class_id = detection.class_id;
    const std::string class_name = detection.class_name;
    float score = detection.score;
    const cv::Rect box = detection.box;

    if (score > CONFIDENCE_THRESHOLD && class_id != 0)
    {
        cv::rectangle(img, box,
                      DEFAULT_COLOR, LINE_WIDTH);
        // Cant use string_view here as std::to_string creates a temporary string
        std::string text = class_name + ": " + std::to_string(int(std::round(score * 100))) + "%";
        cv::putText(img, text, cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    DEFAULT_COLOR, 2);
    }

    else if (
        score > LOW_CONFIDENCE_THRESHOLD &&
        score <= CONFIDENCE_THRESHOLD &&
        SHOW_LOW_CONFIDENCE &&
        (std::find(low_confidence_ids.begin(), low_confidence_ids.end(), class_id) != low_confidence_ids.end()))
    {
        cv::rectangle(img,
                      box,
                      LOW_CONFIDENCE_COLOR, LINE_WIDTH);
        std::string text = class_name + ": " + std::to_string(int(std::round(score * 100))) + "%";
        cv::putText(img, text, cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    LOW_CONFIDENCE_COLOR, 2);
    }
}

void non_maximum_suppression(cv::Mat &detectionMat,
                             const std::vector<Detection> &valid_detections,
                             const std::vector<int> &low_confidence_ids = std::vector<int>())
{
    if (valid_detections.empty())
    {
        std::cerr << "No valid detections provided." << std::endl;
        return;
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    for (const auto &det : valid_detections)
    {
        boxes.push_back(det.box);
        scores.push_back(det.score);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

    for (int idx : indices)
    {
        draw_detection(detectionMat, valid_detections[idx], low_confidence_ids);
    }
}

void save_and_show_image(const cv::Mat &img, const std::string &output_path = "output.jpg")
{
    cv::imwrite(output_path, img);
    cv::imshow("Detections", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main(int argc, char **argv)
{
    if (argc < 7)
    {
        std::cerr << "usage: mobilenet_video <video> <model.pb> <model.pbtxt> <labels.pbtxt> <frame_interval> <output_video>\n";
        std::cerr << "  frame_interval: process every Nth frame (e.g., 5 = process every 5th frame)\n";
        return 1;
    }

    // Parse frame interval
    int frameInterval = std::stoi(argv[5]);
    if (frameInterval < 1)
    {
        std::cerr << "Error: frame_interval must be >= 1\n";
        return 1;
    }

    // Open video file
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened())
    {
        std::cerr << "Could not open the video: " << argv[1] << std::endl;
        return 2;
    }

    // Get video properties
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // Setup video writer
    std::string outputVideoPath = argv[6];

    // MP4V codec
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::VideoWriter videoWriter(outputVideoPath, fourcc, fps, cv::Size(width, height));

    if (!videoWriter.isOpened())
    {
        std::cerr << "Could not open the output video file: " << outputVideoPath << std::endl;
        return 3;
    }

    std::cout << "Video Info:\n";
    std::cout << "  Resolution: " << width << "x" << height << "\n";
    std::cout << "  FPS: " << fps << "\n";
    std::cout << "  Total Frames: " << totalFrames << "\n";
    std::cout << "  Processing every " << frameInterval << " frame(s)\n";
    std::cout << std::string(60, '=') << "\n\n";

    // Load class names
    auto classes = load_tf_label_map(argv[4]);

    // Load TensorFlow model
    std::cout << "Loading model..." << "\n";
    cv::dnn::Net model = load_tf_model(argv[2], argv[3]);

    cv::Mat frame;
    int frameNumber = 0;
    int processedFrames = 0;

    // Cache to store processed frames
    std::unordered_map<int, cv::Mat> processedFrameCache;

    while (cap.read(frame))
    {
        // frame gets reused in video capture loop, so clone it
        cv::Mat outputFrame = frame.clone();

        // Process every Nth frame
        if (frameNumber % frameInterval == 0)
        {
            auto [processedFrame, validDetections] = process_frame(outputFrame, model, classes);

            // Output results
            double timestamp = frameNumber / fps;
            std::cout << "Frame " << frameNumber << " (t=" << std::fixed << std::setprecision(2)
                      << timestamp << "s) - " << validDetections.size() << " detections:\n";

            if (!validDetections.empty())
            {
                non_maximum_suppression(outputFrame, validDetections);
            }

            // Cache the processed frame
            processedFrameCache[frameNumber] = outputFrame.clone();

            std::cout << "Frame " << frameNumber << " (t=" << std::fixed << std::setprecision(2)
                      << timestamp << "s) - " << validDetections.size() << " detections:\n";

            // output for debugging purposes
            if (validDetections.empty())
            {
                std::cout << "  No objects detected\n";
            }
            else
            {
                for (const auto &det : validDetections)
                {
                    std::cout << "  Class: " << det.class_name
                              << " | Score: " << std::round(det.score * 100) << "%"
                              << " | Box: [" << det.box.x << ", " << det.box.y
                              << ", " << det.box.width << ", " << det.box.height << "]\n";
                }
            }
            std::cout << "\n";

            processedFrames++;
        }
        // For frames btn processed frames, use most recent processed frame detections
        else
        {
            int lastProcessedFrame = (frameNumber / frameInterval) * frameInterval;
            if (processedFrameCache.find(lastProcessedFrame) != processedFrameCache.end())
            {
                outputFrame = processedFrameCache[lastProcessedFrame].clone();
            }
        }

        // Write the frame to output video
        videoWriter.write(outputFrame);

        // Progress
        if (frameNumber % 100 == 0)
        {
            double progress = (double)frameNumber / totalFrames * 100.0;
            std::cout << "Progress: " << std::fixed << std::setprecision(1) << progress << "% ("
                      << frameNumber << "/" << totalFrames << " frames)\n";
        }

        frameNumber++;
    }

    std::cout << std::string(60, '=') << "\n";
    std::cout << "Processing complete!\n";
    std::cout << "  Total frames: " << frameNumber << "\n";
    std::cout << "  Processed frames: " << processedFrames << "\n";

    cap.release();
    videoWriter.release();
    return 0;
}
