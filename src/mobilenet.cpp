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

std::pair<cv::Mat, std::vector<Detection>> process_image(
    const std::string &image_path,
    cv::dnn::Net &cvNet,
    const std::unordered_map<int, std::string> &labels)
{
    std::vector<Detection> valid_detections{};
    cv::Mat img = cv::imread(image_path);
    if (img.empty())
    {
        std::cerr << "Could not read the image: " << image_path << std::endl;
        return {img, valid_detections};
    }

    int rows = img.rows;
    int cols = img.cols;

    cv::Mat blob = cv::dnn::blobFromImage(
        img, 1.0 / 127.5, cv::Size(320, 320), cv::Scalar(127.5, 127.5, 127.5), true, false);

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
        float img_area = static_cast<float>(cols * rows);

        // Large boxes probably false positives or background detection
        if (box_area > BOX_TO_IMAGE_RATIO_THRESHOLD * img_area)
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
    return {img, valid_detections};
}

void draw_detection(cv::Mat &img,
                    const Detection &detection,
                    const std::vector<int> &low_confidence_ids)
{
    int class_id = detection.class_id;
    std::string class_name = detection.class_name;
    float score = detection.score;
    cv::Rect box = detection.box;
    int left = box.x;
    int top = box.y;
    int right = box.x + box.width;
    int bottom = box.y + box.height;

    if (score > CONFIDENCE_THRESHOLD && class_id != 0)
    {
        cv::rectangle(img,
                      cv::Rect(left, top, right - left, bottom - top),
                      DEFAULT_COLOR, LINE_WIDTH);
        // Cant use string_view here as std::to_string creates a temporary string
        std::string text = class_name + ": " + std::to_string(int(std::round(score * 100))) + "%";
        cv::putText(img, text, cv::Point(left, top - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    DEFAULT_COLOR, 2);
    }

    else if (
        score > LOW_CONFIDENCE_THRESHOLD &&
        score <= CONFIDENCE_THRESHOLD &&
        SHOW_LOW_CONFIDENCE &&
        (std::find(low_confidence_ids.begin(), low_confidence_ids.end(), class_id) != low_confidence_ids.end()))
    {
        cv::rectangle(img,
                      cv::Rect(left, top, right - left, bottom - top),
                      LOW_CONFIDENCE_COLOR, LINE_WIDTH);
        std::string text = class_name + ": " + std::to_string(int(std::round(score * 100))) + "%";
        cv::putText(img, text, cv::Point(left, top - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5,
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
    if (argc < 5)
    {
        std::cerr << "usage: mobilenet_image <image> <model.pb> <model.pbtxt> <label_map.pbtxt>\n";
        return 1;
    }

    std::cout << "Loading label map..." << std::endl;
    auto labels = load_tf_label_map(argv[4]);

    // Load TensorFlow model
    std::cout << "Loading model..." << std::endl;
    cv::dnn::Net model = load_tf_model(argv[2], argv[3]);

    // Load and process image
    std::string image_path = argv[1];
    auto [output_img, valid_detections] = process_image(image_path, model, labels);

    // Perform non-maximum suppression
    non_maximum_suppression(output_img, valid_detections);

    // Save and show the output image
    save_and_show_image(output_img);

    return 0;
}
