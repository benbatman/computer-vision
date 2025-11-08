// mobilenet_onnx.cpp
// Incomplete, ONNX opset is proving to be problematic for some models.

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

// static std::vector<std::string> readClasses(const std::string &path)
// {
//     std::vector<std::string> names;
//     std::ifstream f(path);
//     for (std::string s; std::getline(f, s);)
//     {
//         if (!s.empty())
//         {
//             names.push_back(s);
//         }
//     }
//     return names;
// }

static std::unordered_map<int, std::string> load_tf_label_map(const std::string &path_to_pbtxt)
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

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        std::cerr << "usage: dnn_detect <image> <model.pb> <model.pbtxt> <label_map.pbtxt>\n";
        return 1;
    }

    cv::Mat img = cv::imread(argv[1]);
    if (img.empty())
    {
        std::cerr << "Could not read the image: " << argv[1] << std::endl;
        return 2;
    }

    cv::Mat blob = cv::dnn::blobFromImage(img, 1.0, cv::Size(320, 320), cv::Scalar(), true, false);

    // 1-based label map
    auto labels = load_tf_label_map(argv[3]);

    // Create high-level DetectionModel wrapper
    cv::dnn::Net model = cv::dnn::readNetFromONNX(argv[2]);
    // SSD Mobilenet v3 TF params: 320x320, mean=127.5, scale=1/127.5, swapRB
    model.setInput(blob);

    // Forward 4 main outputs
    std::vector<cv::Mat> outputs;
    std::vector<std::string> outNames = {
        "detection_boxes", "detection_scores", "detection_classes", "num_detections"};
    model.forward(outputs, outNames);

    cv::Mat boxes = outputs[0];   // [1, N, 4], normalized (ymin, xmin, ymax, xmax)
    cv::Mat scores = outputs[1];  // [1, N]
    cv::Mat classes = outputs[2]; // [1, N]
    cv::Mat num = outputs[3];     // [1] or [1,1]
    int numDetections = static_cast<int>(num.ptr<float>()[0]);

    // Make boxes 2D as Nx4 for easier indexing
    cv::Mat boxes2d = boxes.reshape(1, boxes.total() / 4);

    float confidenceThreshold = 0.5f;
    for (int i = 0; i < numDetections; ++i)
    {
        // Obtain confidence score
        float score = scores.at<float>(0, i);
        if (score < confidenceThreshold)
            continue;

        int clsTF = static_cast<int>(classes.at<float>(0, i));
        auto it = labels.find(clsTF);
        std::string name = (it != labels.end()) ? it->second : std::to_string(clsTF);

        float ymin = boxes2d.at<float>(i, 0);
        float xmin = boxes2d.at<float>(i, 1);
        float ymax = boxes2d.at<float>(i, 2);
        float xmax = boxes2d.at<float>(i, 3);

        int x1 = std::clamp(int(xmin * img.cols), 0, img.cols - 1);
        int y1 = std::clamp(int(ymin * img.rows), 0, img.rows - 1);
        int x2 = std::clamp(int(xmax * img.cols), 0, img.cols - 1);
        int y2 = std::clamp(int(ymax * img.rows), 0, img.rows - 1);

        cv::Rect r(x1, y1, x2 - x1, y2 - y1);
        if (r.width <= 0 || r.height <= 0)
            continue;

        std::string text = name + " " + std::to_string(int(std::round(score * 100))) + "%";

        cv::rectangle(img, r, cv::Scalar(0, 255, 0), 2);
        cv::putText(img, text, r.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("Detections", img);
    cv::waitKey(0);
    return 0;
}
