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
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <map>

const float CONFIDENCE_THRESHOLD = 0.4f;
const float LOW_CONFIDENCE_THRESHOLD = 0.2f;
bool SHOW_LOW_CONFIDENCE = true;
const float BOX_TO_IMAGE_RATIO_THRESHOLD = 0.45f;
const float NMS_THRESHOLD = 0.4f;
const int LINE_WIDTH = 5;

// colors
cv::Scalar LOW_CONFIDENCE_COLOR = cv::Scalar(0, 0, 255); // Red
cv::Scalar DEFAULT_COLOR = cv::Scalar(0, 255, 0);        // Green

template <typename T>
class BlockingQueue
{
public:
    BlockingQueue(size_t capacity) : capacity_(capacity) {}
    bool push(T &&item)
    {
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [&]
                 { return stop_ || q_.size() < capacity_; });
        if (stop_)
            return false;
        q_.push(std::move(item));
        cv_.notify_all();
        return true;
    }

    bool pop(T &out)
    {
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [&]
                 { return stop_ || !q_.empty(); });
        if (stop_ && q_.empty())
            return false;
        out = std::move(q_.front());
        q_.pop();
        cv_.notify_all();
        return true;
    }

    void stop()
    {
        std::lock_guard<std::mutex> lock(m_);
        stop_ = true;
        cv_.notify_all();
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    std::queue<T> q_;
    size_t capacity_;
    bool stop_ = false;
};

// Task and result structs
struct FrameTask
{
    int index;
    double timestamp;
    cv::Mat frame;
};

struct FrameResult
{
    int index;
    cv::Mat processed;
    int numDetections;
};

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
        std::cerr << "usage: mobilenet_video <video> <model.pb> <model.pbtxt> <labels.pbtxt> <frame_interval> <output_video> [num_threads]\n";
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

    unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0)
        hc = 1;
    int numThreads = (argc >= 8) ? std::stoi(argv[7]) : static_cast<int>(hc);
    if (numThreads < 1)
        numThreads = 1;
    std::cout << "Using " << numThreads << " threads for processing.\n";

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
    std::cout << "Initializing video writer with MP4V codec...\n";
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
    if (classes.empty())
    {
        std::cerr << "No class labels loaded. Exiting.\n";
        return 4;
    }

    std::cout << "Loading base model...\n";
    cv::dnn::Net baseNet = load_tf_model(argv[2], argv[3]);
    if (baseNet.empty())
    {
        std::cerr << "Model load failed. Check paths.\n";
        return 4;
    }
    baseNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    baseNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    std::cout << "Spawning " << numThreads << " worker threads\n";

    BlockingQueue<FrameTask> taskQueue(64);
    std::mutex resultMutex;
    std::map<int, FrameResult> pendingResults;
    std::atomic<int> processedFrames{0};
    std::atomic<bool> fatalModelError{false};

    std::vector<cv::dnn::Net> nets;
    nets.reserve(numThreads);
    // Load separate model instance for each thread
    for (int i = 0; i < numThreads; ++i)
    {
        nets.emplace_back(load_tf_model(argv[2], argv[3]));
    }

    // Worker
    auto worker = [&](int tid)
    {
        cv::dnn::Net &localModel = nets[tid];
        FrameTask task;
        while (taskQueue.pop(task))
        {
            if (fatalModelError.load(std::memory_order_relaxed))
            {
                break;
            }
            cv::Mat frameCopy = task.frame.clone();
            std::vector<Detection> detections;
            // Only run detection on interval frames, others will reuse last detections
            if (task.index % frameInterval == 0)
            {
                try
                {
                    auto processedFrame = process_frame(frameCopy, localModel, classes);
                    frameCopy = processedFrame.first;
                    detections = std::move(processedFrame.second);
                    if (!detections.empty())
                    {
                        non_maximum_suppression(frameCopy, detections);
                    }
                }
                catch (const cv::Exception &e)
                {
                    std::cerr << "OpenCV exception in worker " << tid << " on frame " << task.index << ": " << e.what() << std::endl;
                    fatalModelError.store(true, std::memory_order_relaxed);
                    taskQueue.stop();
                    break;
                }
            }
            else
            {
                // No processing, just pass through
            }
            {
                std::lock_guard<std::mutex> lock(resultMutex);
                pendingResults.emplace(task.index, FrameResult{task.index, frameCopy, static_cast<int>(detections.size())});
            }
            processedFrames.fetch_add(1, std::memory_order_relaxed);
        }
    };

    // Launch workers
    std::vector<std::thread> threads;
    for (int i = -0; i < numThreads; ++i)
    {
        threads.emplace_back(worker, i);
    }

    int nextWriteIndex = 0;
    cv::Mat lastProcessedFrame; // cache last interval processed frame

    // Produceer and writer loop
    int frameNumber = 0;

    while (true)
    {
        cv::Mat frame;
        if (!cap.read(frame))
        {
            break; // End of video
        }
        double timestamp = frameNumber / fps;

        // Enqueue task
        taskQueue.push(FrameTask{frameNumber, timestamp, frame});

        // Drain available in-order results
        bool advanced = true;
        while (advanced)
        {
            advanced = false;
            {
                std::lock_guard<std::mutex> lock(resultMutex);
                auto it = pendingResults.find(nextWriteIndex);
                if (it != pendingResults.end())
                {
                    cv::Mat outputFrame = it->second.processed;

                    // If this frame not processed, substitute last processed annotated frame
                    if (nextWriteIndex % frameInterval != 0 && !lastProcessedFrame.empty())
                    {
                        outputFrame = lastProcessedFrame.clone();
                    }
                    else if (nextWriteIndex % frameInterval == 0)
                    {
                        lastProcessedFrame = outputFrame.clone();
                    }

                    videoWriter.write(outputFrame);
                    pendingResults.erase(it);
                    nextWriteIndex++;
                    advanced = true;

                    if (nextWriteIndex % 100 == 0)
                    {
                        double progress = (double)nextWriteIndex / totalFrames * 100.0;
                        std::cout << "Progress: " << std::fixed << std::setprecision(1)
                                  << progress << "% (" << nextWriteIndex << "/" << totalFrames << ")\n";
                    }
                }
            }
        }
        frameNumber++;
    }

    // Signal workers to stop
    taskQueue.stop();
    for (auto &t : threads)
        t.join();

    // Flush reamaining results in order
    while (nextWriteIndex < frameNumber)
    {
        std::lock_guard<std::mutex> lock(resultMutex);
        auto it = pendingResults.find(nextWriteIndex);
        if (it != pendingResults.end())
        {
            cv::Mat outputFrame = it->second.processed;
            if (nextWriteIndex % frameInterval != 0 && !lastProcessedFrame.empty())
            {
                outputFrame = lastProcessedFrame.clone();
            }
            else if (nextWriteIndex % frameInterval == 0)
            {
                lastProcessedFrame = outputFrame.clone();
            }
            videoWriter.write(outputFrame);
            pendingResults.erase(it);
            ++nextWriteIndex;
        }
    }

    std::cout << "============================================\n";
    std::cout << "Parallel processing complete\n";
    std::cout << "Total frames: " << frameNumber << "\n";
    std::cout << "Processed tasks: " << processedFrames.load() << "\n";

    cap.release();
    videoWriter.release();
    return 0;
}
