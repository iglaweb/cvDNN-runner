//
//  main.cpp
//  DNN Executor
//
//  Created by Igor Lashkov on 20.06.2021.
//  Copyright © 2021 Igor Lashkov. All rights reserved.
//

#include <stdio.h>
#include <algorithm>

#include "eval_dnn_executor.h"

using namespace std;

Timer<std::chrono::milliseconds, std::chrono::steady_clock> timer;

inline std::vector<common::ModelConfig> prepareConfigs(
                                                      string &model_name,
                                                      cv::Size& inputSize,
                                                      common::ColorSpace color,
                                                      int batch_size,
                                                      bool isNHWC,
                                                      common::BackendLaunch backendL) {
    std::vector<common::ModelConfig> evalExecs;
    
    if(backendL == common::CPU) {
        common::ModelConfig modelConfig;
        modelConfig.backendId = cv::dnn::DNN_BACKEND_OPENCV;
        modelConfig.prefTarget = cv::dnn::DNN_TARGET_OPENCL;
        modelConfig.modelName = model_name;
        modelConfig.inputSize = inputSize;
        modelConfig.batchSize = batch_size;
        modelConfig.isNHWC = isNHWC;
        modelConfig.colorSpace = color;
        
        evalExecs.push_back(modelConfig);
    } else if(backendL == common::GPU) {
        common::ModelConfig modelConfig;
        modelConfig.backendId = cv::dnn::DNN_BACKEND_CUDA;
        modelConfig.prefTarget = cv::dnn::DNN_TARGET_CUDA_FP16;
        modelConfig.modelName = model_name;
        modelConfig.inputSize = inputSize;
        modelConfig.batchSize = batch_size;
        modelConfig.isNHWC = isNHWC;
        modelConfig.colorSpace = color;
        
        evalExecs.push_back(modelConfig);
    } else {
        for (const auto backend : common::backendIds) {
            for (const auto target : common::prefTarget) {
                common::ModelConfig modelConfig;
                modelConfig.backendId = backend;
                modelConfig.prefTarget = target;
                modelConfig.modelName = model_name;
                modelConfig.inputSize = inputSize;
                modelConfig.batchSize = batch_size;
                modelConfig.isNHWC = isNHWC;
                modelConfig.colorSpace = color;
                evalExecs.push_back(modelConfig);
            }
        }
    }
    return evalExecs;
}

inline void printLog(std::vector<common::InputModel> inputModels) {
    if(inputModels.size() == 1) {
        std::cout << "One model: " << inputModels[0].toString() << std::endl;
    } else {
        common::InputModel minAvgTimeModel = inputModels[0];
        common::InputModel minInitTimeModel = inputModels[0];
        for(auto model : inputModels) {
            if(model.avgTimeMs != -1 && minAvgTimeModel.avgTimeMs > model.avgTimeMs) {
                minAvgTimeModel = model;
            }
            if(model.initTimeMs != -1 && minInitTimeModel.initTimeMs > model.initTimeMs) {
                minInitTimeModel = model;
            }
            auto modelConfig = model.modelConfig;
            auto filename = common::getBaseName(modelConfig.modelName);
            std::cout << "Model: " << model.toString() << std::endl;
        }
        
        std::cout << "Model with min init: " << minInitTimeModel.toString() << std::endl;
        std::cout << "Model with min avg time: " << minAvgTimeModel.toString() << std::endl;
    }
}

inline std::pair<evalexec::EvalExec, long> resolveNextDetector(common::ModelConfig& currentDetector) {
    std::cout << "Take next config: " << currentDetector.toString() << std::endl;
    
    timer.tick();
    evalexec::EvalExec detector = createDetector(currentDetector);
    timer.tock();
    
    
    long time = timer.duration().count();
    return std::pair<evalexec::EvalExec, long>(detector, time);
}

int runInference(string &model_name, cv::Size& input_size, common::ColorSpace& color, int batch_size, bool isNHWC, common::BackendLaunch& backend, string &s) {
    
    bool has_only_digits = (s.find_first_not_of("0123456789") == std::string::npos);
    cv::VideoCapture cap;
    if(has_only_digits) {
        cap.open(std::stoi(s));
    } else {
        cap.open(s);
    }
    
    if(!cap.isOpened()) {
        cout << "Media is not opened" << endl;
        return -1;
    }
    cout << "Media opened" << endl;
    
    
    std::vector<common::ModelConfig> evalExecs = prepareConfigs(model_name, input_size, color, batch_size, isNHWC, backend);
    
    vector<common::ModelConfig>::iterator ptr = evalExecs.begin();
    common::ModelConfig currentConfig = *ptr;
    
    std::pair<evalexec::EvalExec, long> detector = resolveNextDetector(currentConfig);
    common::InputModel currentModel;
    currentModel.initTimeMs = detector.second;
    currentModel.modelConfig = currentConfig;
    
    std::vector<common::InputModel> outputResults;
    
    const int executionsPerDetector = 5;
    int executionsPerDetectorElapsed = executionsPerDetector;
    long elapsedTimeDetector = 0L;
    int successExecutions = 0;
    
    std::vector<cv::Mat> matBuffer;
    EvalDnnExecutor evalDnnExec = EvalDnnExecutor();
    
    int frameCounter = 0;
    while (1) {
        cv::Mat img;
        cap >> img;
        frameCounter++;
        if (img.rows == 0 || img.cols == 0) {
            img.release();
            break;
        }
        
        matBuffer.push_back(img);
        if(frameCounter % batch_size == 0) {
            long time = evalDnnExec.runDnnExecutor(detector.first, matBuffer, currentModel.modelConfig.batchSize);
            if(time >= 0) {
                currentModel.minTime = std::min<long>(time, currentModel.minTime);
                currentModel.maxTime = std::max<long>(time, currentModel.maxTime);
                
                elapsedTimeDetector += time;
                successExecutions++;
                std::cout << "process " << time << "ms to run.\n";
            }
            
            for (std::size_t i = 0; i != matBuffer.size(); ++i) {
                cv::Mat mat = matBuffer[i];
                mat.release();
            }
            matBuffer.clear();
            executionsPerDetectorElapsed--;
        }
        
        //finish with this detector
        if(executionsPerDetectorElapsed == 0) {
            //calc timings
            long avgTimeExecute = successExecutions > 0 ? elapsedTimeDetector / (long)successExecutions : -1L;
            common::InputModel inputModel;
            inputModel.modelConfig = currentModel.modelConfig;
            inputModel.initTimeMs = currentModel.initTimeMs;
            inputModel.minTime = currentModel.minTime;
            inputModel.maxTime = currentModel.maxTime;
            inputModel.avgTimeMs = avgTimeExecute;
            outputResults.push_back(inputModel);
            
            if(++ptr == evalExecs.end()) {
                //we finished
                std::cout << "Detectors ended" << std::endl;
                break;
            }
            
            executionsPerDetectorElapsed = executionsPerDetector;
            currentConfig = *ptr;
            std::cout << "Take next config, " << currentConfig.toString() << std::endl;
            //take next or exit
            detector = resolveNextDetector(currentConfig);
            
            currentModel = common::InputModel();
            currentModel.initTimeMs = detector.second;
            currentModel.modelConfig = currentConfig;
        }
                
        cv::imshow("win", img);
        char c = (char) cv::waitKey(10);
        if (c == 27) {
            break;
        }
        img.release();
    }
    
    printLog(outputResults);
    cv::destroyAllWindows();
    cap.release();
    return 0;
}

int main() {
    string model_name;
    std::cout << "Enter the model name" << std::endl;
    cin >> model_name;
    
    string input_size;
    std::cout << "Enter input image size e.g. 640,480,3 (width,height,channels OR 640,480 (default rgb))" << std::endl;
    cin >> input_size;
    if(std::cin.fail()) { std::cout << "Failed to parse type"; return 0;}
    
    vector<string> listSizes = common::split(input_size, ",");
    if(listSizes.size() < 2) {
        std::cout << "Failed to parse image size. Size is less than 2";
        return 0;
    }
    int n1 = std::stoi(listSizes.at(0));
    int n2 = std::stoi(listSizes.at(1));
    cv::Size imageSize = cv::Size(n1, n2);
    
    int color = listSizes.size() > 2 ? std::stoi(listSizes.at(2)) : 3; // default color
    common::ColorSpace colorSpace = color == 3 ? common::RGB : common::GRAY;
    
    
    int batch_size;
    std::cout << "Enter input batch size (minimum 1)" << std::endl;
    cin >> batch_size;
    if(std::cin.fail()) { std::cout << "Failed to parse type"; return 0;}
    
    int input_format;
    std::cout << "Enter type of input format (0 - NHWC, 1 – NCHW)" << std::endl;
    cin >> input_format;
    bool isNHWC = input_format == 0;
    
    int backendType;
    std::cout << "Enter type of backend (0 - CPU, 1 – GPU/CUDA, 2 - ALL)" << std::endl;
    cin >> backendType;
    
    common::BackendLaunch backend = static_cast<common::BackendLaunch>(backendType);
    
    string mediaPath;
    std::cout << "Enter the fullname of the video or camera index (0, 1, ...)" << std::endl;
    cin >> mediaPath;
    
    runInference(model_name, imageSize, colorSpace, batch_size, isNHWC, backend, mediaPath);
    return 0;
}
