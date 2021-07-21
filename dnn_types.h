//
//  dnn_types.h
//  DNN Executor
//
//  Created by Igor Lashkov on 21.07.2021.
//  Copyright © 2021 Igor Lashkov. All rights reserved.
//

#ifndef dnn_types_h
#define dnn_types_h

#include <iostream>
#include <cstring>
#include <iterator> // for iterators
#include <vector> // for vectors
#include <chrono>

#pragma once

namespace common {

    enum BackendLaunch { CPU = 0, GPU = 1, ALL = 2 };
    enum ColorSpace { GRAY = 0, RGB = 1};

    static const cv::dnn::Backend backendIds[] = {
        cv::dnn::DNN_BACKEND_DEFAULT,
        cv::dnn::DNN_BACKEND_HALIDE,
        cv::dnn::DNN_BACKEND_INFERENCE_ENGINE,            //!< Intel's Inference Engine computational backend
                                                 //!< @sa setInferenceEngineBackendType
        cv::dnn::DNN_BACKEND_OPENCV,
        cv::dnn::DNN_BACKEND_VKCOM,
        cv::dnn::DNN_BACKEND_CUDA,
    };

    static const cv::dnn::Target prefTarget[] = {
        cv::dnn::DNN_TARGET_CPU,
        cv::dnn::DNN_TARGET_OPENCL,
        cv::dnn::DNN_TARGET_OPENCL_FP16,
        cv::dnn::DNN_TARGET_MYRIAD,
        cv::dnn::DNN_TARGET_VULKAN,
        cv::dnn::DNN_TARGET_FPGA,  //!< FPGA device with CPU fallbacks using Inference Engine's Heterogeneous plugin.
        cv::dnn::DNN_TARGET_CUDA,
        cv::dnn::DNN_TARGET_CUDA_FP16
    };

    inline std::vector<std::string> split(std::string s, std::string delimiter) {
        std::vector<std::string> list;
        size_t pos = 0;
        std::string token;
        while ((pos = s.find(delimiter)) != std::string::npos) {
            token = s.substr(0, pos);
            list.push_back(token);
            s.erase(0, pos + delimiter.length());
        }
        list.push_back(s);
        return list;
    }


    inline std::string getBaseName(std::string const &path) {
        auto index = path.find_last_of("/\\");
        if(index == std::string::npos) {
            return path;
        }
        return path.substr(index + 1);
    }

    struct ModelConfig {
        int batchSize;
        cv::Size inputSize;
        bool isNHWC;
        ColorSpace colorSpace;
        int backendId;
        int prefTarget;
        std::string modelName;
        
        std::string toString() {
          return "batchSize = " + std::to_string(batchSize) +
            ", inputSize = " + std::to_string(inputSize.width) + "x" + std::to_string(inputSize.height) +
            ", isNHWC = " + std::to_string(isNHWC) +
            ", color = " + (colorSpace == GRAY ? "gray" : "color") +
            ", backendId = " + std::to_string(backendId) +
            ", prefTarget = " + std::to_string(prefTarget) +
            ", modelName = " + getBaseName(modelName);
        }
    };

    struct InputModel {
        ModelConfig modelConfig;
        long avgTimeMs = -1L;
        long initTimeMs = -1L;
        long maxTime = LONG_MIN;
        long minTime = LONG_MAX;
        
        std::string toString() {
          std::string model = "model = [" + modelConfig.toString() + "]";
          return model +
            ", avgTimeMs = " + std::to_string(avgTimeMs) + " ms" +
            ", initTimeMs = " + std::to_string(initTimeMs) + " ms" +
            ", maxTime = " + std::to_string(maxTime) + " ms" +
            ", minTime = " + std::to_string(minTime) + " ms";
        }
    };
}

#endif /* dnn_types_h */
