#ifndef __EVAL_EXEC
#define __EVAL_EXEC
#include <iostream>
#include <cstring>
#include <iterator> // for iterators
#include <vector> // for vectors
#include <chrono>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <stdio.h>
#include <opencv2/core/types_c.h>
#include <opencv2/opencv.hpp>

#include "dnn_types.h"

namespace evalexec {
	class EvalExec {
	private:
        int channelsCount; //1 for grayscale custom or 3 for rgb mobilenet
        common::ColorSpace colorSpace = common::RGB;
        bool isNHWC; // input shape format
        std::vector<cv::String> outBlobNames;
		std::vector<cv::Mat> out;
        cv::dnn::Net predictor; // dnn model
        cv::Size imageSize;
	public:
        EvalExec(const std::string& modelName,
                 bool isNHWC,
                 common::ColorSpace& colorSpace,
                 cv::Size& inputSize,
                 int prefBackendId,
                 int prefTarget
        );
        EvalExec(common::ModelConfig &modelConfig);
        bool run(std::vector<cv::Mat>& images, int batchSize);
	};
}
#endif // __EVAL_EXEC
