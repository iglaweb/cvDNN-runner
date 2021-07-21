#include "eval_executor.h"

namespace evalexec {

EvalExec::EvalExec(
                   const std::string& modelName,
                   bool isNHWC,
                   common::ColorSpace& colorSpace,
                   cv::Size& inputSize,
                   int prefBackendId,
                   int prefTarget) {
    this->colorSpace = colorSpace;
    this->imageSize = inputSize;
    this->channelsCount = colorSpace == common::GRAY ? 1 : 3;
    this->isNHWC = isNHWC;
    this->predictor = cv::dnn::readNet(modelName); //onnx or pb?
    this->predictor.setPreferableBackend(prefBackendId);
    this->predictor.setPreferableTarget(prefTarget);
    
    //universal way for tensorflow pb and onnx
    std::vector<std::string> outNames = predictor.getUnconnectedOutLayersNames();
    outBlobNames.insert(outBlobNames.end(), outNames.begin(), outNames.end());
}

EvalExec::EvalExec(common::ModelConfig& modelConfig) : EvalExec(
                                                                modelConfig.modelName,
                                                                modelConfig.isNHWC,
                                                                modelConfig.colorSpace,
                                                                modelConfig.inputSize,
                                                                modelConfig.backendId,
                                                                modelConfig.prefTarget
                                                                ) {
}

bool EvalExec::run(std::vector<cv::Mat>& images, int batchSize) {
    if(batchSize < 1) {
        throw std::invalid_argument("Batch size must be positive");
    }
    
    bool generateRandomImg = images.empty();
    if(!images.empty() && images.size() != batchSize) {
        throw std::invalid_argument("image number does not match with batch size");
    }
    
    if(generateRandomImg) {
        for (int i = 0; i < batchSize; i++) {
            cv::Mat3b random_image(imageSize);
            cv::randu(random_image, cv::Scalar(0,0,0), cv::Scalar(256,256,256));
            images.push_back(random_image);
        }
    }
    
    std::vector<cv::Mat> faceImages;
    for (int i = 0; i < batchSize; i++) {
        auto image = images[i];
        if(colorSpace == common::GRAY) {
            cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        }

        if(image.cols != imageSize.width || image.rows != imageSize.height) {
            cv::resize(image, image, imageSize);
        }
        faceImages.push_back(image);
    }
    
    //optimization
    cv::Mat blob;
    double scaleFactor = 1.0;
    if(faceImages.size() > 1) {
        blob = cv::dnn::blobFromImages(faceImages, scaleFactor);
    } else {
        blob = cv::dnn::blobFromImage(faceImages.at(0), scaleFactor);
    }
    if(isNHWC) { //only for onnx
        // (1, 1, 100, 100) -> (1, 100, 100, 1)
        const int imagesCount = static_cast<int>(faceImages.size());
        blob = blob.reshape(1, cv::dnn::MatShape({imagesCount, imageSize.width, imageSize.height, channelsCount}));
    }

    predictor.setInput(blob);
    if(outBlobNames.empty()) {
        predictor.forward(out);
    } else {
        predictor.forward(out, outBlobNames);
    }

    // release
    for (int i = 0; i < out.size(); i++) {
        out[i].release();
    }
    
    for(cv::Mat& faceImg : faceImages) {
        faceImg.release();
    }
    blob.release();
    return true;
  }
}
