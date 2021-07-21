#ifndef __EVAL_DNN_EXECUTOR
#define __EVAL_DNN_EXECUTOR

#include "eval_executor.h"
#include "Timer.h"
#include "dnn_types.h"

using namespace std;


class EvalDnnExecutor {
    
public:
    EvalDnnExecutor();
    long runDnnExecutor(evalexec::EvalExec& detector, std::vector<cv::Mat>& images, int batchSize);
    Timer<std::chrono::milliseconds, std::chrono::steady_clock> _timer;
private:
};

inline evalexec::EvalExec createDetector(common::ModelConfig& modelConfig) {
    return evalexec::EvalExec(modelConfig);
}

#endif // __EVAL_DNN_EXECUTOR
