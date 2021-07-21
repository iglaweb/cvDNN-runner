#include "eval_dnn_executor.h"

using namespace std;

EvalDnnExecutor::EvalDnnExecutor() {
}

long EvalDnnExecutor::runDnnExecutor(evalexec::EvalExec& detector, std::vector<cv::Mat>& images, int batchSize) {
    _timer.tick();
    try {
        detector.run(images, batchSize);
    }
    catch (const cv::Exception& e) {
        cout << e.what() << endl;
        return -2;
    } catch (const std::exception& e) {
        cout << e.what() << endl;
        return -2;
    } catch (...) {
        return -2;
    }
    _timer.tock();
    long timeElapsedMs = _timer.duration().count();
    return timeElapsedMs;
}
