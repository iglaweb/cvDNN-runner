//
//  Timer.h
//  DNN Executor
//
//  Created by Igor Lashkov on 27.05.2021.
//  Copyright © 2021 Igor Lashkov. All rights reserved.
//

#ifndef Timer_h
#define Timer_h

#include <chrono>
#include <cstdio>
#include <fstream>

template <class TimeT = std::chrono::milliseconds,
          class ClockT = std::chrono::steady_clock>
class Timer
{
    using timep_t = typename ClockT::time_point;
    timep_t _start = ClockT::now(), _end = {};

public:
    void tick() {
        _end = timep_t{};
        _start = ClockT::now();
    }
    
    void tock() { _end = ClockT::now(); }
    
    template <class TT = TimeT>
    TT duration() const {
        if(_end == timep_t{}) throw std::invalid_argument("toc before reporting");
        return std::chrono::duration_cast<TT>(_end - _start);
    }
};

#endif /* Timer_h */
