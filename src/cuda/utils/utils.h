#pragma once
#include <iomanip>
#include <iostream>

inline void print_progress(int current, int total) {
    if (total <= 0)
        return;

    int percent = (current * 100) / total;
    int bar_width = 50;
    int filled = (bar_width * current) / total;

    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < filled)
            std::cout << "=";
        else if (i == filled)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << std::setw(3) << percent << "% (" << current << "/" << total << ")"
              << std::flush;

    if (current >= total) {
        std::cout << std::endl;
    }
}
