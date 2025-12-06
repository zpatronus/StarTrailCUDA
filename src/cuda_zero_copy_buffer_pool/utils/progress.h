#pragma once
#include <iomanip>
#include <iostream>

inline void print_progress(int current, int total) {
    const int width = 50;
    double ratio = total ? (double)current / total : 0.0;
    int filled = (int)(ratio * width);

    std::cout << "\r[" << std::string(filled, '=') << std::string(width - filled, ' ') << "] "
              << current << "/" << total << " " << std::setw(3) << (int)(ratio * 100) << "%";
    std::cout.flush();

    if (current >= total)
        std::cout << std::endl;
}
