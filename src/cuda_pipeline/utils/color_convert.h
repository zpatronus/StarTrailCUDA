#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

void yuv420p_to_rgb(const uint8_t* d_y, const uint8_t* d_u, const uint8_t* d_v, uint8_t* d_rgb,
                    int width, int height, int y_pitch, int uv_pitch, int rgb_pitch,
                    cudaStream_t stream);

void rgb_to_yuv420p(const uint8_t* d_rgb, uint8_t* d_y, uint8_t* d_u, uint8_t* d_v, int width,
                    int height, int rgb_pitch, int y_pitch, int uv_pitch, cudaStream_t stream);

void nv12_to_rgb(const uint8_t* d_y, const uint8_t* d_uv, uint8_t* d_rgb, int width, int height,
                 int y_pitch, int uv_pitch, int rgb_pitch, cudaStream_t stream);

void nv12_to_yuv420p(const uint8_t* d_y_in, const uint8_t* d_uv_in, uint8_t* h_y_out,
                     uint8_t* h_u_out, uint8_t* h_v_out, uint8_t* d_u_temp, uint8_t* d_v_temp,
                     int width, int height, int y_pitch_in, int uv_pitch_in, cudaStream_t stream);
