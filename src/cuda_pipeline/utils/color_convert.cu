#include "color_convert.h"

__global__ void nv12_to_rgb_kernel(const uint8_t* y_plane, const uint8_t* uv_plane, uint8_t* rgb,
                                   int width, int height, int y_pitch, int uv_pitch,
                                   int rgb_pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int y_idx = y * y_pitch + x;
        int uv_x = (x / 2) * 2;
        int uv_y = y / 2;
        int uv_idx = uv_y * uv_pitch + uv_x;

        float Y = y_plane[y_idx];
        float U = uv_plane[uv_idx] - 128.0f;
        float V = uv_plane[uv_idx + 1] - 128.0f;

        float R = Y + 1.402f * V;
        float G = Y - 0.344f * U - 0.714f * V;
        float B = Y + 1.772f * U;

        R = fminf(fmaxf(R, 0.0f), 255.0f);
        G = fminf(fmaxf(G, 0.0f), 255.0f);
        B = fminf(fmaxf(B, 0.0f), 255.0f);

        int rgb_idx = y * rgb_pitch + x * 3;
        rgb[rgb_idx + 0] = (uint8_t)R;
        rgb[rgb_idx + 1] = (uint8_t)G;
        rgb[rgb_idx + 2] = (uint8_t)B;
    }
}

__global__ void yuv420p_to_rgb_kernel(const uint8_t* y_plane, const uint8_t* u_plane,
                                      const uint8_t* v_plane, uint8_t* rgb, int width, int height,
                                      int y_pitch, int uv_pitch, int rgb_pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int y_idx = y * y_pitch + x;
        int uv_idx = (y / 2) * uv_pitch + (x / 2);

        float Y = y_plane[y_idx];
        float U = u_plane[uv_idx] - 128.0f;
        float V = v_plane[uv_idx] - 128.0f;

        float R = Y + 1.402f * V;
        float G = Y - 0.344f * U - 0.714f * V;
        float B = Y + 1.772f * U;

        R = fminf(fmaxf(R, 0.0f), 255.0f);
        G = fminf(fmaxf(G, 0.0f), 255.0f);
        B = fminf(fmaxf(B, 0.0f), 255.0f);

        int rgb_idx = y * rgb_pitch + x * 3;
        rgb[rgb_idx + 0] = (uint8_t)R;
        rgb[rgb_idx + 1] = (uint8_t)G;
        rgb[rgb_idx + 2] = (uint8_t)B;
    }
}

__global__ void rgb_to_yuv420p_kernel(const uint8_t* rgb, uint8_t* y_plane, uint8_t* u_plane,
                                      uint8_t* v_plane, int width, int height, int rgb_pitch,
                                      int y_pitch, int uv_pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int rgb_idx = y * rgb_pitch + x * 3;
        float R = rgb[rgb_idx + 0];
        float G = rgb[rgb_idx + 1];
        float B = rgb[rgb_idx + 2];

        float Y = 0.299f * R + 0.587f * G + 0.114f * B;
        float U = -0.169f * R - 0.331f * G + 0.500f * B + 128.0f;
        float V = 0.500f * R - 0.419f * G - 0.081f * B + 128.0f;

        Y = fminf(fmaxf(Y, 0.0f), 255.0f);
        U = fminf(fmaxf(U, 0.0f), 255.0f);
        V = fminf(fmaxf(V, 0.0f), 255.0f);

        int y_idx = y * y_pitch + x;
        y_plane[y_idx] = (uint8_t)Y;

        if ((x % 2 == 0) && (y % 2 == 0)) {
            int uv_idx = (y / 2) * uv_pitch + (x / 2);
            u_plane[uv_idx] = (uint8_t)U;
            v_plane[uv_idx] = (uint8_t)V;
        }
    }
}

void yuv420p_to_rgb(const uint8_t* d_y, const uint8_t* d_u, const uint8_t* d_v, uint8_t* d_rgb,
                    int width, int height, int y_pitch, int uv_pitch, int rgb_pitch,
                    cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    yuv420p_to_rgb_kernel<<<grid, block, 0, stream>>>(d_y, d_u, d_v, d_rgb, width, height, y_pitch,
                                                      uv_pitch, rgb_pitch);
}

void rgb_to_yuv420p(const uint8_t* d_rgb, uint8_t* d_y, uint8_t* d_u, uint8_t* d_v, int width,
                    int height, int rgb_pitch, int y_pitch, int uv_pitch, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    rgb_to_yuv420p_kernel<<<grid, block, 0, stream>>>(d_rgb, d_y, d_u, d_v, width, height,
                                                      rgb_pitch, y_pitch, uv_pitch);
}

void nv12_to_rgb(const uint8_t* d_y, const uint8_t* d_uv, uint8_t* d_rgb, int width, int height,
                 int y_pitch, int uv_pitch, int rgb_pitch, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    nv12_to_rgb_kernel<<<grid, block, 0, stream>>>(d_y, d_uv, d_rgb, width, height, y_pitch,
                                                   uv_pitch, rgb_pitch);
}

__global__ void nv12_deinterleave_uv_kernel(const uint8_t* uv_interleaved, uint8_t* u_plane,
                                            uint8_t* v_plane, int uv_width, int uv_height,
                                            int uv_pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < uv_width && y < uv_height) {
        int uv_idx = y * uv_pitch + x * 2;
        int out_idx = y * uv_width + x;

        u_plane[out_idx] = uv_interleaved[uv_idx];
        v_plane[out_idx] = uv_interleaved[uv_idx + 1];
    }
}

void nv12_to_yuv420p(const uint8_t* d_y_in, const uint8_t* d_uv_in, uint8_t* h_y_out,
                     uint8_t* h_u_out, uint8_t* h_v_out, uint8_t* d_u_temp, uint8_t* d_v_temp,
                     int width, int height, int y_pitch_in, int uv_pitch_in, cudaStream_t stream) {
    cudaMemcpy2DAsync(h_y_out, width, d_y_in, y_pitch_in, width, height, cudaMemcpyDeviceToHost,
                      stream);

    int uv_width = width / 2;
    int uv_height = height / 2;
    dim3 block(16, 16);
    dim3 grid((uv_width + block.x - 1) / block.x, (uv_height + block.y - 1) / block.y);
    nv12_deinterleave_uv_kernel<<<grid, block, 0, stream>>>(d_uv_in, d_u_temp, d_v_temp, uv_width,
                                                            uv_height, uv_pitch_in);
    cudaMemcpyAsync(h_u_out, d_u_temp, uv_width * uv_height, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_v_out, d_v_temp, uv_width * uv_height, cudaMemcpyDeviceToHost, stream);
}
