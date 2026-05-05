#include <cstdio>
#include <cstdint>
#include <cstring>
#include <inttypes.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "esp_heap_caps.h"
#include "driver/usb_serial_jtag.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "camera.h"
#include "model_data.h"

// ANSI color codes
#define COLOR_GREEN  "\033[32m"
#define COLOR_RED    "\033[31m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_RESET  "\033[0m"

#define MODEL_WIDTH 160
#define MODEL_HEIGHT 120
#define MODEL_CHANNELS 3

// Globals for TFLite
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// memory in PSRAM (1 MB)
constexpr int kTensorArenaSize = 1024 * 1024;
uint8_t* tensor_arena = nullptr;

static uint8_t image_buffer[FRAME_W * FRAME_H * FRAME_C];

void init_tflite() {
    tflite::InitializeTarget();

    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model provided is schema version %d not equal to supported version %d.\n",
               (int)model->version(), (int)TFLITE_SCHEMA_VERSION);
        return;
    }

    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddDequantize();
    resolver.AddQuantize();
    resolver.AddShape();
    resolver.AddStridedSlice();
    resolver.AddPack();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("CRITICAL ERROR: AllocateTensors() failed\n");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
}

void process_and_quantize_image() {
    int model_pixel_index = 0;

    for (int y = 0; y < MODEL_HEIGHT; y++) {
        for (int x = 0; x < MODEL_WIDTH; x++) {
            int src_x = x * 2;
            int src_y = y * 2;

            uint16_t pixels[4];
            pixels[0] = ((uint16_t)image_buffer[((src_y)   * FRAME_W + src_x)     * 2] << 8) |
                                   image_buffer[((src_y)   * FRAME_W + src_x)     * 2 + 1];
            pixels[1] = ((uint16_t)image_buffer[((src_y)   * FRAME_W + src_x + 1) * 2] << 8) |
                                   image_buffer[((src_y)   * FRAME_W + src_x + 1) * 2 + 1];
            pixels[2] = ((uint16_t)image_buffer[((src_y+1) * FRAME_W + src_x)     * 2] << 8) |
                                   image_buffer[((src_y+1) * FRAME_W + src_x)     * 2 + 1];
            pixels[3] = ((uint16_t)image_buffer[((src_y+1) * FRAME_W + src_x + 1) * 2] << 8) |
                                   image_buffer[((src_y+1) * FRAME_W + src_x + 1) * 2 + 1];

            uint32_t r_sum = 0, g_sum = 0, b_sum = 0;
            for (int i = 0; i < 4; i++) {
                r_sum += ((pixels[i] >> 11) & 0x1F) << 3;
                g_sum += ((pixels[i] >> 5)  & 0x3F) << 2;
                b_sum += (pixels[i] & 0x1F) << 3;
            }

            // Integer average then subtract 128 — no float math at all
            input->data.int8[model_pixel_index++] = (int8_t)((int)(r_sum / 4) - 128);
            input->data.int8[model_pixel_index++] = (int8_t)((int)(g_sum / 4) - 128);
            input->data.int8[model_pixel_index++] = (int8_t)((int)(b_sum / 4) - 128);
        }
    }
}
void print_tensor_info() {
    printf("Input  tensor: scale=%.6f  zero_point=%" PRId32 "  type=%d\n",
        input->params.scale,
        input->params.zero_point,
        input->type);

    printf("Output tensor: scale=%.6f  zero_point=%" PRId32 "  type=%d\n",
        output->params.scale,
        output->params.zero_point,
        output->type);
}

void setup() {
    tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);

    nvs_flash_init();
    if (!camera_init()) abort();
    init_tflite();
    print_tensor_info();

    usb_serial_jtag_driver_config_t cfg = {
        .tx_buffer_size = 256 * 1024,
        .rx_buffer_size = 512,
    };
    usb_serial_jtag_driver_install(&cfg);

    printf("Smart Lock AI Initialized! Run Python Viewer script to start live feed.\n");
    char c;
    do {
        usb_serial_jtag_read_bytes(&c, 1, portMAX_DELAY);
    } while (c != 'S');
}
void send_preview_frame() {
    const char* preamble = "\n===FRAME===\n";
    usb_serial_jtag_write_bytes(preamble, strlen(preamble), pdMS_TO_TICKS(50));

    uint8_t chunk_buf[1024];
    int chunk_idx = 0;

    // Send full 320x240 frame
    for (int y = 0; y < FRAME_H; y++) {
        for (int x = 0; x < FRAME_W; x++) {
            int idx = (y * FRAME_W + x) * 2;
            chunk_buf[chunk_idx++] = image_buffer[idx];
            chunk_buf[chunk_idx++] = image_buffer[idx + 1];

            if (chunk_idx == sizeof(chunk_buf)) {
                usb_serial_jtag_write_bytes(chunk_buf, chunk_idx, pdMS_TO_TICKS(50));
                chunk_idx = 0;
            }
        }
    }
    if (chunk_idx > 0) {
        usb_serial_jtag_write_bytes(chunk_buf, chunk_idx, pdMS_TO_TICKS(50));
    }
}
void loop() {
    if (input == nullptr) return;

    if (camera_capture_frame(image_buffer)) {
        process_and_quantize_image();
        interpreter->Invoke();

        int best_class = -1;
        int8_t max_score = -128;
        for (int i = 0; i < 3; i++) {
            if (output->data.int8[i] > max_score) {
                max_score = output->data.int8[i];
                best_class = i;
            }
        }

        int8_t SECURITY_THRESHOLD = 20;
        
        if (best_class == 0 && max_score < SECURITY_THRESHOLD) {
            best_class = 1; // Demote to "Unrecognized Face"
        }

        printf("Scores: [%d, %d, %d]\n",
            output->data.int8[0],
            output->data.int8[1],
            output->data.int8[2]);

        printf("\n");
        if (best_class == 0) {
            printf(COLOR_GREEN "Valid Input: Welcome! (Score: %d)" COLOR_RESET "\n", max_score);
        } else if (best_class == 1) {
            printf(COLOR_RED "Invalid Input: Unrecognized Face. Access Denied. (Score: %d)" COLOR_RESET "\n", max_score);
        } else {
            printf(COLOR_YELLOW "Idle: No face detected. Waiting... (Score: %d)" COLOR_RESET "\n", max_score);
        }

        fflush(stdout);
        vTaskDelay(pdMS_TO_TICKS(10));
        send_preview_frame();
    }
}

extern "C" void app_main() {
    setup();
    while (true) loop();
}