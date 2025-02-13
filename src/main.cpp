#include "esp_camera.h"
#include "esp_timer.h"
#include "img_converters.h"
#include "Arduino.h"
#include "fb_gfx.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "NeuralNetwork.h"

#define CAMERA_MODEL_XIAO_ESP32S3
#include "camera_pins.h"

#define IMAGE_WIDTH 320
#define IMAGE_HEIGHT 320
#define LED_PIN 21
#define CONFIDENCE_THRESHOLD 0.5
#define TARGET_CLASS_INDEX 0
#define DEBUG_TFLITE 0

NeuralNetwork *neural_net;

// Define a bounding box structure
struct BoundingBox {
    float centerX;
    float centerY;
    float boxWidth;
    float boxHeight;
    float confidence;
    int classID;
};

// Convert RGB565 to RGB888
uint32_t convertRGB565toRGB888(uint16_t color) {
    uint8_t highByte, lowByte;
    uint32_t red, green, blue;

    lowByte = (color >> 8) & 0xFF;
    highByte = color & 0xFF;

    red = (lowByte & 0x1F) << 3;
    green = ((highByte & 0x07) << 5) | ((lowByte & 0xE0) >> 3);
    blue = (highByte & 0xF8);

    return (red << 16) | (green << 8) | blue;
}

// Extract image data and quantize if required
int processImage(camera_fb_t *frameBuffer, TfLiteTensor *inputTensor, float scale, int zeroPoint) {
    assert(frameBuffer->format == PIXFORMAT_RGB565);

    int pixelIndex = 0;
    int offsetX = (frameBuffer->width - IMAGE_WIDTH) / 2;
    int offsetY = (frameBuffer->height - IMAGE_HEIGHT) / 2;

    int adjustedZeroPoint = static_cast<int>(floor(zeroPoint));

    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            int position = (offsetY + y) * frameBuffer->width + offsetX + x;
            uint16_t color = ((uint16_t *)frameBuffer->buf)[position];
            uint32_t rgb = convertRGB565toRGB888(color);

            if (inputTensor->type == kTfLiteUInt8) {
                uint8_t *imageData = inputTensor->data.uint8;
                imageData[pixelIndex * 3 + 0] = (uint8_t)((((rgb >> 16) & 0xFF) / 255.0f) / scale + adjustedZeroPoint);
                imageData[pixelIndex * 3 + 1] = (uint8_t)((((rgb >> 8) & 0xFF) / 255.0f) / scale + adjustedZeroPoint);
                imageData[pixelIndex * 3 + 2] = (uint8_t)(((rgb & 0xFF) / 255.0f) / scale + adjustedZeroPoint);
            } else {
                float *imageData = inputTensor->data.f;
                imageData[pixelIndex * 3 + 0] = ((rgb >> 16) & 0xFF) / 255.0f;
                imageData[pixelIndex * 3 + 1] = ((rgb >> 8) & 0xFF) / 255.0f;
                imageData[pixelIndex * 3 + 2] = (rgb & 0xFF) / 255.0f;
            }

            pixelIndex++;
        }
    }

    return 0;
}

// Calculate Intersection over Union (IoU)
float calculateIoU(BoundingBox &boxA, BoundingBox &boxB) {
    float x1 = std::max(boxA.centerX - boxA.boxWidth / 2, boxB.centerX - boxB.boxWidth / 2);
    float y1 = std::max(boxA.centerY - boxA.boxHeight / 2, boxB.centerY - boxB.boxHeight / 2);
    float x2 = std::min(boxA.centerX + boxA.boxWidth / 2, boxB.centerX + boxB.boxWidth / 2);
    float y2 = std::min(boxA.centerY + boxA.boxHeight / 2, boxB.centerY + boxB.boxHeight / 2);

    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float areaA = boxA.boxWidth * boxA.boxHeight;
    float areaB = boxB.boxWidth * boxB.boxHeight;

    return intersection / (areaA + areaB - intersection);
}

// Perform Non-Maximum Suppression
std::vector<BoundingBox> performNMS(std::vector<BoundingBox> &boxes, float iouThreshold) {
    std::vector<BoundingBox> selectedBoxes;

    std::sort(boxes.begin(), boxes.end(), [](BoundingBox &a, BoundingBox &b) {
        return a.confidence > b.confidence;
    });

    std::vector<bool> isSuppressed(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); i++) {
        if (isSuppressed[i]) continue;

        selectedBoxes.push_back(boxes[i]);

        for (size_t j = i + 1; j < boxes.size(); j++) {
            if (calculateIoU(boxes[i], boxes[j]) > iouThreshold) {
                isSuppressed[j] = true;
            }
        }
    }

    return selectedBoxes;
}

// Setup function
void setup() {
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

    Serial.begin(115200);

    camera_config_t cameraConfig;
    cameraConfig.ledc_channel = LEDC_CHANNEL_0;
    cameraConfig.ledc_timer = LEDC_TIMER_0;
    cameraConfig.pin_d0 = Y2_GPIO_NUM;
    cameraConfig.pin_d1 = Y3_GPIO_NUM;
    cameraConfig.pin_d2 = Y4_GPIO_NUM;
    cameraConfig.pin_d3 = Y5_GPIO_NUM;
    cameraConfig.pin_d4 = Y6_GPIO_NUM;
    cameraConfig.pin_d5 = Y7_GPIO_NUM;
    cameraConfig.pin_d6 = Y8_GPIO_NUM;
    cameraConfig.pin_d7 = Y9_GPIO_NUM;
    cameraConfig.pin_xclk = XCLK_GPIO_NUM;
    cameraConfig.pin_pclk = PCLK_GPIO_NUM;
    cameraConfig.pin_vsync = VSYNC_GPIO_NUM;
    cameraConfig.pin_href = HREF_GPIO_NUM;
    cameraConfig.pin_sccb_sda = SIOD_GPIO_NUM;
    cameraConfig.pin_sccb_scl = SIOC_GPIO_NUM;
    cameraConfig.pin_pwdn = PWDN_GPIO_NUM;
    cameraConfig.pin_reset = RESET_GPIO_NUM;
    cameraConfig.xclk_freq_hz = 20000000;
    cameraConfig.frame_size = FRAMESIZE_SVGA;
    cameraConfig.pixel_format = PIXFORMAT_RGB565;
    cameraConfig.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    cameraConfig.fb_location = CAMERA_FB_IN_PSRAM;
    cameraConfig.jpeg_quality = 12;
    cameraConfig.fb_count = 1;

    pinMode(LED_PIN, OUTPUT);

    esp_err_t error = esp_camera_init(&cameraConfig);
    if (error != ESP_OK) {
        Serial.printf("Camera initialization failed with error 0x%x", error);
        return;
    }

    neural_net = new NeuralNetwork();
}

// Main loop
void loop() {
    camera_fb_t *frameBuffer = esp_camera_fb_get();
    if (!frameBuffer) {
        Serial.println("Camera capture failed");
        return;
    }

    uint64_t preprocessStart, inferenceStart, postprocessStart;
    uint64_t preprocessDuration, inferenceDuration, postprocessDuration;

    TfLiteTensor *inputTensor = neural_net->getInput();
    float scale = inputTensor->params.scale;
    int zeroPoint = inputTensor->params.zero_point;

    preprocessStart = esp_timer_get_time();
    processImage(frameBuffer, inputTensor, scale, zeroPoint);
    preprocessDuration = esp_timer_get_time() - preprocessStart;

    inferenceStart = esp_timer_get_time();
    neural_net->predict();
    inferenceDuration = esp_timer_get_time() - inferenceStart;
    Serial.printf("Preprocessing: %llu ms, Inference: %llu ms\n", preprocessDuration / 1000, inferenceDuration / 1000);

    TfLiteTensor *outputTensor = neural_net->getOutput();
    float *outputData = nullptr;

    postprocessStart = esp_timer_get_time();
    int adjustedOutputZeroPoint = static_cast<int>(ceil(outputTensor->params.zero_point));

    if (outputTensor->type == kTfLiteUInt8) {
        outputData = new float[outputTensor->bytes / sizeof(uint8_t)];
        for (int i = 0; i < outputTensor->bytes; i++) {
            outputData[i] = (outputTensor->data.uint8[i] - adjustedOutputZeroPoint) * outputTensor->params.scale;
        }
    } else {
        outputData = outputTensor->data.f;
    }

    int numDetections = outputTensor->dims->data[1];
    int detectionSize = outputTensor->dims->data[2];

    std::vector<BoundingBox> detectedBoxes;

    for (int i = 0; i < numDetections; i++) {
        float confidence = outputData[i * detectionSize + 4];
        int classID = static_cast<int>(outputData[i * detectionSize + 5]);

        if (confidence > CONFIDENCE_THRESHOLD && classID == TARGET_CLASS_INDEX) {
            BoundingBox box;
            box.centerX = outputData[i * detectionSize + 0] * IMAGE_WIDTH;
            box.centerY = outputData[i * detectionSize + 1] * IMAGE_HEIGHT;
            box.boxWidth = outputData[i * detectionSize + 2] * IMAGE_WIDTH;
            box.boxHeight = outputData[i * detectionSize + 3] * IMAGE_HEIGHT;
            box.confidence = confidence;
            box.classID = classID;

            detectedBoxes.push_back(box);
        }
    }

    float iouThreshold = 0.5;
    std::vector<BoundingBox> vehicleBoxes = performNMS(detectedBoxes, iouThreshold);

    postprocessDuration = esp_timer_get_time() - postprocessStart;
    Serial.printf("Post-processing: %llu ms\n", postprocessDuration / 1000);

    if (!vehicleBoxes.empty()) {
        int vehicleCount = 0;
        for (const auto &box : vehicleBoxes) {
            if (box.boxWidth != 0 && box.boxHeight != 0) {
                vehicleCount++;
                Serial.printf("Detected Vehicle at: [x=%.2f, y=%.2f, w=%.2f, h=%.2f, conf=%.2f]\n",
                              box.centerX, box.centerY, box.boxWidth, box.boxHeight, box.confidence);
            }
        }
        if (vehicleCount) Serial.printf("Detected %d vehicles\n", vehicleCount);
    } else {
        Serial.println("No Vehicles Detected");
    }

    esp_camera_fb_return(frameBuffer);
}