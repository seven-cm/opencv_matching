#include <stdlib.h>
#include <math.h>
#include <filesystem>  // C++17 标准库中的文件系统库
namespace fs = std::filesystem;

#include "matcher.h"
#ifdef _WIN32
#include "windows.h"
#else
#include <dlfcn.h>
#endif // _WIN32


// 动态库函数指针
typedef template_matching::Matcher* (*InitMD)(const template_matching::MatcherParam&);

int main(int argc, char** argv)
{
    // 检查参数
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <template_image.png> <target_image.png>" << std::endl;
        return -1;
    }

    // 匹配器参数
    template_matching::MatcherParam param;
    param.angle = 0;
    param.iouThreshold = 0;
    param.matcherType = template_matching::MatcherType::PATTERN;
    param.maxCount = 1;
    param.minArea = 256;
    param.scoreThreshold = 0.5;
    
    // 匹配结果
    std::vector<template_matching::MatchResult> matchResults;

#ifdef _WIN32
    // windows 动态加载dll
    HINSTANCE handle = nullptr;
    handle = LoadLibrary("templatematching.dll");
    if (handle == nullptr)
    {
        std::cerr << "Error : failed to load templatematching.dll!" << std::endl;
        return -2;
    }

    // 获取动态库内的函数
    InitMD myGetMatcher;
    myGetMatcher = (InitMD)GetProcAddress(handle, "GetMatcher");
#else
    // linux 动态加载dll
    void* handle = nullptr;
    handle = dlopen("libtemplatematching.so", RTLD_LAZY); 
    if (handle == nullptr)
    {
        char *dlopenError = dlerror();
        if (dlopenError != nullptr)
        {
            std::cerr << "Error : " << dlopenError << std::endl;
        }
        std::cerr << "Error : failed to load libtemplatematching.so!" << std::endl;
        return -2;
    }

    // 获取动态库内的函数
    InitMD myGetMatcher;
    myGetMatcher = (InitMD)dlsym(handle, "GetMatcher");
#endif // _WIN32

    // 初始化匹配器
    std::cout << "initiating..." << std::endl;
    template_matching::Matcher* matcher = myGetMatcher(param);
    std::cout << "initialized." << std::endl;

    if (matcher)
    {
        matcher->setMetricsTime(true);

        std::chrono::steady_clock::time_point startTime, endTime;
        std::chrono::duration<double> timeUse;

        // 读取模板图像
        cv::Mat templateImg = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
        if (templateImg.empty()) {
            std::cerr << "Error: failed to load template image: " << argv[1] << std::endl;
            return -1;
        }

        // 读取目标图像
        cv::Mat targetImg = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
        if (targetImg.empty()) {
            std::cerr << "Error: failed to load target image: " << argv[2] << std::endl;
            return -1;
        }

        // 设置模板
        matcher->setTemplate(templateImg);

        startTime = std::chrono::steady_clock::now();

        // 执行匹配，结果保存在 matchResults
        matcher->match(targetImg, matchResults);

        endTime = std::chrono::steady_clock::now();
        timeUse = std::chrono::duration_cast<std::chrono::duration<double>>(endTime - startTime);
        std::cout << "match time: " << timeUse.count() << "s." << std::endl;

        // 显示匹配结果
        if (!matchResults.empty()) {
            std::cout << "Found " << matchResults.size() << " matches:" << std::endl;
            for (size_t i = 0; i < matchResults.size(); i++) {
                std::cout << "Match " << i+1 << ": Score=" << matchResults[i].Score 
                          << ", Position=(" << matchResults[i].LeftTop.x << "," << matchResults[i].LeftTop.y << ")" 
                          << " to (" << matchResults[i].RightBottom.x << "," << matchResults[i].RightBottom.y << ")" << std::endl;
            }

            // 可视化结果 ** 非docker请取消注释**
            // cv::Mat colorTarget;
            // cv::cvtColor(targetImg, colorTarget, cv::COLOR_GRAY2BGR);

            // for (const auto& result : matchResults) {
            //     cv::Point2i temp;
            //     std::vector<cv::Point2i> pts;
            //     temp.x = std::round(result.LeftTop.x);
            //     temp.y = std::round(result.LeftTop.y);
            //     pts.push_back(temp);
            //     temp.x = std::round(result.RightTop.x);
            //     temp.y = std::round(result.RightTop.y);
            //     pts.push_back(temp);
            //     temp.x = std::round(result.RightBottom.x);
            //     temp.y = std::round(result.RightBottom.y);
            //     pts.push_back(temp);
            //     temp.x = std::round(result.LeftBottom.x);
            //     temp.y = std::round(result.LeftBottom.y);
            //     pts.push_back(temp);

            //     cv::polylines(colorTarget, pts, true, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            // }

            // // 显示图像
            // cv::imshow("Template", templateImg);
            // cv::imshow("Target with matches", colorTarget);
            // cv::waitKey(0);
        } else {
            std::cout << "No matches found." << std::endl;
        }

        delete matcher;
    }
    else
    {
        std::cerr << "Error: failed to get matcher." << std::endl;
    }

    // 释放动态库
    if (handle != nullptr)
    {
#ifdef _WIN32
        FreeLibrary(handle);
#else
        dlclose(handle);
#endif // _WIN32
    }

    return 0;
}