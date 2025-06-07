import sys
import ctypes
import time
import numpy as np
import cv2
import os
from typing import List, Tuple

class MatchResult(ctypes.Structure):
    """匹配结果结构体"""
    _fields_ = [
        ('leftTopX', ctypes.c_double),
        ('leftTopY', ctypes.c_double),
        ('leftBottomX', ctypes.c_double),
        ('leftBottomY', ctypes.c_double),
        ('rightTopX', ctypes.c_double),
        ('rightTopY', ctypes.c_double),
        ('rightBottomX', ctypes.c_double),
        ('rightBottomY', ctypes.c_double),
        ('centerX', ctypes.c_double),
        ('centerY', ctypes.c_double),
        ('angle', ctypes.c_double),
        ('score', ctypes.c_double)
    ]

class FasterTemplateMatch:
    """快速模板匹配服务类"""

    def __init__(self, dll_path: str, max_count: int = 1, score_threshold: float = 0.9,
                 iou_threshold: float = 0.4, angle: float = 0, min_area: float = 25):
        """
        初始化模板匹配器

        Args:
            dll_path: 模板匹配库路径
            max_count: 最大匹配结果数量 (0表示返回所有匹配)
            score_threshold: 分数阈值
            iou_threshold: IOU阈值
            angle: 角度
            min_area: 最小区域面积
        """
        self._load_library(dll_path)

        # 当max_count为0时，我们使用一个较大的默认值，实际结果数量由match函数返回
        self._max_count = max_count if max_count > 0 else 1000  # 假设最多1000个匹配结果
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.angle = angle
        self.min_area = min_area

        self._initialize_matcher()
        self.results = (MatchResult * self._max_count)()

    def _load_library(self, dll_path: str):
        """加载动态链接库"""
        try:
            self.lib = ctypes.CDLL(dll_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load library: {e}")

        # 设置函数参数类型
        self.lib.matcher.argtypes = [
            ctypes.c_int, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float
        ]
        self.lib.matcher.restype = ctypes.c_void_p

        self.lib.setTemplate.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

        self.lib.match.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(MatchResult), ctypes.c_int
        ]
        self.lib.match.restype = ctypes.c_int  # 确保返回匹配数量

    def _initialize_matcher(self):
        """初始化匹配器实例"""
        # 传给底层库的max_count使用实际值，0会被视为无效
        actual_max_count = self._max_count if self._max_count > 0 else 1000
        self.matcher = self.lib.matcher(
            actual_max_count, self.score_threshold,
            self.iou_threshold, self.angle, self.min_area
        )
        if not self.matcher:
            raise RuntimeError("Failed to create matcher instance")

    def set_template(self, template_image: np.ndarray) -> bool:
        """
        设置模板图像

        Args:
            template_image: 模板图像 (灰度图或彩色图)

        Returns:
            bool: 是否设置成功
        """
        if template_image is None:
            raise ValueError("Template image cannot be None")

        # 转换为灰度图
        if template_image.ndim == 3:
            if template_image.shape[2] == 3:
                template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
            elif template_image.shape[2] == 1:
                template_image = template_image[:, :, 0]
            else:
                raise ValueError("Invalid template image shape")

        height, width = template_image.shape[:2]
        channels = 1
        data = template_image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

        return self.lib.setTemplate(self.matcher, data, width, height, channels)

    def match(self, target_image: np.ndarray) -> Tuple[int, List[MatchResult]]:
        """
        在目标图像中匹配模板

        Args:
            target_image: 目标图像 (灰度图或彩色图)

        Returns:
            int: 匹配结果数量 (负数表示匹配失败)
        """
        if target_image is None:
            raise ValueError("Target image cannot be None")

        # 转换为灰度图
        if target_image.ndim == 3:
            if target_image.shape[2] == 3:
                target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
            elif target_image.shape[2] == 1:
                target_image = target_image[:, :, 0]
            else:
                raise ValueError("Invalid target image shape")

        height, width = target_image.shape[:2]
        channels = 1
        data = target_image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

        matches_count = self.lib.match(
            self.matcher, data, width, height, channels,
            self.results, self._max_count
        )

        # 立即复制结果，避免被后续调用覆盖
        result_list = [MatchResult(
            leftTopX=self.results[i].leftTopX,
            leftTopY=self.results[i].leftTopY,
            leftBottomX=self.results[i].leftBottomX,
            leftBottomY=self.results[i].leftBottomY,
            rightTopX=self.results[i].rightTopX,
            rightTopY=self.results[i].rightTopY,
            rightBottomX=self.results[i].rightBottomX,
            rightBottomY=self.results[i].rightBottomY,
            centerX=self.results[i].centerX,
            centerY=self.results[i].centerY,
            angle=self.results[i].angle,
            score=self.results[i].score
        ) for i in range(min(matches_count, self._max_count)) if matches_count > 0]

        return matches_count, result_list

    def visualize_match_result(self, target_image: np.ndarray, results: List[MatchResult], debug: bool = False) -> np.ndarray:
        """
        可视化匹配结果
        """
        if not results:
            return target_image.copy()

        result_image = target_image.copy()
        if debug:
            for result in results:
                if result.score > 0:
                    # 绘制匹配区域
                    points = np.array([
                        [result.leftTopX, result.leftTopY],
                        [result.leftBottomX, result.leftBottomY],
                        [result.rightBottomX, result.rightBottomY],
                        [result.rightTopX, result.rightTopY]
                    ], np.int32)

                    cv2.polylines(result_image, [points], True, (0, 255, 0), 2)

                    # 添加分数文本
                    cv2.putText(
                        result_image, f"Score: {result.score:.2f}",
                        (int(result.leftTopX), int(result.leftTopY) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )

        return result_image

def load_templates(template_paths: List[str]) -> List[Tuple[str, np.ndarray]]:
    """
    加载指定的模板图像数组
    """
    templates = []
    for path in template_paths:
        if not os.path.exists(path):
            print(f"Warning: Template file {path} not found")
            continue
        template_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if template_img is not None:
            template_name = os.path.basename(path)
            templates.append((template_name, template_img))
        else:
            print(f"Warning: Failed to load template {path}")

    return templates

def visualize_all_matches(target_image: np.ndarray,
                         all_results: List[Tuple[str, List[MatchResult]]],
                         debug: bool = True) -> np.ndarray:
    """
    在目标图像上绘制所有模板的匹配结果

    Args:
        target_image: 目标图像
        all_results: 包含所有模板匹配结果的列表，每个元素是 (模板名称, 匹配结果列表)
        debug: 是否显示分数和边框

    Returns:
        绘制了所有匹配结果的图像
    """
    if target_image is None or target_image.size == 0:
        raise ValueError("Target image is invalid or empty")

    # 确保目标图像是彩色图以支持彩色绘制
    result_image = target_image.copy()
    if result_image.ndim == 2:
        result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)

    # 为不同模板分配不同颜色
    colors = [
        (0, 255, 0),  # 绿色
        (0, 0, 255),  # 红色
        (255, 0, 0),  # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255)  # 紫色
    ]

    if debug:
        print(f"Image dimensions: {result_image.shape} (height, width, channels)")

    for idx, (template_name, results) in enumerate(all_results):
        if not results:
            if debug:
                print(f"No results for template {template_name}")
            continue

        color = colors[idx % len(colors)]
        if debug:
            print(f"Template {template_name}: {len(results)} matches, using color {color}")

        for result_idx, result in enumerate(results):
            if result.score <= 0:
                if debug:
                    print(f"Skipping match {result_idx} for template {template_name} due to score {result.score}")
                continue

            # 绘制匹配框
            points = np.array([
                [result.leftTopX, result.leftTopY],
                [result.leftBottomX, result.leftBottomY],
                [result.rightBottomX, result.rightBottomY],
                [result.rightTopX, result.rightTopY]
            ], np.int32)
            cv2.polylines(result_image, [points], True, color, 2)

            # 计算文本及其背景框
            text = f"{template_name}: {result.score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_width, text_height = text_size

            # 图像尺寸
            img_height, img_width = result_image.shape[:2]

            # 初始尝试将文本放在左上方 (leftTopX - 60, leftTopY - 10)
            text_x = int(result.leftTopX) - 60
            text_y = int(result.leftTopY) - 10

            # 检查是否超出边界，动态调整位置
            if text_x < 0:
                # 移到右上方
                text_x = int(result.rightTopX) + 10
            if text_y < text_height:
                # 移到框内或下方
                text_y = int(result.leftTopY) + text_height + 10
            if text_x + text_width > img_width:
                # 确保不超出右边界
                text_x = img_width - text_width - 10
            if text_y > img_height:
                # 确保不超出下边界
                text_y = img_height - 10

            # 绘制文本背景矩形以提高可读性
            bg_top_left = (text_x, text_y - text_height)
            bg_bottom_right = (text_x + text_width, text_y + baseline)
            cv2.rectangle(result_image, bg_top_left, bg_bottom_right, (0, 0, 0), -1)  # 黑色背景

            # 绘制文本
            cv2.putText(result_image, text, (text_x, text_y), font, font_scale, color, thickness)

    return result_image

if __name__ == "__main__":
    try:
        start_time = time.time()

        # 加载目标图像
        target_image = cv2.imread('./assets/match/target.png')
        if target_image is None:
            raise FileNotFoundError("Target image not found")

        # 指定模板图像路径数组
        template_paths = [
            './assets/match/template.png',
        ]

        # 加载所有模板
        templates = load_templates(template_paths)
        if not templates:
            raise ValueError("No valid templates found")

        # 存储所有模板的匹配结果
        all_results = []
        total_matches = 0

        # 为每个模板创建独立的匹配器实例
        for template_name, template_image in templates:
            matcher = FasterTemplateMatch(
                dll_path='libtemplatematching_ctype.so',
                max_count=1,
                score_threshold=0.9,
                iou_threshold=0.3,
                angle=0,
                min_area=256
            )
            matcher.set_template(template_image)
            matches_count, results = matcher.match(target_image)

            if matches_count > 0:
                all_results.append((template_name, results))
                total_matches += matches_count
                print(f"\n模板 {template_name}: 找到 {matches_count} 个匹配结果")
                for i, result in enumerate(results):
                    print(f"结果 {i + 1}: 分数={result.score:.2f}, 中心点=({result.centerX:.1f}, {result.centerY:.1f})")
            else:
                print(f"\n模板 {template_name}: 未找到匹配结果")

        print(f"\n总匹配耗时: {time.time() - start_time:.6f} 秒")
        print(f"总共找到 {total_matches} 个匹配结果")
        print(all_results)

        # 可视化所有匹配结果
        # if total_matches > 0:
        #     result_image = visualize_all_matches(target_image, all_results, debug=True)
        #     cv2.imshow('Target Image', target_image)
        #     cv2.imshow('Matching Result', result_image)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        #     cv2.imwrite('./assets/match/template.png', result_image)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)