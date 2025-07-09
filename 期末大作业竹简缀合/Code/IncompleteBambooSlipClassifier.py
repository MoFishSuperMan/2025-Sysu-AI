import cv2
import numpy as np

class IncompleteBambooSlipClassifier:
    def __init__(self):
        # 轮廓分析参数
        self.contour_area_threshold = 1000  # 最小轮廓面积
        # 角点检测参数
        self.corner_defect_threshold = 0.1  # 角点缺陷阈值
        self.corner_angle_threshold = 60    # 角点角度阈值
        # 边缘检测参数
        self.edge_straightness_threshold = 0.8  # 边缘直线度阈值
        
    def preprocess_image(self, image):
        """图像预处理 - 增强角点检测能力"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 强对比度增强
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16, 16))
        enhanced = clahe.apply(gray)
        # 边缘保留滤波
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        # 二值化
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 形态学操作 - 强化角点特征
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
        return filtered, closed
    
    def detect_main_contour(self, binary_image):
        """检测竹简主体轮廓"""
        # 查找轮廓
        contours, _ = cv2.findContours(
            binary_image, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        # 过滤掉小轮廓
        filtered_contours = [
            cnt for cnt in contours 
            if cv2.contourArea(cnt) > self.contour_area_threshold
        ]
        if not filtered_contours:
            return None
        # 找到最大轮廓（竹简主体）
        return max(filtered_contours, key=cv2.contourArea)
    
    def analyze_corners(self, contour):
        """分析竹简四个角点是否完整"""
        # 计算凸包
        hull = cv2.convexHull(contour, returnPoints=False)
        # 计算凸包缺陷
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            # 没有检测到缺陷，所有角点完整
            return {'top_left': True, 'top_right': True, 
                    'bottom_left': True, 'bottom_right': True}
        # 获取轮廓点
        points = contour.squeeze()
        if points.ndim != 2:
            return {'top_left': False, 'top_right': False, 
                    'bottom_left': False, 'bottom_right': False}
        # 计算轮廓的边界
        x, y, w, h = cv2.boundingRect(contour)
        # 定义四个角区域
        corner_regions = {
            'top_left': (x, y, x + w//4, y + h//4),
            'top_right': (x + 3*w//4, y, x + w, y + h//4),
            'bottom_left': (x, y + 3*h//4, x + w//4, y + h),
            'bottom_right': (x + 3*w//4, y + 3*h//4, x + w, y + h)
        }
        # 初始化角点状态
        corner_status = {corner: True for corner in corner_regions}
        # 分析每个缺陷点
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(points[s])
            end = tuple(points[e])
            far = tuple(points[f])
            # 计算角度
            a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            # 使用余弦定理计算角度
            if b * c == 0:
                continue
            angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) * 180 / np.pi
            # 如果角度小于阈值，可能是角点缺陷
            if angle < self.corner_angle_threshold:
                # 检查缺陷点所在的角区域
                for corner, (x1, y1, x2, y2) in corner_regions.items():
                    if x1 <= far[0] <= x2 and y1 <= far[1] <= y2:
                        corner_status[corner] = False
                        break
        return corner_status
    
    def analyze_edges(self, contour):
        """分析竹简边缘直线度"""
        # 获取轮廓点
        points = contour.squeeze()
        if points.ndim != 2 or len(points) < 4:
            return {}
        # 计算轮廓中心
        center = np.mean(points, axis=0)
        # 划分四个边缘区域
        edge_regions = {
            'top': [p for p in points if p[1] < center[1] - 10],
            'bottom': [p for p in points if p[1] > center[1] + 10],
            'left': [p for p in points if p[0] < center[0] - 10],
            'right': [p for p in points if p[0] > center[0] + 10]
        }
        edge_straightness = {}
        for edge, points in edge_regions.items():
            if len(points) < 2:
                edge_straightness[edge] = 0.0
                continue
            # 拟合直线
            if edge in ['top', 'bottom']:
                # 水平边缘
                y_vals = [p[1] for p in points]
                mean_y = np.mean(y_vals)
                std_y = np.std(y_vals)
                # 直线度 = 1 - 标准化标准差
                straightness = 1 - min(std_y / 10.0, 1.0)
            else:
                # 垂直边缘
                x_vals = [p[0] for p in points]
                mean_x = np.mean(x_vals)
                std_x = np.std(x_vals)
                straightness = 1 - min(std_x / 10.0, 1.0)
            edge_straightness[edge] = straightness
        return edge_straightness
    
    def is_complete_slip(self, corner_status, edge_straightness):
        """判断竹简是否完整 - 基于角点完整性和边缘直线度"""
        # 检查四个角是否完整
        missing_corners = sum(1 for status in corner_status.values() if not status)
        # 如果有任何一个角缺失，直接判断为残断简
        if missing_corners > 0:
            return False
        # 检查边缘直线度
        for edge, straightness in edge_straightness.items():
            # 左右边缘必须非常直
            if edge in ['left', 'right'] and straightness < self.edge_straightness_threshold:
                return False
            # 上下边缘允许一定的弯曲
            if edge in ['top', 'bottom'] and straightness < self.edge_straightness_threshold - 0.2:
                return False
        return True

    def analyze_slip(self, image):
        """分析竹简图像并返回判断结果"""
        try:
            # 图像预处理
            enhanced, cleaned = self.preprocess_image(image)
            # 轮廓检测
            contour = self.detect_main_contour(cleaned)
            if contour is None:
                # 没有检测到轮廓，可能是完整的竹简
                return True, "未检测到轮廓，可能为完整简"
            # 角点分析
            corner_status = self.analyze_corners(contour)
            # 边缘直线度分析
            edge_straightness = self.analyze_edges(contour)
            # 完整性判断
            is_complete = self.is_complete_slip(corner_status, edge_straightness)
            return is_complete
        except Exception as e:
            # 出错时默认返回完整简
            return True, f"分析出错，默认完整简: {str(e)}"

# 测试用例
if __name__ == "__main__":
    analyzer = IncompleteBambooSlipClassifier()
    # 读取图像
    image_path1 = "0001_a.jpeg"
    image_path2 = '2327.jpg'
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    if image1 is None:
        print(f"无法读取图像: {image_path1}")
    else:
        # 分析竹简
        is_complete1 = analyzer.analyze_slip(image1)
        is_complete2 = analyzer.analyze_slip(image2)
        # 输出结果
        print(f"竹简完整性判断: {'完整简' if is_complete1 else '残断简'}")
        print(f"竹简完整性判断: {'完整简' if is_complete2 else '残断简'}")