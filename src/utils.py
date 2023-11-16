import copy
import gc
import json

import cv2
import numpy as np
from .config import shooting_result, HAND_CONNECTIONS, POSE_CONNECTIONS, preelbowAngle, prekneeAngle, preankleAngle, preshoulderAngle, prewristAngle, preelbowCoord, prekneeCoord, preankleCoord, preshoulderCoord, prewristCoord
from PIL import Image, ImageDraw, ImageFont


def fit_func(x, a, b, c):
    return a*(x ** 2) + b * x + c


def distance(x, y):
    return ((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2) ** (1/2)


# 计算角度
# 锐角
def calculateAngle2(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return round(np.degrees(angle), 2)


def calculateAngle(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    # 计算叉积的z分量
    cross_product_z = ba[0] * bc[1] - ba[1] * bc[0]

    # 如果z分量为负，则角度在ba的顺时针方向
    if cross_product_z < 0:
        angle = 2 * np.pi - angle

    return round(np.degrees(angle), 2)


# 从姿态结果中计算角度
def getAngleFromDatum(results, h, w):
    global prewristAngle, prewristCoord, prefingerCoord
    # 屁股
    hipX, hipY = w * \
        results.pose_landmarks.landmark[24].x, h * \
        results.pose_landmarks.landmark[24].y
    # 膝盖
    kneeX, kneeY = w * \
        results.pose_landmarks.landmark[26].x, h * \
        results.pose_landmarks.landmark[26].y
    # 脚踝
    ankleX, ankleY = w * \
        results.pose_landmarks.landmark[28].x, h * \
        results.pose_landmarks.landmark[28].y
    # 脚尖
    toeX, toeY = w * \
        results.pose_landmarks.landmark[32].x, h * \
        results.pose_landmarks.landmark[32].y

    # 肩膀
    shoulderX, shoulderY = w * \
        results.pose_landmarks.landmark[12].x, h * \
        results.pose_landmarks.landmark[12].y
    # 手肘
    elbowX, elbowY = w * \
        results.pose_landmarks.landmark[14].x, h * \
        results.pose_landmarks.landmark[14].y
    # 手腕
    wristX, wristY = w * \
        results.pose_landmarks.landmark[16].x, h * \
        results.pose_landmarks.landmark[16].y

    kneeAngle = calculateAngle2(np.array([hipX, hipY]), np.array(
        [kneeX, kneeY]), np.array([ankleX, ankleY]))
    elbowAngle = calculateAngle2(np.array([shoulderX, shoulderY]), np.array(
        [elbowX, elbowY]), np.array([wristX, wristY]))
    ankleAngle = calculateAngle2(np.array([kneeX, kneeY]), np.array(
        [ankleX, ankleY]), np.array([toeX, toeY]))
    shoulderAngle = calculateAngle2(np.array([elbowX, elbowY]), np.array(
        [shoulderX, shoulderY]), np.array([hipX, hipY]))

    elbowCoord = np.array([int(elbowX), int(elbowY)])
    kneeCoord = np.array([int(kneeX), int(kneeY)])
    ankleCoord = np.array([int(ankleX), int(ankleY)])
    shoulderCoord = np.array([int(shoulderX), int(shoulderY)])
    hipCoord = np.array([int(hipX), int(hipY)])

    # 若检测到手部就更新手部信息
    if (results.right_hand_landmarks):
        # 指尖
        fingerX, fingerY = w * results.right_hand_landmarks.landmark[12].x, h * results.right_hand_landmarks.landmark[
            12].y
        wristAngle = calculateAngle(np.array([fingerX, fingerY]), np.array(
            [wristX, wristY]), np.array([elbowX, elbowY]))
        wristCoord = np.array([int(wristX), int(wristY)])
        fingerCoord = np.array([int(fingerX), int(fingerY)])
        return elbowAngle, kneeAngle, ankleAngle, shoulderAngle, wristAngle, elbowCoord, kneeCoord, ankleCoord, shoulderCoord, wristCoord, hipCoord, fingerCoord
    # 若未检测到手部就保持手部角度信息不变
    return elbowAngle, kneeAngle, ankleAngle, shoulderAngle, prewristAngle, elbowCoord, kneeCoord, ankleCoord, shoulderCoord, prewristCoord, hipCoord, prefingerCoord


def crop_and_resize_to_match(image, wristCoord, target_width):
    height, width = image.shape[:2]
    # 确定理论裁剪坐标
    xmin = int(wristCoord[0]) - 200
    ymin = int(wristCoord[1]) - 200
    xmax = int(wristCoord[0]) + 200
    ymax = int(wristCoord[1]) + 200
    # 如果裁剪区域超出图像左边界，进行调整
    if xmin < 0:
        xmax -= xmin  # 调整右边界
        xmin = 0
        xmax = 400
    # 如果裁剪区域超出图像上边界，进行调整
    if ymin < 0:
        ymax -= ymin  # 调整下边界
        ymin = 0
        ymax = 400
    # 如果裁剪区域超出图像右边界，进行调整
    if xmax > width:
        xmin -= (xmax - width)  # 调整左边界
        xmax = width
        xmin = width - 400
    # 如果裁剪区域超出图像下边界，进行调整
    if ymax > height:
        ymin -= (ymax - height)  # 调整上边界
        ymax = height
        ymin = height - 400
    # 使用调整后的坐标进行裁剪
    cropped_img = image[ymin:ymax, xmin:xmax]
    # 调整高度以保持原始的宽高比
    # original_height, original_width = cropped_img.shape[:2]
    # aspect_ratio = original_height / original_width
    # new_height = int(round(target_width * aspect_ratio))  # 使用round确保new_height是整数
    # # 调整图像大小以匹配目标宽度
    # resized_img = cv2.resize(cropped_img, (int(target_width), new_height), interpolation=cv2.INTER_CUBIC)
    return cropped_img


def paste_on_white_background(resized_img, bg_width, bg_height):
    # 创建一个白色背景
    white_bg = np.ones((int(bg_height), int(bg_width), 3),
                       dtype=np.uint8) * 255
    white_bg[0:resized_img.shape[0], 0:resized_img.shape[1]] = resized_img
    return white_bg


def draw_connections(image, image2, landmarks, connections, color=(0, 255, 0), thickness=3):
    for connection in connections:
        start_point = connection[0]
        end_point = connection[1]
        # 获取关键点的坐标
        x1, y1 = int(landmarks[start_point].x * image.shape[1]
                     ), int(landmarks[start_point].y * image.shape[0])
        x2, y2 = int(landmarks[end_point].x * image.shape[1]
                     ), int(landmarks[end_point].y * image.shape[0])
        # 绘制线条
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)
        cv2.line(image2, (x1, y1), (x2, y2), color, thickness)
        # 绘制关键点
    for idx, landmark in enumerate(landmarks):
        if idx not in [18, 20, 22, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 17, 19, 21]:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
            cv2.circle(image2, (x, y), 3, (255, 0, 0), -1)
#     绘制中心垂线
    midpoint = (int((landmarks[12].x * image.shape[1]+landmarks[24].x * image.shape[1])/2),
                int((landmarks[12].y*image.shape[0]+landmarks[24].y*image.shape[0])/2))
    # cv2.line(image, (midpoint[0]-200, midpoint[1]), (midpoint[0]+200, midpoint[1]), (0, 0, 255), 2)
    cv2.line(image, (midpoint[0], midpoint[1]-400),
             (midpoint[0], midpoint[1]+400), (0, 0, 255), 2)
    # cv2.line(image2, (midpoint[0] - 200, midpoint[1]), (midpoint[0] + 200, midpoint[1]), (0, 0, 255), 2)
    cv2.line(image2, (midpoint[0], midpoint[1] - 400),
             (midpoint[0], midpoint[1] + 400), (0, 0, 255), 2)
#     计算竖直夹角
#     # 定义两个向量
#     v1 = np.array([0, 1])
#     v2 = np.array([landmarks[24].x * image.shape[1] - landmarks[12].x * image.shape[1], landmarks[24].y * image.shape[0] - landmarks[12].y * image.shape[0]])
#     # 计算两个向量的点积
#     dot_product = np.dot(v1, v2)
#     # 计算两个向量的模
#     norm_v1 = np.linalg.norm(v1)
#     norm_v2 = np.linalg.norm(v2)
#     # 计算夹角的余弦值
#     cos_theta = dot_product / (norm_v1 * norm_v2)
#     # 使用arccos函数计算夹角（以弧度为单位）
#     theta_rad = np.arccos(cos_theta)
    # 将弧度转换为度
    theta_deg = calculateAngle2(np.array([midpoint[0], midpoint[1]-200]), np.array(
        [midpoint[0], midpoint[1]]), np.array([landmarks[12].x * image.shape[1], landmarks[12].y * image.shape[0]]))
    return theta_deg, midpoint

# 重新写的检测函数


def detect_pose(frame_index, img, width, height, mp_drawing, mp_holistic, holistic, yolo_model, pose, during_shooting, shot_result, shooting_pose, data_pose_xy):
    global shooting_result
    global preelbowAngle, prekneeAngle, preankleAngle, preshoulderAngle, prewristAngle, preelbowCoord, prekneeCoord, preankleCoord, preshoulderCoord, prewristCoord
    global POSE_CONNECTIONS, HAND_CONNECTIONS
    fingers = []
    resized_img = None
    if (shot_result['displayFrames'] > 0):
        shot_result['displayFrames'] -= 1
    if (shot_result['release_displayFrames'] > 0):
        shot_result['release_displayFrames'] -= 1
    if (shooting_pose['ball_in_hand']):
        shooting_pose['ballInHand_frames'] += 1
    # 初始化 BlazePose 对象
    target_width = width
    target_height = height
    frame = img.copy()  # frame是处理过的图像
    real_img = img.copy()
    # 图像灰度处理
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 高斯滤波降噪
    gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny算子
    canny = cv2.Canny(gaussian, 30, 100)
    # 阈值化处理
    ret, result = cv2.threshold(
        canny, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 将单通道的result转换为三通道的BGR图像
    result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    frame = result_bgr
    # 将两个三通道的图像堆叠起来
    alpha = 1.5  # 叠加的权重因子，可以调整该值来控制叠加的效果
    beta = 0  # 偏置值
    # 使用cv2.addWeighted()函数将frame与自己叠加
    frame = cv2.addWeighted(frame, alpha, frame, beta, 0)

    # # 定义新的宽度
    # new_width = int(target_width * 1.2)  # 例如，将宽度增加50%
    # # 使用cv2.resize()函数调整图像大小
    # frame = cv2.resize(img, (int(new_width), int(target_width)))
    # frame = np.ones((int(target_width), int(target_height), 3), dtype=np.uint8) * 255
    # 将 BGR 图像转换为 RGB 图像
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 使用 BlazePose 进行姿态估计
    try:
        results = holistic.process(rgb_image)
        handX, handY, handConf = target_width * results.pose_landmarks.landmark[16].x, target_height * results.pose_landmarks.landmark[16].y, \
            results.pose_landmarks.landmark[16].visibility
        fingers.append((int(target_width * results.right_hand_landmarks.landmark[4].x), int(
            target_height * results.right_hand_landmarks.landmark[4].y)))
        fingers.append((int(target_width * results.right_hand_landmarks.landmark[8].x),
                        int(target_height * results.right_hand_landmarks.landmark[8].y)))
        fingers.append((int(target_width * results.right_hand_landmarks.landmark[12].x),
                        int(target_height * results.right_hand_landmarks.landmark[12].y)))
        fingers.append((int(target_width * results.right_hand_landmarks.landmark[16].x),
                        int(target_height * results.right_hand_landmarks.landmark[16].y)))
        fingers.append((int(target_width * results.right_hand_landmarks.landmark[20].x),
                        int(target_height * results.right_hand_landmarks.landmark[20].y)))
        elbowAngle, kneeAngle, ankleAngle, shoulderAngle, wristAngle, elbowCoord, kneeCoord, ankleCoord, shoulderCoord, wristCoord, hipCoord, fingerCoord\
            = getAngleFromDatum(results, target_height, target_width)
        shooting_pose['shoulder_xy'].append(shoulderCoord)
        shooting_pose['hip_xy'].append(hipCoord)
        # 记该数组的第一个元素的y为脚踝的起始高度?
        shooting_pose['foot_xy'].append(ankleCoord)
        shooting_pose['wrist_xy'].append(wristCoord)
        shooting_pose['finger_xy'].append(fingerCoord)
        shooting_pose['elbow_xy'].append(elbowCoord)
        # data_pose_xy[str(round(frame_index, 2))] = [ankleCoord.tolist(), shoulderCoord.tolist(),
        #                                              hipCoord.tolist(), wristCoord.tolist(),
        #                                              fingerCoord.tolist()]
        prewristCoord = wristCoord
        preelbowAngle, prekneeAngle, preankleAngle, preshoulderAngle, prewristAngle = elbowAngle, kneeAngle, ankleAngle, shoulderAngle, wristAngle
        preelbowCoord, prekneeCoord, preankleCoord, preshoulderCoord, prewristCoord, prefingerCoord = elbowCoord, kneeCoord, ankleCoord, shoulderCoord, wristCoord, fingerCoord
        # 如果球不在手上了，检查当前脚踝高度，若与起始高度相差较小认为投篮动作结束
        if not shooting_pose['ball_in_hand']:
            if (abs(ankleCoord[1]-shooting_pose['foot_xy'][0][1]) < 7):
                shooting_pose['is_end'] = True
        # 记录膝盖角度最小值
        shooting_pose['knee_min_angle'] = min(
            shooting_pose['knee_min_angle'], kneeAngle)
        # 记录膝盖角度最大值
        shooting_pose['knee_max_angle'] = max(
            shooting_pose['knee_max_angle'], kneeAngle)
        # 记录当前帧id
        if (shooting_pose['knee_min_angle'] == kneeAngle):
            shooting_pose['knee_min_time'] = frame_index
        if (shooting_pose['knee_max_angle'] == kneeAngle):
            shooting_pose['knee_max_time'] = frame_index
        # 记录手肘角度最小值
        shooting_pose['elbow_min_angle'] = min(
            shooting_pose['elbow_min_angle'], elbowAngle)
        # 记录手肘角度最大值
        shooting_pose['elbow_max_angle'] = max(
            shooting_pose['elbow_max_angle'], elbowAngle)
        # 记录当前帧id
        if (shooting_pose['elbow_min_angle'] == elbowAngle):
            shooting_pose['elbow_min_time'] = frame_index
        if (shooting_pose['elbow_max_angle'] == elbowAngle):
            shooting_pose['elbow_max_time'] = frame_index
            # 记录脚踝角度最小值
        shooting_pose['ankle_min_angle'] = min(
            shooting_pose['ankle_min_angle'], ankleAngle)
        # 记录脚踝角度最大值
        shooting_pose['ankle_max_angle'] = max(
            shooting_pose['ankle_max_angle'], ankleAngle)
        # 记录当前帧id
        if (shooting_pose['ankle_min_angle'] == ankleAngle):
            shooting_pose['ankle_min_time'] = frame_index
        if (shooting_pose['ankle_max_angle'] == ankleAngle):
            shooting_pose['ankle_max_time'] = frame_index
            # 记录肩膀角度最小值
        shooting_pose['shoulder_min_angle'] = min(
            shooting_pose['shoulder_min_angle'], shoulderAngle)
        # 记录肩膀角度最大值
        shooting_pose['shoulder_max_angle'] = max(
            shooting_pose['shoulder_max_angle'], shoulderAngle)
        # 记录当前帧id
        if (shooting_pose['shoulder_min_angle'] == shoulderAngle):
            shooting_pose['shoulder_min_time'] = frame_index
        if (shooting_pose['shoulder_max_angle'] == shoulderAngle):
            shooting_pose['shoulder_max_time'] = frame_index
        shooting_result['elbow_min_angle'] = shooting_pose['elbow_min_angle']
        shooting_result['knee_min_angle'] = shooting_pose['knee_min_angle']
        shooting_result['ankle_min_angle'] = shooting_pose['ankle_min_angle']
        shooting_result['shoulder_min_angle'] = shooting_pose['shoulder_min_angle']
        shooting_result['elbow_max_angle'] = shooting_pose['elbow_max_angle']
        shooting_result['knee_max_angle'] = shooting_pose['knee_max_angle']
        shooting_result['ankle_max_angle'] = shooting_pose['ankle_max_angle']
        shooting_result['shoulder_max_angle'] = shooting_pose['shoulder_max_angle']
    except Exception as e:
        print("姿态估计有误:", e)
        white_bg = np.ones((int(target_height), int(
            target_width), 3), dtype=np.uint8) * 255
        merged = np.hstack((img, white_bg))
        white_bg = np.ones((400, 400, 3), dtype=np.uint8) * 255
        return merged, frame, preelbowAngle, prekneeAngle, preankleAngle, preshoulderAngle, prewristAngle, white_bg
    # 如果找到了姿态，将关键点绘制到图像上
    if results.pose_landmarks:
        # 获取segmentation_mask并调整其尺寸以适应原始图像
        # mask = results.segmentation_mask
        # frame[:]=[255,255,255]
        # # 二值化
        # _, mask = cv2.threshold(mask, 0.8, 255, cv2.THRESH_BINARY)
        # # 腐蚀操作
        # kernel = np.ones((3, 3), np.uint8)
        # eroded_mask = cv2.erode(mask, kernel, iterations=5)
        # # 获取边缘
        # edges = mask - eroded_mask
        # # 将 frame 上的边缘位置变为白色
        # frame[edges == 255] = [0, 0, 0]
        # 获取segmentation_mask并调整其尺寸以适应原始图像
        mask = results.segmentation_mask
        # 二值化
        _, mask = cv2.threshold(mask, 0.8, 255, cv2.THRESH_BINARY)
        # 使用形态学膨胀操作来增大mask的范围
        kernel = np.ones((9, 9), np.uint8)  # 你可以调整kernel的大小来控制膨胀的程度
        dilated_mask = cv2.dilate(mask, kernel, iterations=2)
        # 创建一个与原始图像大小相同的白色背景
        white_background = np.ones_like(frame) * 255
        # 使用dilated_mask将原始图像的内容复制到白色背景上
        result = cv2.bitwise_and(
            frame, frame, mask=dilated_mask.astype(np.uint8))
        result += cv2.bitwise_and(white_background, white_background,
                                  mask=~dilated_mask.astype(np.uint8))
        frame = result
        # 现在，result图像只保留了mask内部的部分，其余部分是白色的
        pointStyle = mp_drawing.DrawingSpec(
            color=(255, 0, 0), thickness=3)  # landmark的样式
        linStyle = mp_drawing.DrawingSpec(
            color=(0, 255, 0), thickness=3)  # 连线的样式
        bodyAngle, midCoord = draw_connections(
            real_img, frame, results.pose_landmarks.landmark, POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image=frame, landmark_list=results.right_hand_landmarks,
                                  connections=mp_holistic.HAND_CONNECTIONS,
                                  landmark_drawing_spec=pointStyle,
                                  connection_drawing_spec=linStyle)
        mp_drawing.draw_landmarks(image=real_img, landmark_list=results.right_hand_landmarks,
                                  connections=mp_holistic.HAND_CONNECTIONS,
                                  landmark_drawing_spec=pointStyle,
                                  connection_drawing_spec=linStyle)
        shooting_pose['body_angle'].append(bodyAngle)
        shooting_pose['body_xy'].append(midCoord)
        data_pose_xy[str(round(frame_index, 2))] = [
            [coord[0], int(target_height) - coord[1]] for coord in [
                ankleCoord.tolist(),
                shoulderCoord.tolist(),
                hipCoord.tolist(),
                wristCoord.tolist(),
                fingerCoord.tolist(),
                elbowCoord.tolist(),
                list(midCoord)
            ]
        ]
        filename = "./static/json/data_pose_xy.json"
        with open(filename, "w") as file:
            json.dump(data_pose_xy, file)
        resized_img = crop_and_resize_to_match(
            real_img, wristCoord, target_width)
        # 读取图像
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(rgb_image)
        # 创建一个可以在上面绘图的对象
        draw = ImageDraw.Draw(image_pil)
        # 指定字体和大小
        font = ImageFont.truetype(
            './static/font/AdobeHeitiStd-Regular.otf', 37)  # 更换为你系统中的中文字体文件路径
        # 添加文本
        draw.text((elbowCoord[0] + 20, elbowCoord[1]), "手肘角度： " +
                  str(elbowAngle) + '°', font=font, fill=(0, 139, 0))
        draw.text((kneeCoord[0] + 20, kneeCoord[1]), "膝盖角度： " +
                  str(kneeAngle) + '°', font=font, fill=(0, 139, 0))
        draw.text((ankleCoord[0] + 20, ankleCoord[1]), "脚踝角度： " +
                  str(ankleAngle) + '°', font=font, fill=(0, 139, 0))
        draw.text((shoulderCoord[0] + 20, shoulderCoord[1]), "肩膀角度： " +
                  str(shoulderAngle) + '°', font=font, fill=(0, 139, 0))
        draw.text((wristCoord[0] + 20, wristCoord[1]), "手腕角度： " + str(wristAngle) + '°', font=font,
                  fill=(0, 139, 0))
        draw.text((midCoord[0] + 20, midCoord[1]), "轴线偏离角度： " + str(bodyAngle) + '°', font=font,
                  fill=(0, 139, 0))

        draw.text((70, 30), "篮球轨迹", font=font, fill=(0, 255, 255))
        draw.text((70, 80), "手腕轨迹", font=font, fill=(240, 32, 160))
        draw.text((70, 130), "手肘轨迹", font=font, fill=(0, 140, 255))
        draw.text((70, 180), "身体中轴轨迹", font=font, fill=(96, 48, 176))
        draw.text((70, 230), "脚踝轨迹", font=font, fill=(255, 165, 0))

        if (shooting_pose['ball_in_hand'] == False):
            draw.text((during_shooting['release_point'][0]-150,
                      during_shooting['release_point'][1]), "检测到投篮", font=font, fill=(178, 34, 34))
            # draw.text((during_shooting['release_point'][0]-130, during_shooting['release_point'][1] ), "投篮手肘角度： " + str(shooting_result['release_elbow_angle']) + '°', font=font, fill=(178, 34, 34))
        # 转回OpenCV格式并显示
        frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        cv2.circle(frame, (50, 50), 10, (255, 255, 0), -1)
        cv2.circle(frame, (50, 100), 10, (160, 32, 240), -1)
        cv2.circle(frame, (50, 150), 10, (255, 140, 0), -1)
        cv2.circle(frame, (50, 200), 10, (176, 48, 96), -1)
        cv2.circle(frame, (50, 250), 10, (0, 165, 255), -1)

    # resized_img = crop_and_resize_to_match(frame, wristCoord, target_width)

    # 使用yolo进行检测和分割
    yoloresults = yolo_model(img)
    for result in yoloresults:  # iterate results
        # img = result.plot(labels=False, boxes=False, probs=False, conf=False, img=img,masks=False)
        boxes = result.boxes.cpu().numpy()  # get boxes on cpu in numpy
        names = result.names  # get classes
        masks = result.masks
        # iterate boxes and classes
        for box, name, mask in zip(boxes, names, masks):
            print(names[int(box.cls[0])])
            if names[int(box.cls[0])] == 'sports ball':
                r = box.xyxy[0].astype(int)  # get corner points as int
                cv2.rectangle(frame, r[:2], r[2:], (255, 0, 0), 2)
                xmin = r[0]
                ymin = r[1]
                xmax = r[2]
                ymax = r[3]
                # 平均值为球重心的坐标
                xCoor = int(np.mean([xmin, xmax]))
                yCoor = int(np.mean([ymin, ymax]))
                shooting_pose['ball_xy'].append((xCoor, yCoor, frame_index))
                if len(shooting_pose['ball_xy']) > 35:
                    del shooting_pose['ball_xy'][0]
                # 绘制球心
                ball_radius = int((xmax - xmin) / 2) * 2
                cv2.circle(img=frame, center=(xCoor, yCoor), radius=10,
                           color=(255, 255, 0), thickness=-1)
                cv2.circle(img=real_img, center=(xCoor, yCoor), radius=10,
                           color=(255, 255, 0), thickness=-1)
                cv2.circle(img=frame, center=(xCoor, yCoor), radius=int(ball_radius/2),
                           color=(0, 0, 0), thickness=4)
                for point in fingers:
                    cv2.line(real_img, point, (xCoor, yCoor), (0, 255, 0), 2)
                if ((distance([xCoor, yCoor], [handX, handY]) > ball_radius) and (shooting_pose['ball_in_hand'])):
                    # 认为球心和手腕的距离大于球的半径为投篮
                    shooting_result['release_time'] = str(
                        round(frame_index, 2))
                    shooting_pose['ball_in_hand'] = False
                    during_shooting['release_point'].clear()
                    during_shooting['release_point'].append(xCoor)
                    during_shooting['release_point'].append(yCoor)
                    shooting_result['ball_xy'] = copy.deepcopy(
                        shooting_pose['ball_xy'])
                    # 记录当前帧id
                    shooting_pose['release_time'] = frame_index
                    shooting_result['release_elbow_angle'] = elbowAngle
                    shooting_result['release_knee_angle'] = kneeAngle
                    shooting_result['release_ankle_angle'] = ankleAngle
                    shooting_result['release_shoulder_angle'] = shoulderAngle
                    shooting_result['release_wrist_angle'] = wristAngle
                    # 重置所有变量
                    during_shooting['balls_during_shooting'].clear()
                    during_shooting['isShooting'] = False
                break

    # 绘制篮球的轨迹
    if len(shooting_pose['ball_xy']) >= 2:
        # 使用cv2.circle绘制每个点
        for i in range(len(shooting_pose['ball_xy']) - 1):
            if (shooting_pose['ball_xy'][i][0] != -1):
                # 找到检测到的相邻两帧进行绘图
                for j in range(i+1, len(shooting_pose['ball_xy'])):
                    if (shooting_pose['ball_xy'][j][0] != -1):
                        cv2.line(frame, (shooting_pose['ball_xy'][i][0], shooting_pose['ball_xy'][i][1]),
                                 (shooting_pose['ball_xy'][j][0],
                                  shooting_pose['ball_xy'][j][1]),
                                 (255, 255, 0), 3)
                        break
        for index, point in enumerate(shooting_pose['ball_xy']):
            if (shooting_pose['ball_xy'][i][0] != -1):
                x, y, time = point
                cv2.circle(frame, (x, y), 8, (255, 255, 0), 3)

    # 绘制身体、手肘、手腕的轨迹
    if len(shooting_pose['body_xy']) >= 2:
        # 使用cv2.circle绘制每个点
        for i in range(len(shooting_pose['body_xy']) - 1):
            if (shooting_pose['body_xy'][i][0] != -1):
                # 找到检测到的相邻两帧进行绘图
                for j in range(i+1, len(shooting_pose['body_xy'])):
                    if (shooting_pose['body_xy'][j][0] != -1):
                        cv2.line(frame, (shooting_pose['body_xy'][i][0], shooting_pose['body_xy'][i][1]),
                                 (shooting_pose['body_xy'][j][0],
                                  shooting_pose['body_xy'][j][1]),
                                 (176, 48, 96), 1)
                        break
        for index, point in enumerate(shooting_pose['body_xy']):
            if (shooting_pose['body_xy'][i][0] != -1):
                x, y = point
                cv2.circle(frame, (x, y), 5, (176, 48, 96), -1)
    if len(shooting_pose['wrist_xy']) >= 2:
        # 使用cv2.circle绘制每个点
        for i in range(len(shooting_pose['wrist_xy']) - 1):
            if (shooting_pose['wrist_xy'][i][0] != -1):
                # 找到检测到的相邻两帧进行绘图
                for j in range(i+1, len(shooting_pose['wrist_xy'])):
                    if (shooting_pose['wrist_xy'][j][0] != -1):
                        cv2.line(frame, (shooting_pose['wrist_xy'][i][0], shooting_pose['wrist_xy'][i][1]),
                                 (shooting_pose['wrist_xy'][j][0],
                                  shooting_pose['wrist_xy'][j][1]),
                                 (160, 32, 240), 1)
                        break
        for index, point in enumerate(shooting_pose['wrist_xy']):
            if (shooting_pose['wrist_xy'][i][0] != -1):
                x, y = point
                cv2.circle(frame, (x, y), 5, (160, 32, 240), -1)
    if len(shooting_pose['elbow_xy']) >= 2:
        # 使用cv2.circle绘制每个点
        for i in range(len(shooting_pose['elbow_xy']) - 1):
            if (shooting_pose['elbow_xy'][i][0] != -1):
                # 找到检测到的相邻两帧进行绘图
                for j in range(i+1, len(shooting_pose['elbow_xy'])):
                    if (shooting_pose['elbow_xy'][j][0] != -1):
                        cv2.line(frame, (shooting_pose['elbow_xy'][i][0], shooting_pose['elbow_xy'][i][1]),
                                 (shooting_pose['elbow_xy'][j][0],
                                  shooting_pose['elbow_xy'][j][1]),
                                 (255, 140, 0), 1)
                        break
        for index, point in enumerate(shooting_pose['elbow_xy']):
            if (shooting_pose['elbow_xy'][i][0] != -1):
                x, y = point
                cv2.circle(frame, (x, y), 5, (255, 140, 0), -1)
    if len(shooting_pose['foot_xy']) >= 2:
        # 使用cv2.circle绘制每个点
        for i in range(len(shooting_pose['foot_xy']) - 1):
            if (shooting_pose['foot_xy'][i][0] != -1):
                # 找到检测到的相邻两帧进行绘图
                for j in range(i+1, len(shooting_pose['foot_xy'])):
                    if (shooting_pose['foot_xy'][j][0] != -1):
                        cv2.line(frame, (shooting_pose['foot_xy'][i][0], shooting_pose['foot_xy'][i][1]),
                                 (shooting_pose['foot_xy'][j][0],
                                  shooting_pose['foot_xy'][j][1]),
                                 (0, 165, 255), 1)
                        break
        for index, point in enumerate(shooting_pose['foot_xy']):
            if (shooting_pose['foot_xy'][i][0] != -1):
                x, y = point
                cv2.circle(frame, (x, y), 5, (0, 165, 255), -1)

    # final_img = paste_on_white_background(resized_img, target_width, target_height)
    merged = np.hstack((real_img, frame))
    # 释放内存
    del result
    del yoloresults
    gc.collect()
    return merged, resized_img, elbowAngle, kneeAngle, ankleAngle, shoulderAngle, wristAngle, frame
