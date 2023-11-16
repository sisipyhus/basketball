import math
import time
from PIL import Image, ImageDraw, ImageFont
import cv2
import json
import numpy as np
import gc
from .config import shooting_result, result_scores,hand_scores,hand_result,prewristAngle,prewristCoord,bili
import mediapipe as mp
import matplotlib.pyplot as plt
from .utils import detect_pose,calculateAngle,calculateAngle2
from ultralytics import YOLO
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
import subprocess

def interpolate_missing_points(ball_xy,interpolatedBallXY):
    #把记录到的值填入interpolatedBallXY
    for point in ball_xy:
        x,y,time = point
        for idx, (x2, y2, time2) in enumerate(interpolatedBallXY):
            if time == time2:
                interpolatedBallXY[idx] = (x-100, y+200, time)
                break
    ball_xy = np.array(interpolatedBallXY)
    x_coords, y_coords, times = ball_xy[:, 0], ball_xy[:, 1], ball_xy[:, 2]
    # Find indices where both x and y are not 0
    valid_indices = np.where((x_coords != -1) | (y_coords != -1))[0]
    missing_indices = np.where((x_coords == -1) & (y_coords == -1))[0]
    # Interpolate for x and y separately
    f_x = interp1d(valid_indices, x_coords[valid_indices], kind='linear', fill_value="extrapolate")
    f_y = interp1d(valid_indices, y_coords[valid_indices], kind='linear', fill_value="extrapolate")
    x_coords[missing_indices] = f_x(missing_indices).astype(int)
    y_coords[missing_indices] = f_y(missing_indices).astype(int)
    print(list(zip(x_coords, y_coords, times)))
    return list(zip(x_coords, y_coords, times))

def calculate_point_velocities(interpolatedBallXY):
    velocities = []
    # 遍历列表，但在最后一个点之前停止，以避免索引超出范围
    for i in range(len(interpolatedBallXY) - 1):
        x1, y1, t1 = interpolatedBallXY[i]
        x2, y2, t2 = interpolatedBallXY[i + 1]
        # 计算两点之间的距离
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        # 计算时间差
        time_difference = t2 - t1
        # 计算速度
        if time_difference != 0:  # 避免除以零
            velocity = distance / time_difference
        else:
            velocity = 0  # 如果时间差为零，速度设置为零
        velocities.append(velocity)
    # 对于最后一个点，我们可以直接添加0或与前一个点相同的速度
    velocities.append(0)
    return velocities

def calculate_acceleration(v1, v2, t1, t2):
    delta_v = v2 - v1
    delta_t = t2 - t1
    if delta_t == 0:  # 防止除以0的情况
        return 0
    acceleration = delta_v / delta_t
    return acceleration

def drawBody(points,width, height,name):
    # 创建一个与 frame 同样大小的白色背景
    # background = np.ones_like(img) * 255  # 255表示白色
    background = np.ones((height, width, 3), dtype=np.uint8) * 255
    # 使用 cv2.circle 绘制每个点和线段
    miny = 10000
    maxy = -370
    for i in range(len(points) - 1):
        start_point = tuple(map(int, points[i]))  # 只取前两个元素
        end_point = tuple(map(int, points[i + 1]))  # 只取前两个元素
        cv2.line(background, start_point, end_point, (176, 48, 96), 2)
    point1 = None
    point2 = None
    for index, point in enumerate(points):
        x, y = map(int, point)  # 将x和y转换为整数
        cv2.circle(background, (x, y), 5, (176, 48, 96), -1)
        if (y < miny):
            miny = y
            point1 = (x, y)
            continue
        if (y > maxy):
            # 判断该点是否是极小值点
            maxy = y
            point2 = (x, y)
            continue
    # 最低点坐标轴
    midpoint = (int((point1[0]+point2[0])/2),int((point1[1]+point2[1])/2))
    cv2.line(background, (midpoint[0],midpoint[1]-300), (midpoint[0],midpoint[1]+300), (0, 0, 255), 3)
    cv2.line(background, point1, point2, (0, 0, 255), 3)
    # 计算两向量之间的角度
    angle_deg = calculateAngle2(np.array(point1), np.array(midpoint),
                               np.array([midpoint[0], midpoint[1]-300]))
    # 读取图像
    rgb_image = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(rgb_image)
    # 创建一个可以在上面绘图的对象
    draw = ImageDraw.Draw(image_pil)
    # 指定字体和大小
    font = ImageFont.truetype('./static/font/AdobeHeitiStd-Regular.otf', 40)  # 更换为你系统中的中文字体文件路径
    # 添加文本
    draw.text((midpoint[0] + 50, midpoint[1] - 50), str(angle_deg) + '°', font=font,
              fill=(255, 0, 0))
    if angle_deg > 10:
        draw.text((midpoint[0] - 20, midpoint[1] + 180), "轴线发力方向有误", font=font,
                  fill=(255, 0, 0))
    else:
        draw.text((midpoint[0] - 20, midpoint[1] + 180), "轴线发力方向正确", font=font,
                  fill=(0, 200, 0))
    # 转回OpenCV格式并显示
    background = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    # 保存为本地图片
    cv2.imwrite('./static/img/'+name+'_track.jpg', background)

def drawWrist(points,width, height,name):
    # 创建一个与 frame 同样大小的白色背景
    # background = np.ones_like(img) * 255  # 255表示白色
    background = np.ones((height, width, 3), dtype=np.uint8) * 255
    # 使用 cv2.circle 绘制每个点和线段
    miny = 10000
    maxy = -370
    for i in range(len(points) - 1):
        start_point = tuple(map(int, points[i]))  # 只取前两个元素
        end_point = tuple(map(int, points[i + 1]))  # 只取前两个元素
        cv2.line(background, start_point, end_point, (160 ,32 ,240), 2)
    point1 = None
    point2 = None
    for index, point in enumerate(points):
        x, y = map(int, point)  # 将x和y转换为整数
        cv2.circle(background, (x, y), 5, (160 ,32 ,240), -1)
        if (y < miny):
            miny = y
            point1 = (x, y)
            continue
        if (y > maxy):
            # 判断该点是否是极小值点
            maxy = y
            point2 = (x, y)
            continue
    # 最低点坐标轴
    cv2.line(background, point2, (point2[0] + 300, point2[1]), (0, 0, 255), 2)
    cv2.line(background, point2, (point2[0], point2[1] - 300), (0, 0, 255), 2)
    cv2.line(background, point1, point2, (0, 0, 255), 3)
    # 计算两向量之间的角度
    angle_deg = calculateAngle2(np.array(point1), np.array(point2),
                               np.array([point2[0]+100, point2[1]]))
    # 读取图像
    rgb_image = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(rgb_image)
    # 创建一个可以在上面绘图的对象
    draw = ImageDraw.Draw(image_pil)
    # 指定字体和大小
    font = ImageFont.truetype('./static/font/AdobeHeitiStd-Regular.otf', 40)  # 更换为你系统中的中文字体文件路径
    # 添加文本
    draw.text((point2[0] + 50, point2[1] - 50), str(angle_deg) + '°', font=font,
              fill=(255, 0, 0))
    if angle_deg > 90:
        draw.text((point2[0] - 20, point2[1] + 70), "发力方向有误", font=font,
                  fill=(255, 0, 0))
    else:
        draw.text((point2[0] - 20, point2[1] + 70), "发力方向正确", font=font,
                  fill=(0, 200, 0))

    # 转回OpenCV格式并显示
    background = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    # 保存为本地图片
    cv2.imwrite('./static/img/'+name+'_track.jpg', background)

def drawElbow(points,width, height,name):
    # 创建一个与 frame 同样大小的白色背景
    # background = np.ones_like(img) * 255  # 255表示白色
    background = np.ones((height, width, 3), dtype=np.uint8) * 255
    # 使用 cv2.circle 绘制每个点和线段
    miny = 10000
    maxy = -370
    for i in range(len(points) - 1):
        start_point = tuple(map(int, points[i]))  # 只取前两个元素
        end_point = tuple(map(int, points[i + 1]))  # 只取前两个元素
        cv2.line(background, start_point, end_point, (255,140 ,0), 2)
    point1 = None
    point2 = None
    for index, point in enumerate(points):
        x, y = map(int, point)  # 将x和y转换为整数
        cv2.circle(background, (x, y), 5, (255,140 ,0), -1)
        if (y < miny):
            miny = y
            point1 = (x, y)
            continue
        if (y > maxy):
            # 判断该点是否是极小值点
            maxy = y
            point2 = (x, y)
            continue
    # 最低点坐标轴
    cv2.line(background, point2, (point2[0] + 300, point2[1]), (0, 0, 255), 2)
    cv2.line(background, point2, (point2[0], point2[1] - 300), (0, 0, 255), 2)
    cv2.line(background, point1, point2, (0, 0, 255), 3)
    # 计算两向量之间的角度
    angle_deg = calculateAngle2(np.array(point1), np.array(point2),
                               np.array([point2[0]+100, point2[1]]))
    # 读取图像
    rgb_image = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(rgb_image)
    # 创建一个可以在上面绘图的对象
    draw = ImageDraw.Draw(image_pil)
    # 指定字体和大小
    font = ImageFont.truetype('./static/font/AdobeHeitiStd-Regular.otf', 40)  # 更换为你系统中的中文字体文件路径
    # 添加文本
    draw.text((point2[0] + 50, point2[1] - 50), str(angle_deg) + '°', font=font,
              fill=(255, 0, 0))
    if angle_deg > 45:
        draw.text((point2[0] - 20, point2[1] + 70), "发力方向有误", font=font,
                  fill=(255, 0, 0))
    else:
        draw.text((point2[0] - 20, point2[1] + 70), "发力方向正确", font=font,
                  fill=(0, 200, 0))

    # 转回OpenCV格式并显示
    background = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    # 保存为本地图片
    cv2.imwrite('./static/img/'+name+'_track.jpg', background)

def drawBall(interpolatedBallXY,velocities,width, height):
    # 创建一个与 frame 同样大小的白色背景
    background = np.ones((height, width,3), dtype=np.uint8) * 255
    # 使用 cv2.circle 绘制每个点和线段
    miny = 10000
    maxy = -370
    for i in range(len(interpolatedBallXY) - 1):
        start_point = tuple(map(int, interpolatedBallXY[i][:2]))  # 只取前两个元素
        end_point = tuple(map(int, interpolatedBallXY[i + 1][:2]))  # 只取前两个元素
        cv2.line(background, start_point, end_point, (255, 0, 0), 3)
    point1 = None
    point2 = None
    down_time = 0
    nextpoint = None
    v1 = 0
    v2 = 0
    print(shooting_result['release_time'])
    for index, point in enumerate(interpolatedBallXY):
        x, y = map(int, point[:2])  # 将x和y转换为整数
        frame_index = point[2]
        cv2.circle(background, (x, y), 8, (255, 0, 0), 3)
        if (str(round(frame_index, 2)) == shooting_result['release_time']):
            point1 = (x, y)
            if (len(interpolatedBallXY) > index + 1):
                (nextx, nexty) = interpolatedBallXY[index + 1][:2]
                nextpoint = (int(nextx), int(nexty))
                v1 = velocities[index]
            continue
        if (y > maxy):
            # 判断该点是否是极小值点
            if (index != 0 and index < len(interpolatedBallXY) and interpolatedBallXY[index + 1][1] < y and
                    interpolatedBallXY[index - 1][1] < y):
                maxy = y
                point2 = (x, y)
                down_time = frame_index
                v2 = velocities[index]
                continue
    shooting_result['down_time'] = str(round(down_time, 2))
    # 最低点坐标轴
    cv2.line(background, point2, (point2[0] + 300, point2[1]), (0, 0, 255), 2)
    cv2.line(background, point2, (point2[0], point2[1] - 300), (0, 0, 255), 2)
    cv2.line(background, point1, point2, (0, 0, 255), 3)
    # 投篮点坐标轴
    cv2.line(background, point1, (point1[0] + 300, point1[1]), (0, 0, 255), 2)
    cv2.line(background, point1, (point1[0], point1[1] - 300), (0, 0, 255), 2)
    # 若记录了投篮后的点就绘制
    cv2.line(background, point1, nextpoint, (0, 0, 255), 3)

    # 计算加速度
    acceleration = calculate_acceleration(v2, v1, float(shooting_result['down_time']),
                                          float(shooting_result['release_time']))
    shooting_result['addv'] = round(acceleration/bili, 2)
    # 计算两向量之间的角度
    angle_deg = calculateAngle(np.array(point1), np.array(point2),
                               np.array([point2[0] + 300, point2[1]]))
    angle_deg2 = calculateAngle(np.array(nextpoint), np.array(point1),
                                np.array([point1[0] + 300, point1[1]]))

    # 提取point1和point2之间的所有点
    points_between = [tuple(map(int, interpolatedBallXY[i][:2])) for i in range(len(interpolatedBallXY)) if
                      point1[1] <= interpolatedBallXY[i][1] <= point2[1]]

    # 如果有足够的点来进行线性回归
    if len(points_between) > 1:
        X = [[pt[0]] for pt in points_between]  # x坐标
        y = [pt[1] for pt in points_between]  # y坐标

        # 使用线性回归拟合这些点
        reg = LinearRegression().fit(X, y)
        predicted_y = reg.predict(X)

        # 计算残差平方和
        rss = sum((y[i] - predicted_y[i]) ** 2 for i in range(len(y)))

        # 设置一个阈值，例如10000，根据实际情况调整
        print('rss',rss)
        if rss < 10000:
            print('直线')
            shooting_result['is_line']=True
        else:
            shooting_result['is_line']=False

    # 读取图像
    rgb_image = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(rgb_image)
    # 创建一个可以在上面绘图的对象
    draw = ImageDraw.Draw(image_pil)
    # 指定字体和大小
    font = ImageFont.truetype('./static/font/AdobeHeitiStd-Regular.otf', 35)  # 更换为你系统中的中文字体文件路径
    # 添加文本
    draw.text((point2[0] - 50, point2[1] + 20), "最低点速度：" + str(round(v2/bili, 2))+'cm/s', font=font,
              fill=(255, 0, 0))
    draw.text((point1[0] + 20, point1[1] + 20), "投篮点速度：" + str(round(v1/bili, 2))+'cm/s', font=font,
              fill=(255, 0, 0))
    draw.text((point2[0] + 50, point2[1] - 50), str(angle_deg) + '°', font=font,
              fill=(255, 0, 0))
    draw.text((point1[0] + 50, point1[1] - 50), str(angle_deg2) + '°', font=font,
              fill=(255, 0, 0))
    draw.text((point2[0] - 20, point2[1] + 130), "行程加速度：" + str(round(acceleration/bili, 2))+'cm/s平方', font=font,
              fill=(255, 0, 0))
    if angle_deg > 55:
        draw.text((point2[0] - 20, point2[1] + 70), "投篮发力角度过大", font=font,
                  fill=(255, 0, 0))
    elif angle_deg < 40:
        draw.text((point2[0] - 20, point2[1] + 70), "投篮发力角度过小", font=font,
                  fill=(255, 0, 0))
    else:
        draw.text((point2[0] - 20, point2[1] + 70), "投篮发力角度正确", font=font,
                  fill=(0, 200, 0))

    if angle_deg2 > 48:
        draw.text((point1[0] - 20, point1[1] + 70), "篮球抛物线角度过大", font=font,
                  fill=(255, 0, 0))
    elif angle_deg2 < 42:
        draw.text((point1[0] - 20, point1[1] + 70), "篮球抛物线角度过小", font=font,
                  fill=(255, 0, 0))
    else:
        draw.text((point1[0] - 20, point1[1] + 70), "投篮发力角度正确", font=font,
                  fill=(0, 200, 0))

    if not shooting_result['is_line']:
        draw.text((point2[0] - 80, point2[1] + 190), "投篮内弹道非直线", font=font,
                  fill=(255, 0, 0))
    else:
        draw.text((point2[0] - 80, point2[1] + 190), "投篮内弹道近似直线", font=font,
                  fill=(0, 200, 0))
    # 转回OpenCV格式并显示
    background = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    # 保存为本地图片
    cv2.imwrite('./static/img/track.jpg', background)
    shooting_result['draw_track'] = True

def getVideoStream(video_path):
    global prexmin,prexmax,preymin,preymax
    #球和篮筐
    previous = {
    'ball': np.array([0, 0]),  # x, y
    'hoop': np.array([0, 0, 0, 0]),  # xmin, ymax, xmax, ymin
        'hoop_height': 0
    }
    during_shooting = {
        'isShooting': False,
        'balls_during_shooting': [],
        'release_angle': 0,
        'release_point': []
    }
    shooting_pose = {
        'ball_in_hand': True,
        'elbow_min_angle': 370,
        'elbow_max_angle': -370,
        'knee_min_angle': 370,
        'knee_max_angle': -370,
        'ankle_min_angle': 370,
        'ankle_max_angle': -370,
        'shoulder_min_angle': 370,
        'shoulder_max_angle': -370,
        'ballInHand_frames': 0,
        'knee_min_time':0,
        'elbow_min_time':0,
        'ankle_min_time': 0,
        'shoulder_min_time': 0,
        'knee_max_time': 0,
        'elbow_max_time': 0,
        'ankle_max_time': 0,
        'shoulder_max_time': 0,
        'ball_xy':[],
        'shoulder_xy':[],
        'hip_xy':[],
        'foot_xy':[],
        'wrist_xy':[],
        'elbow_xy': [],
        'finger_xy':[],
        'is_end':False,
        'body_angle':[],
        'body_xy':[]
    }
    shot_result = {
        'displayFrames': 0,
        'release_displayFrames': 0,
        'judgement': ""
    }
    data_pose = {}
    data_pose_xy = {}
    shooting_result['isEnd'] = False

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    #初始化yolo模型
    model = YOLO("../static/models/yolov8s-seg.pt")
    # 上传视频
    cap = cv2.VideoCapture(video_path)
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("无法打开视频文件")
        return
    # 摄像头
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    prexmin = 0
    prexmax = int(width)
    preymin = 0
    preymax = int(height)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(preymax,prexmax)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter("./static/videos/output_pose_video.avi", fourcc, 30.0,(2*prexmax,preymax))
    output_hand_video = cv2.VideoWriter("./static/videos/output_hand_video.avi", fourcc, 30.0, (400, 400))
    # output_std_video = cv2.VideoWriter("./static/videos/output_std_video.avi", fourcc, 30.0, (prexmax, preymax))
    endFlag = 0
    interpolatedBallXY = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,enable_segmentation=True) as holistic:
        while cap.isOpened():
            print('每帧分析中1')
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # 将毫秒转换为秒
            ret, img = cap.read()
            if not ret:
                break
            if shooting_pose['ball_in_hand'] == False:
                endFlag = endFlag + 1
            # 记录投球点后三帧篮球的位置
            if endFlag < 3:
                interpolatedBallXY.append((-1, -1, current_time))
                if len(interpolatedBallXY) > 35:
                    del interpolatedBallXY[0]
            elif shooting_pose['is_end']:
                #获取每个点的坐标与时间，进行插值
                interpolatedBallXY = interpolate_missing_points(shooting_result['ball_xy'], interpolatedBallXY)
                # 每个点的速度
                velocities = calculate_point_velocities(interpolatedBallXY)

                # 绘制手腕、手肘、中轴、脚踝轨迹
                drawWrist(shooting_pose['wrist_xy'],int(width),int(height),'wrist')
                drawElbow(shooting_pose['elbow_xy'],int(width),int(height),'elbow')
                drawBody(shooting_pose['body_xy'],int(width),int(height),'body')
                # 绘制篮球轨迹
                drawBall(interpolatedBallXY, velocities, int(width), int(height))
                # drawAnkle(shooting_pose['foot_xy'])
                # 计算脚起始点与落点的x距离
                shooting_result['foot_dis'] = int((shooting_pose['foot_xy'][-1][0]-shooting_pose['foot_xy'][0][0])/bili)
                print('落点距离',shooting_result['foot_dis'])

                # def check_int32(obj, path=""):
                #     if isinstance(obj, dict):
                #         for k, v in obj.items():
                #             check_int32(v, path + f".{k}")
                #     elif isinstance(obj, list):
                #         for i, v in enumerate(obj):
                #             check_int32(v, path + f"[{i}]")
                #     elif isinstance(obj, np.int32):
                #         print(f"Found int32 at {path}: {obj}")
                #
                # # 使用这个函数检查你的响应数据
                # check_int32(shooting_result)

                break
            detection,hand_detection, elbowAngle, kneeAngle, ankleAngle, shoulderAngle , wristAngle,stdimg = detect_pose(current_time, img, width,
                                                                                      height, mp_drawing, mp_holistic, holistic,
                                                                                      model, pose,
                                                                                      during_shooting, shot_result,
                                                                                      shooting_pose,data_pose_xy)

            frame = cv2.imencode('.jpg', detection)[1].tobytes()
            result = (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # 将数据写入文件
            data_pose[str(round(current_time, 2))] = [round(elbowAngle, 2), round(shoulderAngle, 2),
                                                      round(kneeAngle, 2), round(ankleAngle, 2),round(wristAngle, 2)]

            detection.dtype = np.uint8
            hand_detection.dtype = np.uint8
            # stdimg.dtype = np.uint8
            output_video.write(detection)
            output_hand_video.write(hand_detection)
            # output_std_video.write(stdimg)
            filename = "./static/json/data_pose.json"
            with open(filename, "w") as file:
                json.dump(data_pose, file)
            print('即将分析下一帧')
            yield result
    cap.release()
    output_video.release()
    output_hand_video.release()
    # output_std_video.release()
    # 定义ffmpeg命令
    cmd = ["ffmpeg", "-y", "-i", "./static/videos/output_pose_video.avi", "-vcodec", "h264", "-strict", "-2", "./static/videos/output_pose_video.mp4"]
    # 使用subprocess执行命令
    subprocess.run(cmd)
    cmd = ["ffmpeg", "-y", "-i", "./static/videos/output_hand_video.avi", "-vcodec", "h264", "-strict", "-2",
           "./static/videos/output_hand_video.mp4"]
    subprocess.run(cmd)
    # cmd = ["ffmpeg", "-y", "-i", "./static/videos/output_std_video.avi", "-vcodec", "h264", "-strict", "-2",
    #        "./static/videos/output_std_video.mp4"]
    # subprocess.run(cmd)
    gc.collect()


    # getting average shooting angle
    shooting_result['time_diff'] = round( abs(shooting_pose['knee_min_time'] - shooting_pose['elbow_min_time']),2)
    shooting_result['knee_time_diff'] = round(
        abs(shooting_pose['knee_min_time'] - shooting_pose['knee_max_time']), 2)
    shooting_result['elbow_time_diff'] = round(
        abs(shooting_pose['elbow_min_time'] - shooting_pose['elbow_max_time']), 2)
    shooting_result['ankle_time_diff'] = round(
        abs(shooting_pose['ankle_min_time'] - shooting_pose['ankle_max_time']), 2)
    shooting_result['shoulder_time_diff'] = round(
        abs(shooting_pose['shoulder_min_time'] - shooting_pose['shoulder_max_time']), 2)
    #对数据进行评估
    #脚踝
    if (shooting_result['release_ankle_angle'] < 120):
        result_scores['release_ankle_angle'] = '投篮时脚踝角度过小'
    # elif (shooting_result['release_ankle_angle'] > 140):
    #     result_scores['release_ankle_angle'] = '过大'
    else:
        result_scores['release_ankle_angle'] = '标准'

    if (shooting_result['ankle_min_angle'] < 70):
        result_scores['ankle_min_angle'] = '蓄力时脚踝角度过小'
    elif (shooting_result['ankle_min_angle'] > 85):
        result_scores['ankle_min_angle'] = '蓄力时脚踝角度过大'
    else:
        result_scores['ankle_min_angle'] = '标准'

    if (shooting_result['ankle_max_angle'] < 120):
        result_scores['ankle_max_angle'] = '脚踝伸展角度过小'
    elif (shooting_result['ankle_max_angle'] > 140):
        result_scores['ankle_max_angle'] = '脚踝伸展角度过大'
    else:
        result_scores['ankle_max_angle'] = '标准'

    #膝盖
    if (shooting_result['release_knee_angle'] < 165):
        result_scores['release_knee_angle'] = '投篮时膝盖伸展幅度过小'
    else:
        result_scores['release_knee_angle'] = '标准'

    if (shooting_result['knee_min_angle'] < 80):
        result_scores['knee_min_angle'] = '蓄力时膝盖角度过小'
    elif (shooting_result['knee_min_angle'] > 100):
        result_scores['knee_min_angle'] = '蓄力时膝盖角度过大'
    else:
        result_scores['knee_min_angle'] = '标准'

    if (shooting_result['knee_max_angle'] < 165):
        result_scores['knee_max_angle'] = '膝盖伸展角度过小'
    elif (shooting_result['knee_max_angle'] > 180):
        result_scores['knee_max_angle'] = '过大'
    else:
        result_scores['knee_max_angle'] = '标准'

    if (shooting_result['knee_time_diff'] < 0.25):
        result_scores['knee_time_diff'] = '膝盖发力过度'
    elif (shooting_result['knee_time_diff'] > 0.35):
        result_scores['knee_time_diff'] = '膝盖发力不够'
    else:
        result_scores['knee_time_diff'] = '标准'

    # 手肘
    if (shooting_result['release_elbow_angle'] < 150):
        result_scores['release_elbow_angle'] = '投篮时手肘角度过小'
    else:
        result_scores['release_elbow_angle'] = '标准'

    if (shooting_result['elbow_min_angle'] < 40):
        result_scores['elbow_min_angle'] = '蓄力时手肘角度过小'
    elif (shooting_result['elbow_min_angle'] > 60):
        result_scores['elbow_min_angle'] = '蓄力时手肘角度过大'
    else:
        result_scores['elbow_min_angle'] = '标准'

    if (shooting_result['elbow_max_angle'] < 150):
        result_scores['elbow_max_angle'] = '手肘伸展角度过小'
    elif (shooting_result['elbow_max_angle'] > 180):
        result_scores['elbow_max_angle'] = '手肘伸展角度过大'
    else:
        result_scores['elbow_max_angle'] = '标准'

    if (shooting_result['elbow_time_diff'] < 0.25):
        result_scores['elbow_time_diff'] = '发球加速度过大'
    elif (shooting_result['elbow_time_diff'] > 0.35):
        result_scores['elbow_time_diff'] = '发球加速度过小'
    else:
        result_scores['elbow_time_diff'] = '标准'

    #肩膀
    if(shooting_result['release_shoulder_angle']<120):
        result_scores['release_shoulder_angle'] = '投篮时肩膀角度过小'
    elif (shooting_result['release_shoulder_angle'] > 140):
        result_scores['release_shoulder_angle'] = '投篮时肩膀角度过大'
    else:
        result_scores['release_shoulder_angle'] = '标准'

    if (shooting_result['shoulder_min_angle'] < 60):
        result_scores['shoulder_min_angle'] = '蓄力时肩膀角度过小，肩膀可适度抬高'
    elif (shooting_result['shoulder_min_angle'] >80):
        result_scores['shoulder_min_angle'] = '蓄力时肩膀角度过大，肩膀可适度降低'
    else:
        result_scores['shoulder_min_angle'] = '标准'

    if (shooting_result['shoulder_max_angle'] < 120):
        result_scores['shoulder_max_angle'] = '肩膀伸展角度过小'
    elif (shooting_result['shoulder_max_angle'] >140):
        result_scores['shoulder_max_angle'] = '肩膀伸展角度过大'
    else:
        result_scores['shoulder_max_angle'] = '标准'

    #上下肢协调性
    if (shooting_result['time_diff'] > 0.03):
        result_scores['time_diff'] = '上下肢未同时发力'
    else:
        result_scores['time_diff'] = '标准'

    print("avg", shooting_result['elbow_min_angle'])
    print("avg", shooting_result['knee_min_angle'])
    print('diff',shooting_result['time_diff'])
    # 等待5秒钟
    shooting_result['isEnd']=True
    #time.sleep(5)
    cv2.destroyAllWindows()
    del mp_pose
    del mp_drawing
    del model
    print('即将退出函数')
    return


def getHandVideoStream(video_path):
    video = cv2.VideoCapture(video_path)
    mpHands = mp.solutions.hands  # 检测手的类
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic
    mp_pose = mp.solutions.pose

    mpHandDetesctor = mpHands.Hands(static_image_mode=False,  # 要检测是静态图片还是动态的？视频和影响摄像头是动态的，图片是静态的。
                                    max_num_hands=1,  # 检测最多的手的个数
                                    model_complexity=1,  # 模型复杂度，只能是0或者1，值越大越精准
                                    min_detection_confidence=0.5,  # 最小的置信度，0~1，值越大，检测越严格，值越小，误检测越高
                                    min_tracking_confidence=0.5)  # 最终的严谨度，0~1，值越大，追踪越严谨，值越小，误检测越高
    # 画图的工具
    mpHandDrawer = mp.solutions.drawing_utils

    pointStyle = mpHandDrawer.DrawingSpec(color=(255, 0, 0,), thickness=2)  # landmark的样式
    linStyle = mpHandDrawer.DrawingSpec(color=(0, 255, 0,), thickness=1)  # 连线的样式

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    joint_list = [[4, 3, 2], [7, 6, 5], [11, 10, 9], [15, 14, 13], [19, 18, 17], [6, 5, 0], [10, 9, 0], [14, 13, 0],
                  [18, 17, 0]]  # 手指关节序列

    # 获取视频的帧率和尺寸
    fps = video.get(cv2.CAP_PROP_FPS)
    W = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    text_width = 200  # 说明文本的宽度
    blank_image = np.zeros((H, W + text_width, 3), np.uint8)

    def enlarge_hand_region(image, hand_landmarks, scale_factor, W, H):
        # 获取手部位置的边界框
        x_min, y_min, x_max, y_max = float('inf'), float('inf'), -float('inf'), -float('inf')
        for landmark in hand_landmarks.landmark:
            x_min = int(min(x_min, landmark.x * W))
            y_min = int(min(y_min, landmark.y * H))
            x_max = int(max(x_max, landmark.x * W))
            y_max = int(max(y_max, landmark.y * H))

        # 计算放大后的边界框
        width = x_max - x_min
        height = y_max - y_min
        enlarged_width = int(width * scale_factor)
        enlarged_height = int(height * scale_factor)

        # 根据放大后的边界框尺寸，计算目标位置和大小
        target_x = int(x_min - (enlarged_width - width) / 2)
        target_y = int(y_min - (enlarged_height - height) / 2)
        target_width = int(enlarged_width)
        target_height = int(enlarged_height)

        # # 裁剪手部位置并放大
        # enlarged_region = image[target_y:target_y + target_height, target_x:target_x + target_width]
        # enlarged_region = cv2.resize(enlarged_region, (target_width, target_height))
        #
        # # 将放大后的手部位置替换原始图像中的手部位置
        # image[target_y:target_y + target_height, target_x:target_x + target_width] = enlarged_region

        return target_x, target_y, width, height, x_min, y_min, x_max, y_max

    def calculate_angle(point1, point2, point3):
        if not point1 or not point2 or not point3:
            return 0
        # 将点坐标转换为向量
        vectorAB = np.array(point1) - np.array(point2)
        vectorCB = np.array(point3) - np.array(point2)

        # 计算夹角的余弦值
        cos_angle = np.dot(vectorAB, vectorCB) / (np.linalg.norm(vectorAB) * np.linalg.norm(vectorCB))

        # 将余弦值转换为弧度
        angle_rad = np.arccos(cos_angle)

        # 将弧度转换为角度
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def calculate_angle_2(point1, point2, point3):
        if not point1 or not point2 or not point3:
            return 0

        vectorAB = np.array(point1) - np.array(point2)
        vectorCB = np.array(point3) - np.array(point2)

        normAB = np.linalg.norm(vectorAB)
        normCB = np.linalg.norm(vectorCB)

        if normAB == 0 or normCB == 0:
            return 0

        cos_angle = np.dot(vectorAB, vectorCB) / (normAB * normCB)

        if cos_angle < -1 or cos_angle > 1:
            return 0

        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        if np.isnan(angle_deg):
            return 0
        else:
            return angle_deg

    def is_approximately_increasing(arr):
        if (len(arr) <= 2): return False;
        for i in range(len(arr) - 1):
            if arr[i][-1] >= arr[i + 1][0]:
                return False
        return True

    def smooth_data(data, window_size):
        data = np.array(data)
        padded_data = np.pad(data, (0, window_size - 1), mode='edge')
        smooth_data = np.convolve(padded_data, np.ones((window_size,)) / window_size, mode='valid')
        return smooth_data

    # 添加中文
    def count_numbers(array):
        count = 0
        calculated = set()
        start = -1
        for i in range(len(array)):
            if array[i] > 30:
                start = i
                break
        if start == -1:
            return 0
        for i in range(start + 1, len(array)):
            if array[i] > 90:
                break
            if array[i] > 30 and array[i] not in calculated:
                count += 1
                calculated.add(array[i])
        if count > 30:
            return True
        else:
            return False
    def on_array_change(array):
        x = 50
        y = 50
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        # 创建一个与视频等高但宽度增加一定值的空白图像

        # 清空画布
        blank_image.fill(0)

        for text in array:
            # 测量文本大小
            (text_width, text_height) = cv2.getTextSize(text, font, font_scale, thickness=1)[0]
            # 绘制文本
            cv2.putText(blank_image, str(text), (x, y), font, font_scale, (255, 255, 255), 1)
            # blank_image = cv2ImgAddText(blank_image,text,x,y)
            # 更新文本位置
            y += text_height + 10
        # return blank_image

    # 指定感兴趣区域的坐标和尺寸
    x = 100  # 起始列索引
    y = 100  # 起始行索引
    roi_width = 1980  # 区域宽度
    roi_height = 1080  # 区域高度
    scale_factor = 10  # 放慢一倍
    slow_down_factor = 3

    fig, ax = plt.subplots()

    # 创建视频编写器以保存裁剪后的视频

    wrist_elbow_shoulder_angle_arr = []
    wrist_elbow_shoulder_time_arr = []
    shoulder_elbow_waist_angle_arr = []
    shoulder_elbow_waist_time_arr = []
    hand_wrist_elbow_arr = []
    hand_wrist_elbow_time_arr = []
    # 手指角度
    thumb_angle_arr = []
    index_finger_arr = []
    middle_finger_arr = []
    ring_finger_arr = []
    little_finger_arr = []

    index_finger_arr_5 = []
    middle_finger_arr_9 = []
    ring_finger_arr_13 = []
    little_finger_arr_17 = []

    finger_utils_time_arr = []

    # 手-肘-腰 状态变化

    #

    # 是否开始投篮
    start_to_have_basketball = "start to basketball"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter("../static/uploads/output_video.mp4", fourcc, 30, (W, H))
    data = {};
    analyze = {};
    analyze_info = {}
    hand_pose = {
        'ball_in_hand': True,
        'elbow_min_angle': 370,
        'elbow_max_angle': -370,
        'wrist_min_angle': 370,
        'wrist_max_angle': -370,
        'hand_min_angle': 370,
        'hand_max_angle': -370,
        'shoulder_min_angle': 370,
        'shoulder_max_angle': -370,
        'finger_min_angle': 370,
        'finger_max_angle': -370,
    }
    shoulder_coordinates = []
    elbow_coordinates = []
    wrist_coordinates = []
    waist_coordinates = []

    mid_hand_point = [0, 0]
    end_hand_point = [0, 0]
    begin_hand_point = [0, 0]

    wrist_coordinates_1 = [0, 0]
    waist_coordinates_1 = [0, 0]
    elbow_coordinates_1 = [0, 0]
    shoulder_coordinates_1 = [0, 0]

    wrist_elbow_shoulder_angle = 0
    waist_shoulder_elbow_angle = 0
    hand_wrist_elbow = 0
    begin_mid_end = 0
    wrist_begin_end = 0
    # 逐帧裁剪视频内容并保存
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        current_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000  # 将毫秒转换为秒
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 使用 mediapipe 来整手势识别
        result = mpHandDetesctor.process(imgRGB)
        body_result = pose.process(imgRGB)
        trace = np.full((int(H), int(W * 3), 3), 255, np.uint8)

        if body_result.pose_landmarks:
            for index, lm in enumerate(body_result.pose_landmarks.landmark):
                lm_x, lm_y, lm_z = int(W * lm.x), int(H * lm.y), int(H * lm.z)
                #肩膀
                if index == 12:
                    shoulder_coordinates = [lm_x, lm_y, lm_z]
                    shoulder_coordinates_1 = [lm_x, lm_y]
                    cv2.circle(frame, center=[lm_x, lm_y], radius=10, color=(0, 238, 238), thickness=cv2.FILLED)
                #手肘
                if index == 14:
                    elbow_coordinates = [lm_x, lm_y, lm_z]
                    elbow_coordinates_1 = [lm_x, lm_y]
                    cv2.circle(frame, center=[lm_x, lm_y], radius=10, color=(1, 238, 238), thickness=cv2.FILLED)
                #手腕
                if index == 16:
                    wrist_coordinates = [lm_x, lm_y, lm_z]
                    wrist_coordinates_1 = [lm_x, lm_y]
                    cv2.circle(frame, center=[lm_x, lm_y], radius=10, color=(50, 238, 238), thickness=cv2.FILLED)
                #屁股
                if index == 24:
                    waist_coordinates = [lm_x, lm_y, lm_z]
                    waist_coordinates_1 = [lm_x, lm_y]
                    cv2.circle(frame, center=[lm_x, lm_y], radius=10, color=(150, 238, 238), thickness=cv2.FILLED)
                # if any(elem != 0 for elem in wrist_coordinates_1) and any(
                #         elem != 0 for elem in shoulder_coordinates_1) and any(
                #     elem != 0 for elem in elbow_coordinates_1) and any(elem != 0 for elem in waist_coordinates_1):


                if waist_shoulder_elbow_angle is not None and wrist_elbow_shoulder_angle is not None:
                    current_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    if waist_shoulder_elbow_angle > 120:
                        # 判断是不是开始投篮
                        analyze_info[str(round(current_time, 3))] = [0]
                    else :
                        analyze_info[str(round(current_time, 3))] = [1]
                if result.multi_hand_landmarks:
                    RHL = result.multi_hand_landmarks[0]
                    # 计算角度
                    for joint in joint_list:
                        a = np.array([RHL.landmark[joint[0]].x, RHL.landmark[joint[0]].y])
                        b = np.array([RHL.landmark[joint[1]].x, RHL.landmark[joint[1]].y])
                        c = np.array([RHL.landmark[joint[2]].x, RHL.landmark[joint[2]].y])
                        # 计算弧度
                        radians_fingers = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1],
                                                                                            a[0] - b[0])
                        angle = np.abs(radians_fingers * 180.0 / np.pi)  # 弧度转角度

                        if angle > 180.0:
                            angle = 360 - angle
                        if joint[0] == 4:
                            thumb_angle_arr.append(angle)
                            current_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                            finger_utils_time_arr.append(current_time)
                        if joint[0] == 7:
                            index_finger_arr.append(angle)
                        if joint[0] == 11:
                            middle_finger_arr.append(angle)
                        if joint[0] == 15:
                            ring_finger_arr.append(angle)
                        if joint[0] == 19:
                            little_finger_arr.append(angle)
                        if joint[0] == 5:
                            index_finger_arr_5.append(angle)
                        if joint[0] == 9:
                            middle_finger_arr_9.append(angle)
                        if joint[0] == 13:
                            ring_finger_arr_13.append(angle)
                        if joint[0] == 17:
                            little_finger_arr_17.append(angle)
                wrist_elbow_shoulder_angle_arr.append(wrist_elbow_shoulder_angle)
                #current_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                wrist_elbow_shoulder_time_arr.append(current_time)
                shoulder_elbow_waist_angle_arr.append(waist_shoulder_elbow_angle)

            waist_shoulder_elbow_angle = calculate_angle(elbow_coordinates, shoulder_coordinates,
                                                         waist_coordinates)
            wrist_elbow_shoulder_angle = calculate_angle_2(shoulder_coordinates_1, elbow_coordinates_1,
                                                           wrist_coordinates_1)
            cv2.line(frame, shoulder_coordinates_1, wrist_coordinates_1, (0, 0, 100), 2)
            cv2.line(frame, wrist_coordinates_1, elbow_coordinates_1, (80, 255, 40), 2)
            cv2.line(frame, shoulder_coordinates_1, elbow_coordinates_1, (40, 200, 0), 2)
            cv2.line(frame, shoulder_coordinates_1, waist_coordinates_1, (200, 255, 0), 2)
            cv2.line(frame, elbow_coordinates_1, waist_coordinates_1, (0, 255, 200), 2)
            # 文字显示
            # 读取图像
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(rgb_image)
            # 创建一个可以在上面绘图的对象
            draw = ImageDraw.Draw(image_pil)
            # 指定字体和大小
            font = ImageFont.truetype('./static/font/AdobeHeitiStd-Regular.otf', 45)  # 更换为你系统中的中文字体文件路径
            # 添加文本
            draw.text((shoulder_coordinates_1[0] - 400, shoulder_coordinates_1[1]),
                      "肩膀角度： " + str(round(waist_shoulder_elbow_angle, 2)) + '°', font=font,
                      fill=(0, 255, 0))

            # cv2.putText(img=frame, text=str(round(waist_shoulder_elbow_angle, 3)), org=waist_coordinates_1,
            #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale=3, color=(0, 0, 255), thickness=1)
            if wrist_elbow_shoulder_angle != 0:
                # cv2.putText(img=frame, text=str(round(wrist_elbow_shoulder_angle, 3)), org=shoulder_coordinates_1,
                #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #             fontScale=3, color=(0, 0, 255), thickness=1)
                draw.text((elbow_coordinates_1[0] - 400, elbow_coordinates_1[1]),
                          "手肘角度： " + str(round(wrist_elbow_shoulder_angle, 2)) + '°', font=font,
                          fill=(0, 255, 0))
            # 转回OpenCV格式并显示
            frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks:
            for one_hand in result.multi_hand_landmarks:  # 遍历每一个手的坐标信息
                mpHandDrawer.draw_landmarks(image=frame, landmark_list=one_hand, connections=mpHands.HAND_CONNECTIONS,
                                            landmark_drawing_spec=pointStyle,
                                            connection_drawing_spec=linStyle)
                for i, lm in enumerate(one_hand.landmark):
                    lm_x, lm_y = int(W * lm.x), int(H * lm.y)
                    if i == 12:
                        end_hand_point = (lm_x, lm_y)
                    if i == 10:
                        mid_hand_point = (lm_x, lm_y)
                    if i == 9:
                        begin_hand_point = (lm_x, lm_y)
                hand_wrist_elbow = calculate_angle_2(elbow_coordinates_1, wrist_coordinates_1, end_hand_point)
                begin_mid_end = calculate_angle_2(begin_hand_point, mid_hand_point, end_hand_point)
                wrist_begin_end = calculate_angle_2(wrist_coordinates_1, begin_hand_point, end_hand_point)
                # cv2.putText(img=frame, text=str(round(hand_wrist_elbow, 3)), org=wrist_coordinates_1,
                #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #             fontScale=3, color=(0, 0, 255), thickness=1)
                # 读取图像
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(rgb_image)
                # 创建一个可以在上面绘图的对象
                draw = ImageDraw.Draw(image_pil)
                # 指定字体和大小
                font = ImageFont.truetype('./static/font/AdobeHeitiStd-Regular.otf', 45)  # 更换为你系统中的中文字体文件路径
                # 添加文本
                draw.text((wrist_coordinates_1[0] + 20, wrist_coordinates_1[1]),
                          "手腕角度： " + str(round(hand_wrist_elbow, 2)) + '°', font=font,
                          fill=(0, 255, 0))
                draw.text((begin_hand_point[0] + 20, begin_hand_point[1]),
                          "手掌角度： " + str(round(wrist_begin_end, 2)) + '°', font=font,
                          fill=(0, 255, 0))
                draw.text((mid_hand_point[0] + 20, mid_hand_point[1]),
                          "手指角度： " + str(round(begin_mid_end, 2)) + '°', font=font,
                          fill=(0, 255, 0))
                # 转回OpenCV格式并显示
                frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                hand_wrist_elbow_arr.append(hand_wrist_elbow)
                # current_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                hand_wrist_elbow_time_arr.append(current_time)

        #手肘、肩膀、手腕、手掌、手指
        data[str(round(current_time, 2))] = [round(wrist_elbow_shoulder_angle,2), round(waist_shoulder_elbow_angle,2),round(hand_wrist_elbow,2),
                                             round(wrist_begin_end, 2),round(begin_mid_end, 2)]
        wrist_elbow_shoulder_angle=round(wrist_elbow_shoulder_angle, 2)
        waist_shoulder_elbow_angle=round(waist_shoulder_elbow_angle, 2)
        hand_wrist_elbow=round(hand_wrist_elbow, 2)
        wrist_begin_end=round(wrist_begin_end, 2)
        begin_mid_end=round(begin_mid_end, 2)
        #保存角度最大值与最小值
        #手腕
        hand_pose['wrist_min_angle'] = min(
            hand_pose['wrist_min_angle'], hand_wrist_elbow)
        hand_pose['wrist_max_angle'] = max(
            hand_pose['wrist_max_angle'], hand_wrist_elbow)
        #肩膀
        hand_pose['shoulder_min_angle'] = min(
            hand_pose['shoulder_min_angle'], waist_shoulder_elbow_angle)
        hand_pose['shoulder_max_angle'] = max(
            hand_pose['shoulder_max_angle'], waist_shoulder_elbow_angle)
        #手肘
        hand_pose['elbow_min_angle'] = min(
            hand_pose['elbow_min_angle'], wrist_elbow_shoulder_angle)
        hand_pose['elbow_max_angle'] = max(
            hand_pose['elbow_max_angle'], wrist_elbow_shoulder_angle)
        #手掌
        hand_pose['hand_min_angle'] = min(
            hand_pose['hand_min_angle'], wrist_begin_end)
        hand_pose['hand_max_angle'] = max(
            hand_pose['hand_max_angle'], wrist_begin_end)
        #手指
        hand_pose['finger_min_angle'] = min(
            hand_pose['finger_min_angle'], begin_mid_end)
        hand_pose['finger_max_angle'] = max(
            hand_pose['finger_max_angle'], begin_mid_end)

        hand_result['wrist_min_angle']=hand_pose['wrist_min_angle']
        hand_result['wrist_max_angle'] = hand_pose['wrist_max_angle']
        hand_result['elbow_min_angle'] = hand_pose['elbow_min_angle']
        hand_result['elbow_max_angle'] = hand_pose['elbow_max_angle']
        hand_result['shoulder_min_angle'] = hand_pose['shoulder_min_angle']
        hand_result['shoulder_max_angle'] = hand_pose['shoulder_max_angle']
        hand_result['hand_min_angle'] = hand_pose['hand_min_angle']
        hand_result['hand_max_angle'] = hand_pose['hand_max_angle']
        hand_result['finger_min_angle'] = hand_pose['finger_min_angle']
        hand_result['finger_max_angle'] = hand_pose['finger_max_angle']

        output_video.write(frame)
        delay = int(500 / (fps * slow_down_factor))
        # time.sleep(delay / 500.0)
        filename = "data.json"
        with open(filename, "w") as file:
            json.dump(data, file)
        ret, buffer = cv2.imencode('.jpg', frame)
        # print(data)
        # print(wrist_elbow_shoulder_angle_arr)

        frame_byes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_byes + b'\r\n')

    #处理完视频后进行评价，并清空变量
    hand_result['isEnd']=True
    # 等待5秒钟
    time.sleep(5)
    # 重置所有变量
    for key in hand_result:
        hand_result[key] = 0
    # 将result_scores字典中的所有值都重置为'D待评价'
    for key in hand_scores:
        hand_scores[key] = '待评价'
    return


