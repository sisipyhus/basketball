prewristAngle = 180
preelbowAngle,prekneeAngle,preankleAngle,preshoulderAngle = 0,0,0,0
preelbowCoord,prekneeCoord,preankleCoord, preshoulderCoord ,prewristCoord,prefingerCoord = (0,0),(0,0),(0,0),(0,0),(0,0),(0,0)
POSE_CONNECTIONS = [
    [11, 13], [11, 12],
    [12, 14],
    [14, 16], [11, 23],
    [12, 24], [23, 24], [23, 25],
    [24, 26], [25, 27], [26, 28],
    [27, 29], [28, 30], [29, 31],
    [30, 32], [27, 31], [28, 32]
]
HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3],
    [3, 4], [0, 5], [5, 6],
    [6, 7], [7, 8], [5, 9],
    [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15],
    [15, 16], [13, 17], [0, 17],
    [17, 18], [18, 19], [19, 20]
]
bili = 6
shooting_result = {
    'is_line':False,
    # 投篮时间点
    'release_time':0,
    'addv':0,
    # 球在最低点的时间点
    'down_time':0,
    'attempts': 0,
    'made': 0,
    'miss': 0,
    #视频是否播完
    'isEnd': False,
    #手肘最小角度
    'elbow_min_angle': 0,
    #膝盖最小角度
    'knee_min_angle': 0,
    #脚踝最小角度
    'ankle_min_angle': 0,
    #肩膀最小角度
    'shoulder_min_angle': 0,
    #手腕最小角度
    'wrist_min_angle': 0,
    #手肘最大角度
    'elbow_max_angle': 0,
    #膝盖最大角度
    'knee_max_angle': 0,
    #脚踝最大角度
    'ankle_max_angle': 0,
    #肩膀最大角度
    'shoulder_max_angle': 0,
    #手腕最大角度
    'wrist_max_angle': 0,
    #投篮手肘角度
    'release_elbow_angle': 0,
    'release_ankle_angle': 0,
    'release_knee_angle': 0,
    'release_shoulder_angle': 0,
    'release_wrist_angle': 0,
    'avg_ballInHand_time': 0,
    #上下肢发力时间差
    'time_diff':0,
    #手臂伸展时间
    'elbow_time_diff':0,
    'ankle_time_diff':0,
    #膝盖伸展时间
    'knee_time_diff':0,
    #肩膀伸展时间
    'shoulder_time_diff':0,
    # 手腕下勾时间
    'wrist_time_diff': 0,
    #篮球路径坐标
    'ball_xy':[],
    #是否绘制轨迹图
    'draw_track':False,
    #脚的落点距离
    'foot_dis':0,
}

result_scores = {
    #手肘最小角度
    'elbow_min_angle': '待评价',
    #膝盖最小角度
    'knee_min_angle': '待评价',
    #脚踝最小角度
    'ankle_min_angle': '待评价',
    #肩膀最小角度
    'shoulder_min_angle': '待评价',
#手腕最小角度
    'wrist_min_angle': '待评价',
    #手肘最大角度
    'elbow_max_angle': '待评价',
    #膝盖最大角度
    'knee_max_angle': '待评价',
    #脚踝最大角度
    'ankle_max_angle': '待评价',
    #肩膀最大角度
    'shoulder_max_angle': '待评价',
    #手腕最小角度
    'wrist_max_angle': '待评价',
    #投篮手肘角度
    'release_elbow_angle': '待评价',
    'release_ankle_angle': '待评价',
    'release_knee_angle': '待评价',
    'release_shoulder_angle': '待评价',
'release_wrist_angle': '待评价',
# 手腕下勾时间
    'wrist_time_diff': '待评价',
    #上下肢发力时间差
    'time_diff':'待评价',
    #手臂伸展时间
    'elbow_time_diff':'待评价',
    'ankle_time_diff':'待评价',
    #膝盖伸展时间
    'knee_time_diff':'待评价',
    #肩膀伸展时间
    'shoulder_time_diff':'待评价',
}


hand_result = {
    #视频是否播完
    'isEnd': False,
    #手肘最小角度
    'elbow_min_angle': 0,
    #手掌最小角度
    'hand_min_angle': 0,
    #手指最小角度
    'finger_min_angle': 0,
#手腕最小角度
    'wrist_min_angle': 0,
    #肩膀最小角度
    'shoulder_min_angle': 0,
    # 手肘最大角度
    'elbow_max_angle': 0,
    # 手掌最大角度
    'hand_max_angle': 0,
    # 手指最大角度
    'finger_max_angle': 0,
    # 手腕最大角度
    'wrist_max_angle': 0,
    # 肩膀最小角度
    'shoulder_max_angle': 0,
    #投篮手肘角度
    # 'release_elbow_angle': 0,
    # 'release_wrist_angle': 0,
    # 'release_knee_angle': 0,
    # 'release_shoulder_angle': 0,
    # 'avg_ballInHand_time': 0,
    # #上下肢发力时间差
    # 'time_diff':0,
    # #手臂伸展时间
    # 'elbow_time_diff':0,
    # 'wrist_time_diff':0,
    # #膝盖伸展时间
    # 'knee_time_diff':0,
    # #肩膀伸展时间
    # 'shoulder_time_diff':0,
}

hand_scores = {
    # 手肘最小角度
    'elbow_min_angle': '待评价',
    # 手掌最小角度
    'hand_min_angle': '待评价',
    # 手指最小角度
    'finger_min_angle': '待评价',
    # 手腕最小角度
    'wrist_min_angle': '待评价',
    # 肩膀最小角度
    'shoulder_min_angle': '待评价',
    # 手肘最大角度
    'elbow_max_angle': '待评价',
    # 手掌最大角度
    'hand_max_angle': '待评价',
    # 手指最大角度
    'finger_max_angle': '待评价',
    # 手腕最大角度
    'wrist_max_angle': '待评价',
    # 肩膀最小角度
    'shoulder_max_angle': '待评价',
}

