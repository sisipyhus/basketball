import os
import json
import logging
from flask import Flask, render_template, Response, request, session, redirect, url_for, send_from_directory, flash, \
    jsonify
from werkzeug.utils import secure_filename
from src.config import shooting_result , result_scores , hand_result,hand_scores
from src.app_helper import getVideoStream, getHandVideoStream

app = Flask(__name__, static_folder="static")
current_directory = os.path.dirname(__file__)
UPLOAD_FOLDER = combined_path = os.path.join(current_directory, "static/uploads")
current_dir = os.getcwd()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super secret key"

# 配置日志记录
logging.basicConfig(level=logging.DEBUG)  # 设置日志记录级别为DEBUG

@app.route("/")
def index():
    return render_template("index.html")

# @app.route('/qrcode')
# def index():
#     # 生成二维码
#     img = qrcode.make('http://your_server_address/upload')
#     img.save('./static/qrcode.png')
#     return render_template('qrcode.html')  # 在这个模板中显示二维码

@app.route("/videoer")
def jump_to_hand():
    return render_template("videoer.html")

@app.route("/shooting_pose_analysis")
def jump_to_body():
    return render_template("shooting_pose_analysis.html")

@app.route('/videoer', methods=['GET', 'POST'])
def hand_sample_video():
    global hand_result
    hand_result['isEnd'] = False
    if request.method == 'POST':
        f = request.files['video']
        # create a secure filename
        filename = secure_filename(f.filename)
        # save file to /static/uploads
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)
        session['video_path'] = filepath
        video_path = session.get('video_path', None)
        return render_template("videoer.html")


@app.route('/json_data', methods=['GET'])
def get_data():
    # 从文件中加载JSON数据
    with open('data.json') as file:
        data = json.load(file)
    # 将加载的数据作为JSON响应返回
    return jsonify(data)

@app.route('/json_data_pose', methods=['GET'])
def get_data_pose():
    # 从文件中加载JSON数据
    with open('./static/json/data_pose.json') as file:
        data = json.load(file)
    # 将加载的数据作为JSON响应返回
    return jsonify(data)

@app.route('/json_data_pose_xy', methods=['GET'])
def get_data_pose_xy():
    # 从文件中加载JSON数据
    with open('./static/json/data_pose_xy.json') as file:
        data = json.load(file)
    # 将加载的数据作为JSON响应返回
    return jsonify(data)


@app.route('/video_feed')
def video_feed():
    global shooting_result
    global result_scores
    video_path = session.get('video_path', None)
    # 重置所有变量
    for key in shooting_result:
        shooting_result[key] = 0
    shooting_result['ball_xy'] = []
    shooting_result['draw_track'] = False
    shooting_result['isEnd'] = False
    # 将result_scores字典中的所有值都重置为'D待评价'
    for key in result_scores:
        result_scores[key] = '待评价'
    stream = getVideoStream(video_path)
    return Response(stream,
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_hand_feed')
def video_hand_feed():
    video_path = session.get('video_path', None)
    stream = getHandVideoStream(video_path)
    return Response(stream,
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload_hand', methods=['POST'])
def upload_hand():
    global hand_result
    hand_result['isEnd'] = False
    video = request.files.get('video')
    response = {}
    if video:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        video.save(os.path.join(UPLOAD_FOLDER, "recorded.mp4"))
        path = os.path.join(UPLOAD_FOLDER, "recorded.mp4")
        session['video_path'] = path
        video_path = session.get('video_path', None)
    return render_template("videoer.html")

@app.route('/upload_pose', methods=['POST'])
def upload_pose():
    global shooting_result
    shooting_result['isEnd'] = False
    video = request.files.get('video')
    response = {}
    if video:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        video.save(os.path.join(UPLOAD_FOLDER, "recorded.mp4"))
        path = os.path.join(UPLOAD_FOLDER, "recorded.mp4")
        session['video_path'] = path
        video_path = session.get('video_path', None)
    return render_template("shooting_pose_analysis.html")


@app.route('/current-video', methods=['GET', 'POST'])
def upload_video_html():
    return render_template("current-video.html")

@app.route('/current-pose-video', methods=['GET', 'POST'])
def upload_video_pose_html():
    return render_template("current-pose-video.html")

@app.route('/shooting_pose_analysis', methods=['GET', 'POST'])
def upload_body_video():
    global shooting_result
    shooting_result['isEnd'] = False
    if (os.path.exists("./static/detections/trajectory_fitting.jpg")):
        os.remove("./static/detections/trajectory_fitting.jpg")
    if request.method == 'POST':
        f = request.files['video']
        print(f)
        # create a secure filename
        filename = secure_filename(f.filename)
        print("filename", filename)
        # save file to /static/uploads
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        current_directory = os.getcwd()
        filepath2 = os.path.join(current_directory, '/output_pose_video.mp4')
        print("filepath", filepath)
        f.save(filepath)
        session['video_path'] = filepath
        session['replay_path'] = filepath2
        return render_template("shooting_pose_analysis.html")

@app.route('/get_shooting_result')
def get_shooting_result():
    # 从数据库或其他地方获取更新后的数据
    updated_result = shooting_result
    updated_score = result_scores
    # 将数据合并到一个字典中
    response_data = {
        'result': updated_result,
        'score': updated_score
    }
    return jsonify(response_data)

@app.route('/get_hand_result')
def get_hand_result():
    # 从数据库或其他地方获取更新后的数据
    updated_result = hand_result
    updated_score = hand_scores
    # 将数据合并到一个字典中
    response_data = {
        'result': hand_result,
        'score': hand_scores
    }
    return jsonify(response_data)

@app.route('/standard_pose')
def jump_to_stdpose():
    #跳转到标准动作界面
    return render_template("standard_pose.html")

@app.route('/standard_hand')
def jump_to_stdhand():
    #跳转到标准动作界面
    return render_template("standard_hand.html")


#disable caching
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

if __name__ == '__main__':
    app.run(use_reloader=True,port=5001,debug=False)
