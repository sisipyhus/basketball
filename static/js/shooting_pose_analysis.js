var isEnd = false
var isDraw = false
var pose_data
//时间序列
var timestamps = []
//角度序列
var elbowAngle = [], shoulderAngle = [], kneeAngle = [], ankleAngle = [] , wristAngle = []
//高度序列
var fingerHeight=[],wristHeight=[],ankleHeight=[]
//平滑处理后的角度序列
var smt_elbowAngle = [],smt_shoulderAngle = [],smt_kneeAngle = [],smt_ankleAngle = [],smt_wristAngle =[]
var release_time = 0 , down_time = 0
var release_index = 0, down_index = 0
let ifdrawEchart = false
let finger_wrist_min_diff_time = 0
var height,width

//实时获取计算的数据
function updateStats() {
           $.get("/get_shooting_result", function(data) {
//                    $('#elbow_min_angle').text(data.result.elbow_min_angle + "°");
//                    $('#knee_min_angle').text(data.result.knee_min_angle + "°");
//                    $('#ankle_min_angle').text(data.result.ankle_min_angle + "°");
//                    $('#shoulder_min_angle').text(data.result.shoulder_min_angle + "°");
//
                    $('#time_diff').text(data.result.time_diff + "秒");
                    $('#time_diff_score').text(data.score.time_diff);
                    $('#ball_speed').text(data.result.addv)
                    $('#foot_dis').text(data.result.foot_dis)
//
//
//                    $('#elbow_max_angle').text(data.result.elbow_max_angle + "°");
//                    $('#knee_max_angle').text(data.result.knee_max_angle + "°");
//                    $('#ankle_max_angle').text(data.result.ankle_max_angle + "°");
//                    $('#shoulder_max_angle').text(data.result.shoulder_max_angle + "°");
//
//                    $('#release_elbow_angle').text(data.result.release_elbow_angle + "°");
//                    $('#release_knee_angle').text(data.result.release_knee_angle + "°");
//                    $('#release_ankle_angle').text(data.result.release_ankle_angle + "°");
//                    $('#release_shoulder_angle').text(data.result.release_shoulder_angle + "°");

//                    $('#elbow_time_diff').text(data.result.elbow_time_diff + "秒");
//                    $('#knee_time_diff').text(data.result.knee_time_diff + "秒");
//                    $('#ankle_time_diff').text(data.result.ankle_time_diff + "秒");
//                    $('#shoulder_time_diff').text(data.result.shoulder_time_diff + "秒");

                    $('#elbow_min_angle_score').text(data.score.elbow_min_angle);
                    $('#elbow_max_angle_score').text(data.score.elbow_max_angle);
                    $('#release_elbow_angle_score').text(data.score.release_elbow_angle);
                    $('#elbow_time_diff_score').text(data.score.elbow_time_diff);

                    $('#knee_min_angle_score').text(data.score.knee_min_angle);
                    $('#knee_max_angle_score').text(data.score.knee_max_angle);
                    $('#release_knee_angle_score').text(data.score.release_knee_angle);
                    $('#knee_time_diff_score').text(data.score.knee_time_diff);

                    $('#shoulder_min_angle_score').text(data.score.shoulder_min_angle);
                    $('#shoulder_max_angle_score').text(data.score.shoulder_max_angle);
                    $('#release_shoulder_angle_score').text(data.score.release_shoulder_angle);
                    $('#shoulder_time_diff_score').text(data.score.shoulder_time_diff);

                    $('#ankle_min_angle_score').text(data.score.ankle_min_angle);
                    $('#ankle_max_angle_score').text(data.score.ankle_max_angle);
                    $('#release_ankle_angle_score').text(data.score.release_ankle_angle);
                    $('#ankle_time_diff_score').text(data.score.ankle_time_diff);

                    release_time = data.result.release_time
//                    down_time = data.result.down_time

                    //是否该获取轨迹图
                    if(data.result.draw_track) {
                        // 获取要更改的容器元素
                        var container = document.getElementById('track-container')
                        // 创建一个新的图片元素
                        var imgElement1 = document.createElement('img')
                        // 设置图片元素的src属性为本地文件路径
                        imgElement1.src = '../static/img/track.jpg'
                        // 获取div块的宽度和高度
                        var divWidth = container.clientWidth
                        var divHeight = container.clientHeight
                        // 设置图片元素的宽度和高度与div块一样大
                        imgElement1.width = divWidth
                        imgElement1.height = divHeight
                        // 清空容器并将图片元素添加到容器中
                        container.innerHTML = '' // 清空容器
                        container.appendChild(imgElement1) // 将图片元素添加到容器中

                        var imgElement2 = document.createElement('img')
                        imgElement2.src = '../static/img/body_track.jpg'
                        imgElement2.width = divWidth
                        imgElement2.height = divHeight
                        container = document.getElementById('bodytrack-container')
                        container.innerHTML = ''
                        container.appendChild(imgElement2)

                        var imgElement3 = document.createElement('img')
                        imgElement3.src = '../static/img/wrist_track.jpg'
                        imgElement3.width = divWidth
                        imgElement3.height = divHeight
                        container = document.getElementById('wristtrack-container')
                        container.innerHTML = ''
                        container.appendChild(imgElement3)

                        var imgElement4 = document.createElement('img')
                        imgElement4.src = '../static/img/elbow_track.jpg'
                        imgElement4.width = divWidth
                        imgElement4.height = divHeight
                        container = document.getElementById('elbowtrack-container')
                        container.innerHTML = ''
                        container.appendChild(imgElement4)

                    }

                    //评估数据显示
                    if(!data.result.isEnd) return 0
                    isEnd = true
                    //手肘
                    scoreText = ''
                    if(data.score.elbow_min_angle!='标准' && data.score.elbow_min_angle!='待评价'){
                        $('#elbow_min_angle_score').css('color','red')
                        scoreText = scoreText+'<br>'+data.score.elbow_min_angle
                    }else if(data.score.elbow_min_angle=='标准'){
                        $('#elbow_min_angle_score').css('color','green')
                    }else{
                        $('#elbow_min_angle_score').css('color','blue')
                    }
                    if(data.score.elbow_max_angle!='标准' && data.score.elbow_max_angle!='待评价'){
                        $('#elbow_max_angle_score').css('color','red')
                        scoreText = scoreText+'<br>'+data.score.elbow_max_angle
                    }else if(data.score.elbow_max_angle=='标准'){
                        $('#elbow_max_angle_score').css('color','green')
                    }else{
                        $('#elbow_max_angle_score').css('color','blue')
                    }
                    if(data.score.release_elbow_angle!='标准' && data.score.release_elbow_angle!='待评价'){
                        $('#release_elbow_angle_score').css('color','red')
                        scoreText = scoreText+'<br>'+data.score.release_elbow_angle
                    }else if(data.score.release_elbow_angle=='标准'){
                        $('#release_elbow_angle_score').css('color','green')
                    }else{
                        $('#release_elbow_angle_score').css('color','blue')
                    }
                    if(data.score.elbow_time_diff!='标准' && data.score.elbow_time_diff!='待评价'){
                        $('#elbow_time_diff_score').css('color','red')
                        scoreText = scoreText+'<br>'+data.score.elbow_time_diff
                    }else if(data.score.elbow_time_diff=='标准'){
                        $('#elbow_time_diff_score').css('color','green')
                    }else{
                        $('#elbow_time_diff_score').css('color','blue')
                    }

                    //膝盖
                    if(data.score.knee_min_angle!='标准' && data.score.knee_min_angle!='待评价'){
                        $('#knee_min_angle_score').css('color','red')
                        scoreText = scoreText+'<br>'+data.score.knee_min_angle
                    }else if(data.score.knee_min_angle=='标准'){
                        $('#knee_min_angle_score').css('color','green')
                    }else{
                        $('#knee_min_angle_score').css('color','blue')
                    }
                    if(data.score.knee_max_angle!='标准' && data.score.knee_max_angle!='待评价'){
                        $('#knee_max_angle_score').css('color','red')
                        scoreText = scoreText+'<br>'+data.score.knee_max_angle
                    }else if(data.score.knee_max_angle=='标准'){
                        $('#knee_max_angle_score').css('color','green')
                    }else{
                        $('#knee_max_angle_score').css('color','blue')
                    }
                    if(data.score.release_knee_angle!='标准' && data.score.release_knee_angle!='待评价'){
                        $('#release_knee_angle_score').css('color','red')
                        scoreText = scoreText+'<br>'+data.score.release_knee_angle
                    }else if(data.score.release_knee_angle=='标准'){
                        $('#release_knee_angle_score').css('color','green')
                    }else{
                        $('#release_knee_angle_score').css('color','blue')
                    }
                    if(data.score.knee_time_diff!='标准' && data.score.knee_time_diff!='待评价'){
                        $('#knee_time_diff_score').css('color','red')
                         scoreText = scoreText+'<br>'+data.score.knee_time_diff
                    }else if(data.score.knee_time_diff=='标准'){
                        $('#knee_time_diff_score').css('color','green')
                    }else{
                        $('#knee_time_diff_score').css('color','blue')
                    }

                    //脚踝
                    if(data.score.ankle_min_angle!='标准' && data.score.ankle_min_angle!='-'){
                        $('#ankle_min_angle_score').css('color','red')
                        scoreText = scoreText+'<br>'+data.score.ankle_min_angle
                    }else if(data.score.ankle_min_angle=='标准'){
                        $('#ankle_min_angle_score').css('color','green')
                    }else{
                        $('#ankle_min_angle_score').css('color','blue')
                    }
                    if(data.score.ankle_max_angle!='标准' && data.score.ankle_max_angle!='-'){
                        $('#ankle_max_angle_score').css('color','red')
                        scoreText = scoreText+'<br>'+data.score.ankle_max_angle
                    }else if(data.score.ankle_max_angle=='标准'){
                        $('#ankle_max_angle_score').css('color','green')
                    }else{
                        $('#ankle_max_angle_score').css('color','blue')
                    }
                    if(data.score.release_ankle_angle!='标准' && data.score.release_ankle_angle!='-'){
                        $('#release_ankle_angle_score').css('color','red')
                        scoreText = scoreText+'<br>'+data.score.release_ankle_angle
                    }else if(data.score.release_ankle_angle=='标准'){
                        $('#release_ankle_angle_score').css('color','green')
                    }else{
                        $('#release_ankle_angle_score').css('color','blue')
                    }
                    if(data.score.ankle_time_diff!='标准' && data.score.ankle_time_diff!='待评价'){
                        $('#ankle_time_diff_score').css('color','red')
                        scoreText = scoreText+'<br>'+data.score.ankle_time_diff
                    }else if(data.score.ankle_time_diff=='标准'){
                        $('#ankle_time_diff_score').css('color','green')
                    }else{
                        $('#ankle_time_diff_score').css('color','blue')
                    }

                    //肩膀
                    if(data.score.shoulder_min_angle!='标准' && data.score.shoulder_min_angle!='-'){
                        $('#shoulder_min_angle_score').css('color','red')
                        scoreText = scoreText+'<br>'+data.score.shoulder_min_angle
                    }else if(data.score.shoulder_min_angle=='标准'){
                        $('#shoulder_min_angle_score').css('color','green')
                    }else{
                        $('#shoulder_min_angle_score').css('color','blue')
                    }
                    if(data.score.shoulder_max_angle!='标准' && data.score.ankle_max_angle!='-'){
                        $('#shoulder_max_angle_score').css('color','red')
                        scoreText = scoreText+'<br>'+data.score.ankle_max_angle
                    }else if(data.score.shoulder_max_angle=='标准'){
                        $('#shoulder_max_angle_score').css('color','green')
                    }else{
                        $('#shoulder_max_angle_score').css('color','blue')
                    }
                    if(data.score.release_shoulder_angle!='标准' && data.score.release_shoulder_angle!='-'){
                        $('#release_shoulder_angle_score').css('color','red')
                        scoreText = scoreText+'<br>'+data.score.release_shoulder_angle
                    }else if(data.score.release_shoulder_angle=='标准'){
                        $('#release_shoulder_angle_score').css('color','green')
                    }else{
                        $('#release_shoulder_angle_score').css('color','blue')
                    }
                    if(data.score.shoulder_time_diff!='标准' && data.score.shoulder_time_diff!='待评价'){
                        $('#shoulder_time_diff_score').css('color','red')
                        scoreText = scoreText+'<br>'+data.score.shoulder_time_diff
                    }else if(data.score.shoulder_time_diff=='标准'){
                        $('#shoulder_time_diff_score').css('color','green')
                    }else{
                        $('#shoulder_time_diff_score').css('color','blue')
                    }

                    if(data.score.time_diff!='标准' && data.score.time_diff!='待评价'){
                        $('#time_diff_score').css('color','red')
                        scoreText = scoreText+'<br>'+data.score.time_diff
                    }else if(data.score.time_diff=='标准'){
                        $('#time_diff_score').css('color','green')
                    }else{
                        $('#time_diff_score').css('color','blue')
                    }

                    // 获取元素引用
                    let textElement = document.getElementById("textElement")
                    // 更新文字内容
                    // 设置字体大小
                    textElement.style.fontSize = "20px"
                    textElement.style.fontStyle = "italic"
                    textElement.innerHTML = scoreText

                    if (isEnd){
                        draw_echart();
                    }
                    return isEnd;
            });
        }

//滑动窗口平滑
function movingAverage(data, windowSize) {
    let result = [];
    for (let i = 0; i < data.length; i++) {
        let start = Math.max(0, i - Math.floor(windowSize / 2));
        let end = Math.min(data.length, i + Math.floor(windowSize / 2) + 1)
        let sum = 0;
        for (let j = start; j < end; j++) {
            sum += data[j];
        }
        result.push(Number((sum / (end - start)).toFixed(2)))
    }
    return result
}

function analyzeJointData(timestamps, jointData) {
//    let maxVal = Math.max(...jointData)
//    let minVal = Math.min(...jointData)
//
//    let maxTime = timestamps[jointData.indexOf(maxVal)];
//    let minTime = timestamps[jointData.indexOf(minVal)];
    let maxVal = -100
    let minVal = 5000
    let maxTime = 0
    let minTime = 0
    for (let i=0;i<jointData.length;i++){
//        console.log("Current jointData value:", jointData[i],timestamps[i])
        if (jointData[i]>maxVal){
            maxVal = jointData[i]
            maxTime = timestamps[i]
        }
        if (jointData[i]<minVal){
            minVal = jointData[i]
            minTime = timestamps[i]
        }
    }
    return {
        max: {
            value: maxVal,
            time: maxTime
        },
        min: {
            value: minVal,
            time: minTime
        }
    }
}

function findIndices(timestamps, release_time, down_time) {
    console.log(timestamps, release_time, down_time)
    let releaseIndex = timestamps.indexOf(release_time)
    let downIndex = timestamps.indexOf(down_time)

    // If not found, set to null
    if (releaseIndex === -1) releaseIndex = 0
    if (downIndex === -1) downIndex = 0

    release_index = releaseIndex
    down_index = downIndex
}

function draw_heigt_chart(){
    // 获取图表容器元素
    var chartContainer = document.getElementById('height-chart-container');
    // 使用 ECharts 初始化图表
    var chart = echarts.init(chartContainer)
    // 异步请求数据
    fetch('/json_data_pose_xy')
    .then(response => response.json())
    .then(data => {
        let timestamps = Object.keys(data)
        for (let key in data) {
            fingerHeight.push(data[key][4][1])
            wristHeight.push(data[key][3][1])
            ankleHeight.push(data[key][0][1])
        }
        smt_fingerHeight = movingAverage(fingerHeight, 3)
        smt_wristHeight = movingAverage(wristHeight, 3)
        smt_ankleHeight = movingAverage(ankleHeight, 3)

        fingerResult = analyzeJointData(timestamps, smt_fingerHeight)
//        console.log("手腕",timestamps, smt_wristHeight)
        wristResult = analyzeJointData(timestamps, smt_wristHeight)
//        console.log("手腕结果",timestamps, wristResult)
        ankleResult = analyzeJointData(timestamps, smt_ankleHeight)

        let differences = smt_fingerHeight.map((f, index) => Math.abs(f - smt_wristHeight[index]))
        let minDifference = Math.min(...differences)
        let minDifferenceIndex = differences.indexOf(minDifference)
        finger_wrist_min_diff_time = timestamps[minDifferenceIndex]
        console.log("指尖平齐时间",finger_wrist_min_diff_time)

        down_time = wristResult.min.time
        findIndices(timestamps, release_time, down_time)

//        $('#elbow_min_angle').text(elbowResult.min.value + "°")
//        $('#knee_min_angle').text(kneeResult.min.value + "°")
//        $('#ankle_min_angle').text(ankleResult.min.value + "°")
//        $('#shoulder_min_angle').text(shoulderResult.min.value + "°")
//        $('#wrist_min_angle').text(wristResult.min.value + "°")
//
//        $('#elbow_max_angle').text(elbowResult.max.value + "°")
//        $('#knee_max_angle').text(kneeResult.max.value + "°")
//        $('#ankle_max_angle').text(ankleResult.max.value + "°")
//        $('#shoulder_max_angle').text(shoulderResult.max.value + "°")
//        $('#wrist_max_angle').text(wristResult.max.value + "°")
//
//        $('#release_elbow_angle').text(smt_elbowAngle[release_index] + "°")
//        $('#release_knee_angle').text(smt_kneeAngle[release_index] + "°")
//        $('#release_ankle_angle').text(smt_ankleAngle[release_index] + "°")
//        $('#release_shoulder_angle').text(smt_shoulderAngle[release_index] + "°")
//        $('#release_wrist_angle').text(smt_wristAngle[release_index] + "°")
//
//        let difference = wristResult.min.time - wristResult.max.time
//        $('#wrist_time_diff').text(Math.abs(difference).toFixed(2) + "秒")
//        difference = elbowResult.min.time - elbowResult.max.time
//        $('#elbow_time_diff').text(Math.abs(difference).toFixed(2) + "秒")
//        difference = shoulderResult.min.time - shoulderResult.max.time
//        $('#shoulder_time_diff').text(Math.abs(difference).toFixed(2) + "秒")
//        difference = kneeResult.min.time - kneeResult.max.time
//        $('#knee_time_diff').text(Math.abs(difference).toFixed(2) + "秒")
//        difference = ankleResult.min.time - ankleResult.max.time
//        $('#ankle_time_diff').text(Math.abs(difference).toFixed(2) + "秒")

        console.log(release_index)
          // 转换数据为适用于 ECharts 的格式
          // 使用 timestamps 和 smoothedJoint 数据创建新的 transformedData
        var transformedData = timestamps.map((x, index) => {
            return {
                x: x,
                y1: smt_fingerHeight[index],
                y2: smt_wristHeight[index],
                y3: smt_ankleHeight[index],
            }
        })
        // 配置图表
        var options = {
            xAxis: {
                type: 'category',
                data: transformedData.map(item => item.x),
            },
            yAxis: {
                type: 'value'
            },
            series: [
                {
                    name: '指尖',
                    type: 'line',
                    data: transformedData.map(item => item.y1),
                    smooth: false,
                    symbol: 'none',  // 不显示每个数据点的圆
//                    markPoint: {
//                        data: [{
//                            type: 'max',
//                            name: '最大值'
//                        },
//                        {
//                            type: 'min',
//                            name: '最小值'
//                        }]
//                    },
                    markLine: {
                        silent: true,
                        label: {
                            show: true,  // 显示标签
                            position: 'start',  // 标签的位置，可以是'start', 'middle', 'end'
                            formatter: function(params) {
                                if (params.data.xAxis === release_index) {
                                    return '投球点';  // 设置标签的文字内容
                                } else {
                                    return '发力点';  // 为第二条线设置不同的标签内容
                                }
                            }
                        },
                        data: [{
                            xAxis: release_index
                        },
                        {
                            xAxis: down_index  // 添加第二条线的x轴位置
                        }],
                        lineStyle: {
                            color: 'red',  // 设置线的颜色
                            width: 2       // 设置线的宽度
                        }
                    },

                },
                {
                    name: '手腕',
                    type: 'line',
                    data: transformedData.map(item => item.y2),
                    smooth: false,
                    symbol: 'none',  // 不显示每个数据点的圆
                    markPoint: {
                        data: [{
                            type: 'max',
                            name: '最大值'
                        }],
                    },
                    markLine: {
                        silent: true,
                        label: {
                            show: true,  // 显示标签
                            position: 'end',  // 标签的位置，可以是'start', 'middle', 'end'
                            formatter: function(params) {
                                if (params.data.xAxis === release_index) {
                                    return '投球点';  // 设置标签的文字内容
                                } else {
                                    return '发力点';  // 为第二条线设置不同的标签内容
                                }
                            }
                        },
                        data: [{
                            xAxis: release_index
                        },
                        {
                            xAxis: down_index  // 添加第二条线的x轴位置
                        }],
                        lineStyle: {
                            color: 'red',  // 设置线的颜色
                            width: 2       // 设置线的宽度
                        }
                    }
                },
                {
                    name: '脚踝',
                    type: 'line',
                    data: transformedData.map(item => item.y3),
                    smooth: false,
                    symbol: 'none',  // 不显示每个数据点的圆
                    markPoint: {
                        data: [{
                            type: 'max',
                            name: '最大值'
                        }],
                    },
                },
            ],
            tooltip: {
                trigger: 'axis'
            },
            legend: {
                data: ['指尖','手腕','脚踝']
            },
        }
        // 使用配置项显示图表
        chart.setOption(options)
     })

}

function draw_echart(){
    if(ifdrawEchart){
        return
    }
    ifdrawEchart = true
    // 获取图表容器元素
    var chartContainer = document.getElementById('chart-container');
    // 使用 ECharts 初始化图表
    var chart = echarts.init(chartContainer)
    draw_heigt_chart()
    // 异步请求数据
    fetch('/json_data_pose')
    .then(response => response.json())
    .then(data => {
        let timestamps = Object.keys(data)
        for (let key in data) {
            elbowAngle.push(data[key][0])
            shoulderAngle.push(data[key][1])
            kneeAngle.push(data[key][2])
            ankleAngle.push(data[key][3])
            wristAngle.push(data[key][4])
        }
        smt_elbowAngle = movingAverage(elbowAngle, 3)
        smt_shoulderAngle = movingAverage(shoulderAngle, 3)
        smt_kneeAngle = movingAverage(kneeAngle, 3)
        smt_ankleAngle = movingAverage(ankleAngle, 5)
        smt_wristAngle = movingAverage(wristAngle, 5)

        console.log("胳膊",timestamps, smt_elbowAngle)
        elbowResult = analyzeJointData(timestamps, smt_elbowAngle)
        console.log("肩膀",timestamps, smt_elbowAngle)
        shoulderResult = analyzeJointData(timestamps, smt_shoulderAngle)
        console.log("膝盖",timestamps, smt_elbowAngle)
        kneeResult = analyzeJointData(timestamps, smt_kneeAngle)
        console.log("脚踝",timestamps, smt_elbowAngle)
        ankleResult = analyzeJointData(timestamps, smt_ankleAngle)
        console.log("手腕",timestamps, smt_elbowAngle)
        wristResult = analyzeJointData(timestamps, smt_wristAngle)
        console.log("Joint 2 Analysis:", wristResult)
//        findIndices(timestamps, release_time, down_time)

        $('#elbow_min_angle').text(elbowResult.min.value + "°")
        $('#knee_min_angle').text(kneeResult.min.value + "°")
        $('#ankle_min_angle').text(ankleResult.min.value + "°")
        $('#shoulder_min_angle').text(shoulderResult.min.value + "°")
        $('#wrist_min_angle').text(wristResult.min.value + "°")

        $('#elbow_max_angle').text(elbowResult.max.value + "°")
        $('#knee_max_angle').text(kneeResult.max.value + "°")
        $('#ankle_max_angle').text(ankleResult.max.value + "°")
        $('#shoulder_max_angle').text(shoulderResult.max.value + "°")
        $('#wrist_max_angle').text(wristResult.max.value + "°")

        $('#release_elbow_angle').text(smt_elbowAngle[release_index] + "°")
        $('#release_knee_angle').text(smt_kneeAngle[release_index] + "°")
        $('#release_ankle_angle').text(smt_ankleAngle[release_index] + "°")
        $('#release_shoulder_angle').text(smt_shoulderAngle[release_index] + "°")
        $('#release_wrist_angle').text(smt_wristAngle[release_index] + "°")


        let difference = wristResult.min.time - wristResult.max.time
        $('#wrist_time_diff').text(Math.abs(difference).toFixed(2) + "秒")
        difference = elbowResult.min.time - elbowResult.max.time
        $('#elbow_time_diff').text(Math.abs(difference).toFixed(2) + "秒")
        difference = shoulderResult.min.time - shoulderResult.max.time
        $('#shoulder_time_diff').text(Math.abs(difference).toFixed(2) + "秒")
        difference = kneeResult.min.time - kneeResult.max.time
        $('#knee_time_diff').text(Math.abs(difference).toFixed(2) + "秒")
        difference = ankleResult.min.time - ankleResult.max.time
        $('#ankle_time_diff').text(Math.abs(difference).toFixed(2) + "秒")

        console.log(release_index)
          // 转换数据为适用于 ECharts 的格式
          // 使用 timestamps 和 smoothedJoint 数据创建新的 transformedData
        var transformedData = timestamps.map((x, index) => {
            return {
                x: x,
                y1: smt_elbowAngle[index],
                y2: smt_shoulderAngle[index],
                y3: smt_kneeAngle[index],
                y4: smt_ankleAngle[index],
                y5: smt_wristAngle[index]
            }
        })
        // 配置图表
        var options = {
            xAxis: {
                type: 'category',
                data: transformedData.map(item => item.x),
            },
            yAxis: {
                type: 'value'
            },
            series: [
                {
                    name: '手肘',
                    type: 'line',
                    data: transformedData.map(item => item.y1),
                    smooth: false,
                    symbol: 'none',  // 不显示每个数据点的圆
                    markPoint: {
                        data: [{
                            type: 'max',
                            name: '最大值'
                        },
                        {
                            type: 'min',
                            name: '最小值'
                        }]
                    },
                    markLine: {
                        silent: true,
                        label: {
                            show: true,  // 显示标签
                            position: 'start',  // 标签的位置，可以是'start', 'middle', 'end'
                            formatter: function(params) {
                                if (params.data.xAxis === release_index) {
                                    return '投球点';  // 设置标签的文字内容
                                } else {
                                    return '发力点';  // 为第二条线设置不同的标签内容
                                }
                            }
                        },
                        data: [{
                            xAxis: release_index
                        },
                        {
                            xAxis: down_index  // 添加第二条线的x轴位置
                        }],
                        lineStyle: {
                            color: 'red',  // 设置线的颜色
                            width: 2       // 设置线的宽度
                        }
                    },

                },
                {
                    name: '肩膀',
                    type: 'line',
                    data: transformedData.map(item => item.y2),
                    smooth: false,
                    symbol: 'none',  // 不显示每个数据点的圆
                    markPoint: {
                        data: [{
                            type: 'max',
                            name: '最大值'
                        },
                        {
                            type: 'min',
                            name: '最小值'
                        }]
                    },
                    markLine: {
                        silent: true,
                        label: {
                            show: true,  // 显示标签
                            position: 'end',  // 标签的位置，可以是'start', 'middle', 'end'
                            formatter: function(params) {
                                if (params.data.xAxis === release_index) {
                                    return '投球点';  // 设置标签的文字内容
                                } else {
                                    return '发力点';  // 为第二条线设置不同的标签内容
                                }
                            }
                        },
                        data: [{
                            xAxis: release_index
                        },
                        {
                            xAxis: down_index  // 添加第二条线的x轴位置
                        }],
                        lineStyle: {
                            color: 'red',  // 设置线的颜色
                            width: 2       // 设置线的宽度
                        }
                    }
                },
                {
                    name: '膝盖',
                    type: 'line',
                    data: transformedData.map(item => item.y3),
                    smooth: false,
                    symbol: 'none',  // 不显示每个数据点的圆
                    markPoint: {
                        data: [{
                            type: 'max',
                            name: '最大值'
                        },
                        {
                            type: 'min',
                            name: '最小值'
                        }]
                    },
                },
                {
                    name: '脚踝',
                    type: 'line',
                    data: transformedData.map(item => item.y4),
                    smooth: false,
                    symbol: 'none',  // 不显示每个数据点的圆
                    markPoint: {
                        data: [{
                            type: 'max',
                            name: '最大值'
                        },
                        {
                            type: 'min',
                            name: '最小值'
                        }]
                    },

                },
                {
                    name: '手腕',
                    type: 'line',
                    data: transformedData.map(item => item.y5),
                    smooth: false,
                    symbol: 'none',  // 不显示每个数据点的圆
                    markPoint: {
                        data: [{
                            type: 'max',
                            name: '最大值'
                        },
                        {
                            type: 'min',
                            name: '最小值'
                        }]
                    },

                },
            ],
            tooltip: {
                trigger: 'axis'
            },
            legend: {
                data: ['手肘', '肩膀','膝盖','脚踝','手腕']
            },
        }
        // 使用配置项显示图表
        chart.setOption(options)
        document.getElementById("realreplay").disabled = false
//        document.getElementById("realplay").disabled = false
//        document.getElementById("realstop").disabled = false
//        document.getElementById("realspeed").disabled = false
    })
    .catch(error => {
       console.error('请求数据时发生错误:', error);
    })
}

function showVideo() {
    // 获取img和video元素
    var imgElement = document.getElementById('video');
    var videoElement = document.getElementById('resultvideo');
    // 隐藏img元素
    imgElement.style.display = 'none'
    // 显示video元素
    videoElement.style.display = 'block'
    // 设置video元素的src属性
    videoElement.src = realvideoUrl // 替换为你的视频URL
    // 获取 img 元素的宽度和高度
//    var videoWidth = videoElement.clientWidth // 获取元素的客户端宽度
//    var videoHeight = videoElement.clientHeight // 获取元素的客户端高度
//    videoElement.style.width = videoWidth + 'px' // 设置新宽度
//    videoElement.style.height = videoHeight + 'px' // 设置新高度
    videoElement.load()
    document.getElementById("realspeed").disabled = false
    document.getElementById("togetherplay").disabled = false
    document.getElementById("realreplay").disabled = true
    videoElement.play()

//    获取手部视频
    var realhandElement = document.getElementById('realhandvideo')
    var stdhandElement = document.getElementById('stdhandvideo')
    realhandElement.src = realhandUrl
    stdhandElement.src = stdhandUrl
    realhandElement.style.width = '500px' // 设置新宽度
    realhandElement.style.height = '500px' // 设置新高度
    stdhandElement.style.width = '500px' // 设置新宽度
    stdhandElement.style.height = '500px' // 设置新高度
    document.getElementById("handreplay").disabled = false
}

function togetherplay(){
    var videoElement = document.getElementById('resultvideo')
    var stdvideoElement = document.getElementById('stdvideo')
    if(videoElement.readyState > 0){
        videoElement.play()
    }
    if(stdvideoElement.readyState > 0){
        stdvideoElement.play()
    }
}

function handtogetherplay(){
    var videoElement = document.getElementById('realhandvideo')
    var stdvideoElement = document.getElementById('stdhandvideo')
    if(videoElement.readyState > 0){
        videoElement.play()
    }
    if(stdvideoElement.readyState > 0){
        stdvideoElement.play()
    }
}

function stopVideo() {
    var videoElement = document.getElementById('resultvideo')
    if(videoElement.readyState > 0){
        videoElement.pause()
    }
}

function playVideo() {
    var videoElement = document.getElementById('resultvideo')
    if(videoElement.readyState > 0){
        videoElement.play()
    }
}

document.addEventListener('DOMContentLoaded', function(event) {
    // 获取返回按钮的元素，可以是一个按钮或其他触发返回操作的元素
    var backButton = document.getElementById('closeBtn')
    // 获取 img 元素
    var imgElement = document.getElementById('video')
    var realElement = document.getElementById('resultvideo')
    var stdElement = document.getElementById('stdvideo')
    var realhandElement = document.getElementById('realhandvideo')
    var stdhandElement = document.getElementById('stdhandvideo')
    stdElement.src = videoUrl
    //获取速度按钮
    var speedButton  = document.getElementById('realspeed')
    var speedButton2  = document.getElementById('stdspeed')
    var handspeedButton  = document.getElementById('realhandspeed')
    var handspeedButton2  = document.getElementById('stdhandspeed')
    // 获取 img 元素的宽度和高度
    var imgWidth = imgElement.clientWidth // 获取元素的客户端宽度
    var imgHeight = imgElement.clientHeight // 获取元素的客户端高度
    imgElement.style.width = imgWidth + 'px' // 设置新宽度
    imgElement.style.height = imgHeight + 'px' // 设置新高度
    realElement.style.width = imgWidth + 'px' // 设置新宽度
    realElement.style.height = imgHeight + 'px' // 设置新高度
    var stdWidth = stdElement.clientWidth // 获取元素的客户端宽度
    var stdHeight = stdElement.clientHeight // 获取元素的客户端高度
    stdElement.style.width = stdWidth + 'px' // 设置新宽度
    stdElement.style.height = stdHeight + 'px' // 设置新高度
    // 添加点击事件监听器
    backButton.addEventListener('click', function() {
        // 返回上一页
        window.location.href = '/'
    })
    // 获取所有具有 data-speed 属性的 <a> 元素
    var speedLinks = document.querySelectorAll('a[data-speed]')

    // 为每个 <a> 元素添加点击事件监听器
    speedLinks.forEach(function(link) {
        link.addEventListener('click', function(event) {
            // 阻止 <a> 元素的默认行为（导航到 href）
            event.preventDefault();

            // 获取播放速度
            var speed = parseFloat(this.getAttribute('data-speed'));
            console.log(speed);

            // 设置视频播放速度
            realElement.playbackRate = speed;

            // 更新按钮文本
            speedButton.innerText = speed + '倍速';
        })
    })

    // 获取所有具有 data-speed 属性的 <a> 元素
    var speedLinks2 = document.querySelectorAll('a[stddata-speed]')
    // 为每个 <a> 元素添加点击事件监听器
    speedLinks2.forEach(function(link) {
        link.addEventListener('click', function(event) {
            // 阻止 <a> 元素的默认行为（导航到 href）
            event.preventDefault();

            // 获取播放速度
            var speed = parseFloat(this.getAttribute('stddata-speed'));
            console.log(speed);

            // 设置视频播放速度
            stdElement.playbackRate = speed;

            // 更新按钮文本
            speedButton2.innerText = speed + '倍速';
        })
    })

    // 获取所有具有 data-speed 属性的 <a> 元素
    var speedLinks3 = document.querySelectorAll('a[handdata-speed]')
    // 为每个 <a> 元素添加点击事件监听器
    speedLinks3.forEach(function(link) {
        link.addEventListener('click', function(event) {
            // 阻止 <a> 元素的默认行为（导航到 href）
            event.preventDefault();

            // 获取播放速度
            var speed = parseFloat(this.getAttribute('handdata-speed'));
            console.log(speed);

            // 设置视频播放速度
            realhandElement.playbackRate = speed;

            // 更新按钮文本
            handspeedButton.innerText = speed + '倍速';
        })
    })

    // 获取所有具有 data-speed 属性的 <a> 元素
    var speedLinks4 = document.querySelectorAll('a[stdhanddata-speed]')
    // 为每个 <a> 元素添加点击事件监听器
    speedLinks4.forEach(function(link) {
        link.addEventListener('click', function(event) {
            // 阻止 <a> 元素的默认行为（导航到 href）
            event.preventDefault();

            // 获取播放速度
            var speed = parseFloat(this.getAttribute('stdhanddata-speed'));
            console.log(speed);

            // 设置视频播放速度
            stdhandElement.playbackRate = speed

            // 更新按钮文本
            handspeedButton2.innerText = speed + '倍速'
        })
    })

    var intervalId = setInterval(function() {
    updateStats() // 每隔0.5秒，调用 updateStats 函数
    if (isEnd) {
        clearInterval(intervalId) //停止定时器
    }
}, 500)
})


