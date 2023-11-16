var isEnd = false;

function medianFilter(array, windowSize) {
  // 计算中值的方法
  const getMedian = (items) => {
    return items.sort((a, b) => a - b)[Math.floor(items.length / 2)];
  }

  const result = [];
  // 中值滤波
  for (let i = 0; i < array.length; i++) {
    const window = array.slice(i, i + windowSize);
    const median = getMedian(window);
    result.push(median);
  }
  return result;
}

//实时获取计算的数据
function updateStats() {
            $.get("/get_hand_result", function(data) {
                    $('#elbow_min_angle').text(data.result.elbow_min_angle + "°");
                    $('#shoulder_min_angle').text(data.result.shoulder_min_angle + "°");
                    $('#wrist_min_angle').text(data.result.wrist_min_angle + "°");
                    $('#hand_min_angle').text(data.result.hand_min_angle + "°");
                    $('#finger_min_angle').text(data.result.finger_min_angle + "°");

                    $('#elbow_max_angle').text(data.result.elbow_max_angle + "°");
                    $('#shoulder_max_angle').text(data.result.shoulder_max_angle + "°");
                    $('#wrist_max_angle').text(data.result.wrist_max_angle + "°");
                    $('#hand_max_angle').text(data.result.hand_max_angle + "°");
                    $('#finger_max_angle').text(data.result.finger_max_angle + "°");

//                    $('#elbow_min_angle_score').text(data.score.elbow_min_angle);
//                    $('#elbow_max_angle_score').text(data.score.elbow_max_angle);
//                    $('#release_elbow_angle_score').text(data.score.release_elbow_angle);
//
//                    $('#knee_min_angle_score').text(data.score.knee_min_angle);
//                    $('#knee_max_angle_score').text(data.score.knee_max_angle);
//                    $('#release_knee_angle_score').text(data.score.release_knee_angle);
//
//                    $('#shoulder_min_angle_score').text(data.score.shoulder_min_angle);
//                    $('#shoulder_max_angle_score').text(data.score.shoulder_max_angle);
//                    $('#release_shoulder_angle_score').text(data.score.release_shoulder_angle);
//
//                    $('#ankle_min_angle_score').text(data.score.ankle_min_angle);
//                    $('#ankle_max_angle_score').text(data.score.ankle_max_angle);
//                    $('#release_ankle_angle_score').text(data.score.release_ankle_angle);

                    //评估数据显示
                    if(!data.result.isEnd) return 0;
                    isEnd = true;
                    //手肘
//                    if(data.score.elbow_min_angle!='标准' && data.score.elbow_min_angle!='待评价'){
//                        $('#elbow_min_angle_score').css('color','red')
//                    }else if(data.score.elbow_min_angle=='标准'){
//                        $('#elbow_min_angle_score').css('color','green')
//                    }else{
//                        $('#elbow_min_angle_score').css('color','blue')
//                    }
//                    if(data.score.elbow_max_angle!='标准' && data.score.elbow_max_angle!='待评价'){
//                        $('#elbow_max_angle_score').css('color','red')
//                    }else if(data.score.elbow_max_angle=='标准'){
//                        $('#elbow_max_angle_score').css('color','green')
//                    }else{
//                        $('#elbow_max_angle_score').css('color','blue')
//                    }
//                    if(data.score.release_elbow_angle!='标准' && data.score.release_elbow_angle!='待评价'){
//                        $('#release_elbow_angle_score').css('color','red')
//                    }else if(data.score.release_elbow_angle=='标准'){
//                        $('#release_elbow_angle_score').css('color','green')
//                    }else{
//                        $('#release_elbow_angle_score').css('color','blue')
//                    }
//
//                    //膝盖
//                    if(data.score.knee_min_angle!='标准' && data.score.knee_min_angle!='待评价'){
//                        $('#knee_min_angle_score').css('color','red')
//                    }else if(data.score.knee_min_angle=='标准'){
//                        $('#knee_min_angle_score').css('color','green')
//                    }else{
//                        $('#knee_min_angle_score').css('color','blue')
//                    }
//                    if(data.score.knee_max_angle!='标准' && data.score.knee_max_angle!='待评价'){
//                        $('#knee_max_angle_score').css('color','red')
//                    }else if(data.score.knee_max_angle=='标准'){
//                        $('#knee_max_angle_score').css('color','green')
//                    }else{
//                        $('#knee_max_angle_score').css('color','blue')
//                    }
//                    if(data.score.release_knee_angle!='标准' && data.score.release_knee_angle!='待评价'){
//                        $('#release_knee_angle_score').css('color','red')
//                    }else if(data.score.release_knee_angle=='标准'){
//                        $('#release_knee_angle_score').css('color','green')
//                    }else{
//                        $('#release_knee_angle_score').css('color','blue')
//                    }
//
//                    //脚踝
//                    if(data.score.ankle_min_angle!='标准' && data.score.ankle_min_angle!='-'){
//                        $('#ankle_min_angle_score').css('color','red')
//                    }else if(data.score.ankle_min_angle=='标准'){
//                        $('#ankle_min_angle_score').css('color','green')
//                    }else{
//                        $('#ankle_min_angle_score').css('color','blue')
//                    }
//                    if(data.score.ankle_max_angle!='标准' && data.score.ankle_max_angle!='-'){
//                        $('#ankle_max_angle_score').css('color','red')
//                    }else if(data.score.ankle_max_angle=='标准'){
//                        $('#ankle_max_angle_score').css('color','green')
//                    }else{
//                        $('#ankle_max_angle_score').css('color','blue')
//                    }
//                    if(data.score.release_ankle_angle!='标准' && data.score.release_ankle_angle!='-'){
//                        $('#release_ankle_angle_score').css('color','red')
//                    }else if(data.score.release_ankle_angle=='标准'){
//                        $('#release_ankle_angle_score').css('color','green')
//                    }else{
//                        $('#release_ankle_angle_score').css('color','blue')
//                    }
//
//                    //肩膀
//                    if(data.score.shoulder_min_angle!='标准' && data.score.shoulder_min_angle!='-'){
//                        $('#shoulder_min_angle_score').css('color','red')
//                    }else if(data.score.shoulder_min_angle=='标准'){
//                        $('#shoulder_min_angle_score').css('color','green')
//                    }else{
//                        $('#shoulder_min_angle_score').css('color','blue')
//                    }
//                    if(data.score.shoulder_max_angle!='标准' && data.score.ankle_max_angle!='-'){
//                        $('#shoulder_max_angle_score').css('color','red')
//                    }else if(data.score.shoulder_max_angle=='标准'){
//                        $('#shoulder_max_angle_score').css('color','green')
//                    }else{
//                        $('#shoulder_max_angle_score').css('color','blue')
//                    }
//                    if(data.score.release_shoulder_angle!='标准' && data.score.release_shoulder_angle!='-'){
//                        $('#release_shoulder_angle_score').css('color','red')
//                    }else if(data.score.release_shoulder_angle=='标准'){
//                        $('#release_shoulder_angle_score').css('color','green')
//                    }else{
//                        $('#release_shoulder_angle_score').css('color','blue')
//                    }
                    if (isEnd){
                        draw_echarts();
                    }
                    return isEnd;
            });
        }

function medianFilter(array, windowSize) {
  // 计算中值的方法
  const getMedian = (items) => {
    return items.sort((a, b) => a - b)[Math.floor(items.length / 2)];
  }

  const result = [];
  // 中值滤波
  for (let i = 0; i < array.length; i++) {
    const window = array.slice(i, i + windowSize);
    const median = getMedian(window);
    result.push(median);
  }

  return result;
}

function draw_echarts() {
      // 获取图表容器元素
      var chartContainer = document.getElementById('chart-container');
      // 使用 ECharts 初始化图表
      var chart = echarts.init(chartContainer);
      // 异步请求数据
      fetch('/json_data')
        .then(response => response.json())
        .then(data => {
          // 转换数据为适用于 ECharts 的格式
          var transformedData = Object.entries(data).map(([x, [y1, y2,y3,y4,y5]]) => {
              return {
        x: x,
        y1: y1,
        y2: y2,
        y3: y3,
        y4: y4,
        y5: y5,
      }
          }
          );
          console.log(transformedData.map(item => item.y2))
          console.log(medianFilter(transformedData.map(item => item.y2)))
      // 配置图表
      var options = {
        xAxis: {
          type: 'category',
          data: transformedData.map(item => item.x)
        },
        yAxis: {
          type: 'value'
        },
        series: [
        //     {
        //   name: '手肘',
        //   type: 'line',
        //   data: transformedData.map(item => item.y1)
        // },
          {
          name: '手肘',
          type: 'line',
          data: medianFilter(transformedData.map(item => item.y1),10)
        },
        //   {
        //   name: '肩膀',
        //   type: 'line',
        //   data: transformedData.map(item => item.y2)
        // },
            {
          name: '肩膀',
          type: 'line',
          data: medianFilter(transformedData.map(item => item.y2),10)
        },
          // {
        //   name: '手腕',
        //   type: 'line',
        //   data: transformedData.map(item => item.y3)
        // }
            {
           name: '手腕',
          type: 'line',
          data: medianFilter(transformedData.map(item => item.y3),10)
        },
        {
           name: '手掌',
          type: 'line',
          data: medianFilter(transformedData.map(item => item.y4),10)
        },
        {
           name: '手指',
          type: 'line',
          data: medianFilter(transformedData.map(item => item.y5),10)
        },
        ],
         tooltip: {
    trigger: 'axis'
  },
        legend: {
  data: ['手肘', '肩膀','手腕','手掌','手指']
},
      };
          // 使用配置项显示图表
          chart.setOption(options);
        })
        .catch(error => {
          console.error('请求数据时发生错误:', error);
        });
}

document.addEventListener('DOMContentLoaded', function(event) {
    // 获取返回按钮的元素，可以是一个按钮或其他触发返回操作的元素
    var backButton = document.getElementById('closeBtn');
    // 获取 img 元素
    var imgElement = document.getElementById('video');

    // 获取 img 元素的宽度和高度
    var imgWidth = imgElement.clientWidth; // 获取元素的客户端宽度
    var imgHeight = imgElement.clientHeight; // 获取元素的客户端高度

    imgElement.style.width = imgWidth + 'px'; // 设置新宽度
    imgElement.style.height = imgHeight + 'px'; // 设置新高度
    // 添加点击事件监听器
    backButton.addEventListener('click', function() {
        // 返回上一页
        window.location.href = '/';
    });
    var intervalId = setInterval(function() {
    updateStats(); // 每隔0.5秒，调用 updateStats 函数
    if (isEnd) {
        clearInterval(intervalId); // 如果 isEnd 为 true，停止定时器
    }
}, 500);
});


