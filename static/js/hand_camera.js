// JavaScript 部分
let video = document.getElementById('video')
let mediaRecorder
let chunks = []
const overlayDiv = document.getElementById("overlayDiv")
const timeDisplay = document.getElementById("time")
let timerInterval
let elapsedTime = 0
const modal = document.getElementById("staticBackdrop")

  // 获取视频流
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
      video.srcObject = stream;
      mediaRecorder = new MediaRecorder(stream);

      // 监听录制数据
      mediaRecorder.ondataavailable = handleDataAvailable;
    })
    .catch(error => {
      console.error(error);
    });

  // 开始录制
  function startRecording() {
    // 显示大 div 块
    overlayDiv.style.display = "block"
     // 关闭模态框
       $('#staticBackdrop').modal('hide')
    // 启动计时器，每秒更新时间显示
    timerInterval = setInterval(() => {
        elapsedTime++;
        const minutes = Math.floor(elapsedTime / 60)
        const seconds = elapsedTime % 60
        timeDisplay.textContent = `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`
    }, 1000);

    chunks = []
    mediaRecorder.start()
  }

  // 停止录制
  function stopRecording() {
    // 停止计时器
    clearInterval(timerInterval)
    // 隐藏大 div 块
    overlayDiv.style.display = "none"
    // 重置计时器
    elapsedTime = 0
    timeDisplay.textContent = "00:00"
    mediaRecorder.stop()
  }

  // 处理录制数据
  function handleDataAvailable(event) {
    chunks.push(event.data)
    if (mediaRecorder.state === 'inactive') {
      // 录制结束，创建 Blob 并发送给 Flask 服务器
      let blob = new Blob(chunks, { type: 'video/mp4' })
      sendToServer(blob)
    }
  }

  // 发送数据给 Flask 服务器
  function sendToServer(blob) {
    let formData = new FormData()
    formData.append('video', blob, 'recorded.mp4')

    fetch('/upload_hand', {
      method: 'POST',
      body: formData
    })
    .then(response => {
  if (response.status === 200) {
    window.location.href = '/videoer'
  } else {
    throw new Error('请求失败')
  }
})
}


