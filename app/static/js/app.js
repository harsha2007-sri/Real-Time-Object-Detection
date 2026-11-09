const navItems = document.querySelectorAll('.nav-item');
const panels = document.querySelectorAll('.panel');
const dashboardCards = {
  total: document.getElementById('dashTotalDetections'),
  visible: document.getElementById('dashVisibleObjects'),
  in: document.getElementById('dashInTotal'),
  out: document.getElementById('dashOutTotal'),
};
const liveCounters = {
  in: document.getElementById('liveIn'),
  out: document.getElementById('liveOut'),
  visible: document.getElementById('liveCurrent'),
};
const fpsIndicator = document.getElementById('liveFps');

let charts = {};
let streamActive = false;
let cameraStarting = false;
let ws;
let videoStream;
let framePending = false;
let socketReady = false;
let frameSequence = 0;
let consecutiveFrameErrors = 0;
let retryHandle;
let connectionTimeout;
let cameraCanvas;
let cameraContext;

function updateDashboardCards(stats) {
  const keys = ['total_detections', 'visible_objects', 'total_in', 'total_out'];
  if (!stats || keys.some((key) => !Number.isSafeInteger(stats[key]) || stats[key] < 0)) {
    throw new Error('Dashboard response contained invalid statistics.');
  }

  dashboardCards.total.textContent = String(stats.total_detections);
  dashboardCards.visible.textContent = String(stats.visible_objects);
  dashboardCards.in.textContent = String(stats.total_in);
  dashboardCards.out.textContent = String(stats.total_out);
}

function setDashboardUnavailable(message) {
  Object.values(dashboardCards).forEach((card) => {
    card.textContent = '—';
  });
  liveCounters.visible.textContent = '—';
  const error = document.getElementById('dashboardError');
  error.textContent = message;
  error.hidden = false;
  console.error(message);
}

function activatePanel(targetId) {
  navItems.forEach((button) => {
    const isActive = button.dataset.target === targetId;
    button.classList.toggle('active', isActive);
  });

  panels.forEach((panel) => {
    panel.classList.toggle('active', panel.id === targetId);
  });
}

navItems.forEach((button) => {
  button.addEventListener('click', () => {
    activatePanel(button.dataset.target);
    if (button.dataset.target === 'analytics') {
      fetchAnalytics();
    }
  });
});

function buildCharts() {
  Object.values(charts).forEach((chart) => chart.destroy());
  charts = {};

  const activityCtx = document.getElementById('activityChart');
  const inOutCtx = document.getElementById('inOutChart');
  const distributionCtx = document.getElementById('distributionChart');
  const trafficCtx = document.getElementById('trafficChart');

  charts.activity = new Chart(activityCtx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Detections',
        data: [],
        borderColor: '#4f6cf5',
        backgroundColor: 'rgba(79,108,245,0.16)',
        tension: 0.3,
        fill: true,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { beginAtZero: true } },
    },
  });

  charts.inOut = new Chart(inOutCtx, {
    type: 'bar',
    data: {
      labels: ['IN', 'OUT'],
      datasets: [{
        label: 'Crossings',
        data: [],
        backgroundColor: ['#20b57d', '#eb5b5b'],
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { beginAtZero: true } },
    },
  });

  charts.distribution = new Chart(distributionCtx, {
    type: 'doughnut',
    data: {
      labels: [],
      datasets: [{
        data: [],
        backgroundColor: ['#4f6cf5', '#7aa2ff', '#4ac5a5', '#eb5b5b', '#f4b63f', '#9a7ff8'],
      }],
    },
    options: { responsive: true, maintainAspectRatio: false },
  });

  charts.traffic = new Chart(trafficCtx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'IN', data: [], borderColor: '#20b57d', tension: 0.2, fill: false },
        { label: 'OUT', data: [], borderColor: '#eb5b5b', tension: 0.2, fill: false },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { beginAtZero: true } },
    },
  });
}

function updateDashboard(stats, system) {
  updateDashboardCards(stats);
  document.getElementById('dashboardError').hidden = true;

  liveCounters.in.textContent = String(stats.total_in || 0);
  liveCounters.out.textContent = String(stats.total_out || 0);
  liveCounters.visible.textContent = String(stats.visible_objects);

  document.getElementById('systemModelStatus').textContent = system.model_ready ? 'Ready' : 'Error';
  document.getElementById('systemDevice').textContent = system.device || '—';
  document.getElementById('systemGpu').textContent = system.gpu_name || '—';
  document.getElementById('systemStatus').textContent = system.message || '—';
}

function renderHistory(entries) {
  const tbody = document.getElementById('historyTable');
  tbody.innerHTML = '';
  entries.slice(-12).reverse().forEach((entry) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${entry.timestamp || 'n/a'}</td>
      <td>${entry.object_name || 'Unknown'}</td>
      <td>${entry.event_type || 'detection'}</td>
      <td>${Number(entry.confidence || 0).toFixed(2)}</td>
      <td>${entry.tracking_id ?? '—'}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderRecentEvents(entries) {
  const list = document.getElementById('recentEventsList');
  list.innerHTML = '';
  const items = entries.slice(0, 6);
  items.forEach((entry) => {
    const li = document.createElement('li');
    const details = document.createElement('span');
    const confidence = entry.confidence === null || entry.confidence === undefined
      ? Number.NaN
      : Number(entry.confidence);
    const confidenceLabel = Number.isFinite(confidence) ? confidence.toFixed(2) : '—';
    const trackingId = entry.tracking_id ?? '—';
    details.className = 'analytics-event-details';
    details.textContent = `${entry.timestamp || 'Time unavailable'} · ${entry.object_name || 'Unknown object'} · Confidence ${confidenceLabel} · ID ${trackingId}`;
    const eventType = document.createElement('strong');
    eventType.textContent = entry.event_type || 'Unknown event';
    li.append(details, eventType);
    list.appendChild(li);
  });
}

const chartElements = {
  activity: { canvas: 'activityChart', state: 'activityChartState' },
  inOut: { canvas: 'inOutChart', state: 'inOutChartState' },
  distribution: { canvas: 'distributionChart', state: 'distributionChartState' },
  traffic: { canvas: 'trafficChart', state: 'trafficChartState' },
};

function setChartState(chartName, state, message) {
  const elements = chartElements[chartName];
  const canvas = document.getElementById(elements.canvas);
  const status = document.getElementById(elements.state);
  canvas.hidden = state !== 'ready';
  status.hidden = state === 'ready';
  status.dataset.state = state;
  status.textContent = message;
  if (state === 'ready') {
    charts[chartName].resize();
  }
}

function setAnalyticsLoading() {
  Object.keys(chartElements).forEach((chartName) => {
    setChartState(chartName, 'loading', 'Loading analytics data…');
  });
  const recentStatus = document.getElementById('recentEventsState');
  recentStatus.hidden = false;
  recentStatus.dataset.state = 'loading';
  recentStatus.textContent = 'Loading recent events…';
  document.getElementById('recentEventsList').replaceChildren();
}

function updateAnalyticsCharts(data) {
  charts.activity.data.labels = data.detection_activity.map((item) => item.timestamp);
  charts.activity.data.datasets[0].data = data.detection_activity.map((item) => item.count);
  charts.activity.update();
  setChartState(
    'activity',
    data.detection_activity.length ? 'ready' : 'empty',
    data.detection_activity.length ? '' : 'No detection records yet.',
  );

  const { total_in: totalIn, total_out: totalOut } = data.entry_exit;
  charts.inOut.data.datasets[0].data = [totalIn, totalOut];
  charts.inOut.update();
  const hasCrossings = totalIn > 0 || totalOut > 0;
  setChartState('inOut', hasCrossings ? 'ready' : 'empty', hasCrossings ? '' : 'No crossing events yet.');

  charts.distribution.data.labels = data.object_distribution.map((item) => item.object_name);
  charts.distribution.data.datasets[0].data = data.object_distribution.map((item) => item.count);
  charts.distribution.update();
  setChartState(
    'distribution',
    data.object_distribution.length ? 'ready' : 'empty',
    data.object_distribution.length ? '' : 'No detection records yet.',
  );

  charts.traffic.data.labels = data.traffic_flow.map((item) => item.timestamp);
  charts.traffic.data.datasets[0].data = data.traffic_flow.map((item) => item.total_in);
  charts.traffic.data.datasets[1].data = data.traffic_flow.map((item) => item.total_out);
  charts.traffic.update();
  setChartState(
    'traffic',
    data.traffic_flow.length ? 'ready' : 'empty',
    data.traffic_flow.length ? '' : 'No crossing events yet.',
  );

  renderRecentEvents(data.recent_events);
  const recentStatus = document.getElementById('recentEventsState');
  recentStatus.hidden = data.recent_events.length > 0;
  recentStatus.dataset.state = 'empty';
  recentStatus.textContent = 'No recent events.';
}

function setAnalyticsUnavailable(message) {
  Object.keys(chartElements).forEach((chartName) => {
    setChartState(chartName, 'error', message);
  });
  const recentStatus = document.getElementById('recentEventsState');
  recentStatus.hidden = false;
  recentStatus.dataset.state = 'error';
  recentStatus.textContent = message;
  document.getElementById('recentEventsList').replaceChildren();
}

async function fetchAnalytics() {
  setAnalyticsLoading();
  try {
    const response = await fetch('/api/analytics');
    if (!response.ok) {
      throw new Error(`Analytics API returned HTTP ${response.status}.`);
    }
    const payload = await response.json();
    if (!payload.entry_exit
      || !Number.isSafeInteger(payload.entry_exit.total_in)
      || !Number.isSafeInteger(payload.entry_exit.total_out)
      || !Array.isArray(payload.detection_activity)
      || !Array.isArray(payload.traffic_flow)
      || !Array.isArray(payload.object_distribution)
      || !Array.isArray(payload.recent_events)) {
      throw new Error('Analytics API returned an invalid response.');
    }
    updateAnalyticsCharts(payload);
  } catch (error) {
    setAnalyticsUnavailable(`Analytics data is unavailable. ${error.message || 'Check the server and try again.'}`);
    console.error(error);
  }
}

function renderObjectList(detections) {
  const list = document.getElementById('objectList');
  list.innerHTML = '';
  if (!detections || detections.length === 0) {
    const empty = document.createElement('li');
    empty.textContent = 'No objects in frame';
    list.appendChild(empty);
    return;
  }

  detections.forEach((item) => {
    const li = document.createElement('li');
    const identity = document.createElement('span');
    identity.textContent = `${item.label || 'Object'} · ID ${item.tracking_id ?? '—'}`;
    const confidence = document.createElement('strong');
    confidence.textContent = `${(Number(item.confidence || 0) * 100).toFixed(1)}%`;
    li.append(identity, confidence);
    list.appendChild(li);
  });
}

async function fetchDashboard() {
  try {
    const response = await fetch('/api/dashboard');
    if (!response.ok) {
      throw new Error(`Dashboard API returned HTTP ${response.status}.`);
    }
    const payload = await response.json();
    const stats = payload.stats || {};
    updateDashboard(stats, payload.system || {});
  } catch (error) {
    setDashboardUnavailable(`Dashboard data is unavailable. ${error.message || 'Check the server and try again.'}`);
  }
}

async function fetchHistory() {
  const response = await fetch('/api/history');
  const payload = await response.json();
  renderHistory(payload || []);
}

async function fetchSettings() {
  try {
    const response = await fetch('/api/settings');
    if (!response.ok) {
      throw new Error(`Settings API returned HTTP ${response.status}.`);
    }
    const payload = await response.json();
    if (!Array.isArray(payload.object_prompts) || payload.object_prompts.length === 0) {
      throw new Error('Settings API returned an invalid object prompt list.');
    }
    document.getElementById('confidence').value = payload.confidence;
    document.getElementById('iou').value = payload.iou;
    document.getElementById('lineY').value = payload.line_y_ratio;
    document.getElementById('prompts').value = payload.object_prompts.join(', ');
  } catch (error) {
    const status = document.getElementById('settingsStatus');
    status.hidden = false;
    status.dataset.state = 'error';
    status.textContent = `Saved settings could not be loaded. ${error.message || 'Check the server and try again.'}`;
    console.error(error);
  }
}

function connectWebSocket() {
  const isLocalDevelopment = window.location.hostname === 'localhost' && window.location.port === '8002';
  const socketUrl = isLocalDevelopment
    ? 'ws://localhost:8002/ws'
    : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`;
  ws = new WebSocket(socketUrl);

  return new Promise((resolve, reject) => {
    let settled = false;
    connectionTimeout = setTimeout(() => {
      if (!settled) {
        settled = true;
        reject(new Error('Timed out connecting to the detection backend.'));
      }
    }, 10000);

    ws.onopen = () => {
      ws.send(JSON.stringify({ type: 'init', config: getSettingsFromForm() }));
    };

    ws.onmessage = (event) => {
      let payload;
      try {
        payload = JSON.parse(event.data);
      } catch {
        window.DetectionFpsIndicator.reset(fpsIndicator);
        setCameraStatus('The backend returned an invalid response.', 'error');
        return;
      }

      if (payload.type === 'ready') {
        socketReady = true;
        clearTimeout(connectionTimeout);
        setCameraStatus('Connected to detection backend.', 'connected');
        if (!settled) {
          settled = true;
          resolve();
        }
        return;
      }

      if (payload.type === 'error') {
        window.DetectionFpsIndicator.reset(fpsIndicator);
        framePending = false;
        consecutiveFrameErrors += 1;
        setCameraStatus(`Detection backend error: ${payload.message || 'frame processing failed.'}`, 'error');
        if (payload.stats) {
          updateLiveStats(payload.stats, payload.system || {});
        }
        if (consecutiveFrameErrors < 3 && streamActive) {
          retryHandle = setTimeout(captureAndSendFrame, 1500);
        } else if (streamActive) {
          setCameraStatus('Detection paused after repeated backend errors. Raw camera preview is still active.', 'error');
        }
        if (!settled) {
          settled = true;
          clearTimeout(connectionTimeout);
          reject(new Error(payload.message || 'The detection backend could not process a frame.'));
        }
        return;
      }

      if (payload.type === 'result') {
        framePending = false;
        consecutiveFrameErrors = 0;
        window.DetectionFpsIndicator.update(fpsIndicator, payload.inference_ms);
        displayAnnotatedFrame(payload.image);
        updateLiveStats(payload.stats || {}, payload.system || {});
        renderObjectList(payload.detections || []);
        const timing = Number(payload.inference_ms);
        setCameraStatus(
          `Backend received frame ${payload.frame_id ?? frameSequence}${Number.isFinite(timing) ? ` · inference ${timing.toFixed(0)} ms` : ''}`,
          'connected',
        );
        if (streamActive) {
          retryHandle = setTimeout(captureAndSendFrame, 150);
        }
      }
    };

    ws.onerror = () => {
      window.DetectionFpsIndicator.reset(fpsIndicator);
      setCameraStatus('Could not connect to the detection backend. Check that the server is running.', 'error');
      if (!settled) {
        settled = true;
        clearTimeout(connectionTimeout);
        reject(new Error('WebSocket connection to the detection backend failed.'));
      }
    };

    ws.onclose = (event) => {
      window.DetectionFpsIndicator.reset(fpsIndicator);
      socketReady = false;
      framePending = false;
      if (!settled) {
        settled = true;
        clearTimeout(connectionTimeout);
        reject(new Error(`Detection backend closed the connection (${event.code}).`));
      } else if (streamActive) {
        stopCamera('Detection backend connection closed.');
      }
    };
  });
}

function setCameraStatus(message, state = 'idle') {
  const status = document.getElementById('cameraStatus');
  status.textContent = message;
  status.dataset.state = state;
}

function updateLiveStats(stats, system) {
  updateDashboardCards(stats);
  document.getElementById('dashboardError').hidden = true;
  liveCounters.in.textContent = String(stats.total_in);
  liveCounters.out.textContent = String(stats.total_out);
  liveCounters.visible.textContent = String(stats.visible_objects);
  if (system.model_ready !== undefined) {
    document.getElementById('systemModelStatus').textContent = system.model_ready ? 'Ready' : 'Error';
    document.getElementById('systemDevice').textContent = system.device || '—';
    document.getElementById('systemGpu').textContent = system.gpu_name || '—';
    document.getElementById('systemStatus').textContent = system.message || '—';
  }
}

function displayAnnotatedFrame(imageData) {
  if (typeof imageData !== 'string' || imageData.length === 0) {
    setCameraStatus('The backend returned no annotated image.', 'error');
    return;
  }
  const image = document.getElementById('cameraFrame');
  image.onerror = () => {
    image.hidden = true;
    image.removeAttribute('src');
    document.getElementById('cameraVideo').hidden = false;
    setCameraStatus('The backend image could not be decoded; showing the raw camera feed.', 'error');
  };
  image.src = `data:image/jpeg;base64,${imageData}`;
  image.hidden = false;
  document.getElementById('cameraVideo').hidden = true;
  document.getElementById('cameraMessage').hidden = true;
}

function getSettingsFromForm() {
  const promptsField = document.getElementById('prompts').value;
  return {
    confidence: Number(document.getElementById('confidence').value || 0.25),
    iou: Number(document.getElementById('iou').value || 0.5),
    line_y_ratio: Number(document.getElementById('lineY').value || 0.72),
    object_prompts: promptsField.split(',').map((item) => item.trim()).filter(Boolean),
  };
}

async function saveSettings(event) {
  event.preventDefault();
  const payload = getSettingsFromForm();
  const status = document.getElementById('settingsStatus');
  status.hidden = false;
  status.dataset.state = 'loading';
  status.textContent = 'Saving settings…';
  try {
    const response = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.detail || `Settings API returned HTTP ${response.status}.`);
    }
    status.dataset.state = 'success';
    status.textContent = 'Settings saved and applied.';
    await fetchDashboard();
    await fetchHistory();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'init', config: payload }));
    }
  } catch (error) {
    status.dataset.state = 'error';
    status.textContent = `Settings could not be saved. ${error.message || 'Check the server and try again.'}`;
    console.error(error);
  }
}

async function startCamera() {
  if (streamActive || cameraStarting) {
    return;
  }
  window.DetectionFpsIndicator.reset(fpsIndicator);
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setCameraStatus('Webcam access is unavailable. Use a supported browser on localhost or HTTPS.', 'error');
    return;
  }

  cameraStarting = true;
  document.getElementById('startCamera').disabled = true;
  setCameraStatus('Requesting camera access…');

  try {
    videoStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
    videoStream.getVideoTracks().forEach((track) => {
      track.addEventListener('ended', () => stopCamera('Camera disconnected.'), { once: true });
    });
    const video = document.getElementById('cameraVideo');
    video.srcObject = videoStream;
    video.hidden = false;
    document.getElementById('cameraFrame').hidden = true;
    document.getElementById('cameraFrame').removeAttribute('src');
    document.getElementById('cameraMessage').hidden = true;
    await video.play();

    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      await new Promise((resolve, reject) => {
        video.addEventListener('loadeddata', resolve, { once: true });
        video.addEventListener('error', () => reject(new Error('The browser could not read frames from the webcam.')), { once: true });
      });
    }

    cameraCanvas = document.createElement('canvas');
    cameraContext = cameraCanvas.getContext('2d');
    if (!cameraContext) {
      throw new Error('The browser could not create a canvas for camera frames.');
    }

    streamActive = true;
    framePending = false;
    socketReady = false;
    frameSequence = 0;
    consecutiveFrameErrors = 0;
    setCameraStatus('Camera active. Connecting to detection backend…');
    await connectWebSocket();
    captureAndSendFrame();
  } catch (error) {
    const message = error.name === 'NotAllowedError'
      ? 'Camera permission was denied. Allow camera access in the browser and try again.'
      : error.name === 'NotFoundError'
        ? 'No webcam was found on this device.'
        : error.message || 'Could not start the camera.';
    stopCamera(message);
    setCameraStatus(message, 'error');
  } finally {
    cameraStarting = false;
    document.getElementById('startCamera').disabled = false;
  }
}

function captureAndSendFrame() {
  if (!streamActive || !socketReady || !ws || ws.readyState !== WebSocket.OPEN || framePending) {
    return;
  }
  const video = document.getElementById('cameraVideo');
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !video.videoWidth || !video.videoHeight) {
    retryHandle = setTimeout(captureAndSendFrame, 250);
    return;
  }

  const scale = Math.min(640 / video.videoWidth, 480 / video.videoHeight, 1);
  cameraCanvas.width = Math.max(1, Math.round(video.videoWidth * scale));
  cameraCanvas.height = Math.max(1, Math.round(video.videoHeight * scale));
  cameraContext.drawImage(video, 0, 0, cameraCanvas.width, cameraCanvas.height);

  try {
    frameSequence += 1;
    ws.send(JSON.stringify({
      type: 'frame',
      frame_id: frameSequence,
      image: cameraCanvas.toDataURL('image/jpeg', 0.72),
      config: getSettingsFromForm(),
    }));
    framePending = true;
    setCameraStatus(`Camera active · sending frame ${frameSequence}…`);
  } catch (error) {
    setCameraStatus(`Could not send camera frame: ${error.message}`, 'error');
  }
}

function stopCamera(message = 'Camera stopped') {
  window.DetectionFpsIndicator.reset(fpsIndicator);
  streamActive = false;
  socketReady = false;
  framePending = false;
  clearTimeout(retryHandle);
  clearTimeout(connectionTimeout);
  if (videoStream) {
    videoStream.getTracks().forEach((track) => track.stop());
    videoStream = null;
  }
  const video = document.getElementById('cameraVideo');
  if (video) {
    video.pause();
    video.srcObject = null;
    video.hidden = true;
  }
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    ws.close();
  }
  ws = null;
  const frame = document.getElementById('cameraFrame');
  if (frame) {
    frame.removeAttribute('src');
    frame.hidden = true;
  }
  document.getElementById('cameraMessage').hidden = false;
  liveCounters.visible.textContent = '0';
  dashboardCards.visible.textContent = '0';
  renderObjectList([]);
  setCameraStatus(message, message === 'Camera stopped' ? 'idle' : 'error');
}

document.getElementById('refreshMetrics').addEventListener('click', async () => {
  await fetchDashboard();
  await fetchHistory();
  await fetchAnalytics();
});

document.getElementById('refreshAnalytics').addEventListener('click', fetchAnalytics);
document.getElementById('startCamera').addEventListener('click', startCamera);
document.getElementById('stopCamera').addEventListener('click', () => stopCamera());
document.getElementById('settingsForm').addEventListener('submit', saveSettings);

buildCharts();
fetchSettings();
fetchDashboard();
fetchHistory();
fetchAnalytics();
