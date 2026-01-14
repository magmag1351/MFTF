const { createApp, ref, shallowRef, onMounted, onBeforeUnmount } = Vue;

createApp({
    setup() {
        const socket = ref(null);
        const socketConnected = ref(false);
        // const frame = ref(""); // No longer receiving frame from backend for display
        const cameraReady = ref(false);
        const cameraError = ref(null);

        const fatigue = ref(0);
        const prediction = ref(0);
        const currentView = ref('monitor'); // monitor, history, settings
        const status = ref('active'); // active, away
        const awayRemaining = ref(0);
        const musicPlaying = ref(false);
        const chartInstance = shallowRef(null);

        let audioContext = null;
        let videoElement = null;
        let captureCanvas = null;
        let captureInterval = null;

        // History Mockup Data
        const dummyHistory = ref([
            { id: 1, timestamp: '2024-01-14 10:30', duration: '1h 20m', peak: 85 },
            { id: 2, timestamp: '2024-01-13 14:15', duration: '2h 15m', peak: 35 },
            { id: 3, timestamp: '2024-01-13 09:00', duration: '45m', peak: 60 },
            { id: 4, timestamp: '2024-01-12 16:45', duration: '1h 10m', peak: 40 },
            { id: 5, timestamp: '2024-01-12 11:30', duration: '3h 05m', peak: 92 }
        ]);

        // Chart Data
        const chartData = {
            labels: [],
            datasets: [
                {
                    label: 'Current Fatigue',
                    borderColor: '#64D2FF', // Cyan
                    backgroundColor: 'rgba(100, 210, 255, 0.1)',
                    borderWidth: 2,
                    data: [],
                    tension: 0.4,
                    pointRadius: 0
                },
                {
                    label: 'Predicted',
                    borderColor: '#BF5AF2', // Purple
                    borderWidth: 2,
                    borderDash: [5, 5],
                    data: [],
                    tension: 0.4,
                    pointRadius: 0
                }
            ]
        };

        const connectWebSocket = () => {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            socket.value = new WebSocket(`${protocol}//${window.location.host}/ws`);

            socket.value.onopen = () => {
                socketConnected.value = true;
                console.log("WebSocket connected");
            };

            socket.value.onclose = () => {
                socketConnected.value = false;
                console.log("WebSocket disconnected. Retrying in 1s...");
                setTimeout(connectWebSocket, 1000);
            };

            socket.value.onmessage = (event) => {
                const data = JSON.parse(event.data);

                // Update State
                // if (data.image_base64) frame.value = data.image_base64; // Don't use backend image
                if (data.fatigue_current !== undefined) fatigue.value = data.fatigue_current;
                if (data.fatigue_pred !== undefined) prediction.value = data.fatigue_pred;
                status.value = data.status;
                awayRemaining.value = data.away_remaining;

                // Audio Alert
                if (data.alert) {
                    playAlert();
                }

                // Update Chart
                if (data.chart_data && chartInstance.value) {
                    updateChart(data.chart_data);
                }
            };
        };

        const updateChart = (points) => {
            if (!chartInstance.value) return;

            const labels = points.map(p => p.time.toFixed(1));
            const fatigueValues = points.map(p => p.value);
            // In the backend, we might want to send historical predictions too. 
            // For now, let's at least show the current prediction at the end or use what's sent.
            const predValues = points.map(p => p.pred !== undefined ? p.pred : null);

            chartInstance.value.data.labels = labels;
            chartInstance.value.data.datasets[0].data = fatigueValues;

            if (points.some(p => p.pred !== undefined)) {
                chartInstance.value.data.datasets[1].data = predValues;
            }

            chartInstance.value.update('none'); // Update without animation for performance
        };

        const initChart = () => {
            const ctx = document.getElementById('fatigueChart').getContext('2d');
            chartInstance.value = new Chart(ctx, {
                type: 'line',
                data: chartData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    scales: {
                        x: {
                            display: true,
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: { color: '#8E8E93', maxTicksLimit: 10 }
                        },
                        y: {
                            min: 0,
                            max: 100,
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: { color: '#8E8E93' }
                        }
                    },
                    plugins: {
                        legend: { display: false }
                    }
                }
            });
        };

        const toggleMusic = () => {
            const audio = document.getElementById('bgMusic');
            if (audio.paused) {
                audio.play().then(() => {
                    musicPlaying.value = true;
                }).catch(e => console.error("Audio play failed", e));
            } else {
                audio.pause();
                musicPlaying.value = false;
            }
        };

        const playAlert = () => {
            const audio = document.getElementById('alertSound');
            audio.play().catch(e => console.error("Alert play failed", e));
        };

        const setAway = () => {
            socket.value.send(JSON.stringify({ type: 'set_away', value: 300 }));
        };

        const resume = () => {
            socket.value.send(JSON.stringify({ type: 'resume' }));
        };

        const formatTime = (seconds) => {
            const m = Math.floor(seconds / 60);
            const s = seconds % 60;
            return `${m}:${s.toString().padStart(2, '0')}`;
        };

        const getFatigueClass = (val) => {
            if (val >= 90) return 'text-critical';
            if (val >= 70) return 'text-warning';
            return 'text-normal';
        };

        const getBadgeClass = (val) => {
            if (val >= 80) return 'badge-high';
            if (val >= 50) return 'badge-medium';
            return 'badge-low';
        };

        const getStatusLabel = (val) => {
            if (val >= 80) return 'High';
            if (val >= 50) return 'Medium';
            return 'Low';
        };

        const startCamera = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        frameRate: { ideal: 30 }
                    },
                    audio: false
                });

                videoElement = document.getElementById('webcamVideo');
                videoElement.srcObject = stream;

                videoElement.onloadedmetadata = () => {
                    cameraReady.value = true;
                    startStreaming();
                };
            } catch (err) {
                console.error("Camera Error:", err);
                cameraError.value = "Access denied or no camera found.";
            }
        };

        const startStreaming = () => {
            if (!captureCanvas) {
                captureCanvas = document.createElement('canvas');
            }

            // Send frame every ~100ms (10 FPS)
            captureInterval = setInterval(() => {
                if (socketConnected.value && status.value === 'active' && videoElement && videoElement.videoWidth > 0) {
                    // Draw video to canvas
                    captureCanvas.width = videoElement.videoWidth;
                    captureCanvas.height = videoElement.videoHeight;
                    const ctx = captureCanvas.getContext('2d');
                    ctx.drawImage(videoElement, 0, 0);

                    // Convert to base64
                    const dataURL = captureCanvas.toDataURL('image/jpeg', 0.6); // Quality 0.6
                    const base64Content = dataURL.split(',')[1];

                    // Send
                    socket.value.send(JSON.stringify({
                        type: 'frame',
                        data: base64Content
                    }));
                }
            }, 100);
        };

        onMounted(() => {
            initChart();
            startCamera();
            connectWebSocket();
        });

        onBeforeUnmount(() => {
            if (captureInterval) clearInterval(captureInterval);
        });

        return {
            socketConnected,
            cameraReady,
            cameraError,
            // frame,
            fatigue,
            prediction,
            currentView,
            status,
            awayRemaining,
            musicPlaying,
            toggleMusic,
            setAway,
            resume,
            formatTime,
            getFatigueClass,
            dummyHistory,
            getBadgeClass,
            getStatusLabel
        };
    }
}).mount('#app');
