const { createApp, ref, shallowRef, watch, onMounted, onBeforeUnmount } = Vue;

createApp({
    setup() {
        const socket = ref(null);
        const socketConnected = ref(false);
        // const frame = ref(""); // No longer receiving frame from backend for display
        const cameraReady = ref(false);
        const cameraError = ref(null);

        const history = ref([]);
        const stats = ref({
            avg_fatigue: 0,
            total_alerts: 0,
            total_time: 0
        });

        // Tasks State
        const tasks = ref([]);
        const newTaskTitle = ref("");

        // Settings State
        const settings = ref({
            fatigueThreshold: localStorage.getItem('fatigueThreshold') || 90,
            enablePredictiveAlerts: localStorage.getItem('enablePredictiveAlerts') !== 'false',
            alertVolume: localStorage.getItem('alertVolume') || 80,
            alertSound: localStorage.getItem('alertSound') || 'default'
        });

        const saveSettings = () => {
            localStorage.setItem('fatigueThreshold', settings.value.fatigueThreshold);
            localStorage.setItem('enablePredictiveAlerts', settings.value.enablePredictiveAlerts);
            localStorage.setItem('alertVolume', settings.value.alertVolume);
            localStorage.setItem('alertSound', settings.value.alertSound);

            // Notify backend if necessary (e.g. threshold)
            if (socketConnected.value) {
                socket.value.send(JSON.stringify({
                    type: 'update_settings',
                    threshold: settings.value.fatigueThreshold
                }));
            }
            alert("Settings saved successfully!");
        };

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

        const initAudioContext = () => {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (audioContext.state === 'suspended') {
                audioContext.resume();
            }
        };

        const playBeep = (freq = 440, duration = 0.2) => {
            initAudioContext();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();

            oscillator.type = 'sine';
            oscillator.frequency.setValueAtTime(freq, audioContext.currentTime);

            gainNode.gain.setValueAtTime(settings.value.alertVolume / 100, audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + duration);

            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);

            oscillator.start();
            oscillator.stop(audioContext.currentTime + duration);
        };

        // History Data (Removed Dummy)
        const fetchHistory = async () => {
            try {
                const response = await fetch('/history');
                const data = await response.json();
                history.value = data.history;
                stats.value = data.stats;
            } catch (err) {
                console.error("Failed to fetch history:", err);
            }
        };

        const clearHistory = async () => {
            if (!confirm("Are you sure you want to permanently delete all history?")) return;
            try {
                await fetch('/history', { method: 'DELETE' });
                await fetchHistory();
                alert("History cleared.");
            } catch (err) {
                console.error("Failed to clear history:", err);
            }
        };

        // Tasks Logic
        const fetchTasks = async () => {
            try {
                const res = await fetch('/tasks');
                tasks.value = await res.json();
            } catch (err) { console.error("Fetch tasks failed", err); }
        };

        const addTask = async () => {
            if (!newTaskTitle.value.trim()) return;
            try {
                const res = await fetch('/tasks', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title: newTaskTitle.value })
                });
                if (res.ok) {
                    newTaskTitle.value = "";
                    fetchTasks();
                }
            } catch (err) { console.error("Add task failed", err); }
        };

        const toggleTaskStatus = async (task) => {
            const nextStatus = { 'todo': 'in_progress', 'in_progress': 'done', 'done': 'todo' };
            const newStatus = nextStatus[task.status];
            try {
                await fetch(`/tasks/${task.id}`, {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ status: newStatus })
                });
                fetchTasks();
            } catch (err) { console.error("Update task failed", err); }
        };

        const deleteTask = async (task_id) => {
            if (!confirm("Delete this task?")) return;
            try {
                await fetch(`/tasks/${task_id}`, { method: 'DELETE' });
                fetchTasks();
            } catch (err) { console.error("Delete task failed", err); }
        };

        const updateAccentColor = (color) => {
            settings.value.accentColor = color;
            document.documentElement.style.setProperty('--primary', color);
            // Derive glow from color
            const glow = color === '#64D2FF' ? 'rgba(100, 210, 255, 0.4)' :
                color === '#BF5AF2' ? 'rgba(191, 90, 242, 0.4)' : 'rgba(48, 209, 88, 0.4)';
            document.documentElement.style.setProperty('--primary-glow', glow);
        };
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
            initAudioContext();
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
            playBeep(880, 0.3); // High pitch beep for alert
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

        watch(currentView, (newView) => {
            if (newView === 'history') {
                fetchHistory();
            }
            if (newView === 'tasks') {
                fetchTasks();
            }
        });

        onMounted(() => {
            // Apply loaded accent color
            if (settings.value.accentColor) {
                updateAccentColor(settings.value.accentColor);
            }
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
            history,
            stats,
            settings,
            saveSettings,
            clearHistory,
            updateAccentColor,
            tasks,
            newTaskTitle,
            addTask,
            toggleTaskStatus,
            deleteTask,
            getBadgeClass,
            getStatusLabel
        };
    }
}).mount('#app');
