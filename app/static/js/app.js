document.addEventListener('DOMContentLoaded', function () {
    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const removeFileBtn = document.getElementById('removeFile');
    const durationSlider = document.getElementById('durationSlider');
    const durationValue = document.getElementById('durationValue');
    const processBtn = document.getElementById('processBtn');
    const processingInfo = document.getElementById('processingInfo');
    const progressBar = document.getElementById('progressBar');
    const statusMessage = document.getElementById('statusMessage');
    const resultContainer = document.getElementById('resultContainer');
    const outputVideo = document.getElementById('outputVideo');
    const videoSource = document.getElementById('videoSource');
    const downloadBtn = document.getElementById('downloadBtn');
    const newSummaryBtn = document.getElementById('newSummaryBtn');

    // App State
    let selectedFile = null;
    let taskId = null;
    let checkStatusInterval = null;

    // Event Listeners
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);
    removeFileBtn.addEventListener('click', handleRemoveFile);
    durationSlider.addEventListener('input', updateDurationValue);
    processBtn.addEventListener('click', processVideo);
    newSummaryBtn.addEventListener('click', resetUI);

    // Drag & Drop Handlers
    function handleDragOver(e) {
        e.preventDefault();
        dropZone.classList.add('dragover');
    }

    function handleDragLeave(e) {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    }

    function handleDrop(e) {
        e.preventDefault();
        dropZone.classList.remove('dragover');

        if (e.dataTransfer.files.length) {
            handleFiles(e.dataTransfer.files);
        }
    }

    // File Handlers
    function handleFileSelect(e) {
        if (fileInput.files.length) {
            handleFiles(fileInput.files);
        }
    }

    function handleFiles(files) {
        const file = files[0];
        if (file && file.type.startsWith('video/')) {
            selectedFile = file;
            fileName.textContent = file.name;
            fileInfo.classList.remove('d-none');
            processBtn.disabled = false;
        } else {
            alert('Please select a valid video file.');
        }
    }

    function handleRemoveFile() {
        selectedFile = null;
        fileInput.value = '';
        fileInfo.classList.add('d-none');
        processBtn.disabled = true;
    }

    // Duration Slider
    function updateDurationValue() {
        durationValue.textContent = durationSlider.value;
    }

    // Process Video
    async function processVideo() {
        if (!selectedFile) return;

        // Prepare form data
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('target_duration', parseInt(durationSlider.value) * 60); // Convert to seconds

        // Update UI to show processing
        processBtn.disabled = true;
        processingInfo.classList.remove('d-none');
        updateProgress(5, 'Initializing...');

        try {
            // Submit the video for processing
            const response = await fetch('/api/v1/summarize', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}`);
            }

            const data = await response.json();

            if (data.task_id) {
                taskId = data.task_id;
                updateProgress(10, 'Processing started. Extracting audio...');
                startProgressCheck();
            } else {
                throw new Error('No task ID received from server');
            }
        } catch (error) {
            handleError(error);
        }
    }

    // Poll for progress
    function startProgressCheck() {
        if (checkStatusInterval) {
            clearInterval(checkStatusInterval);
        }

        checkStatusInterval = setInterval(checkTaskStatus, 2000); // Check every 2 seconds
    }

    async function checkTaskStatus() {
        if (!taskId) return;

        try {
            const response = await fetch(`/api/v1/task-status/${taskId}`);

            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}`);
            }

            const data = await response.json();

            switch (data.status) {
                case 'PENDING':
                    updateProgress(10, 'Task pending in queue...');
                    break;
                case 'PROCESSING':
                    handleProcessingSteps(data);
                    break;
                case 'COMPLETED':
                    completeTask(data);
                    break;
                case 'FAILED':
                    throw new Error(data.message || 'Task failed');
                default:
                    updateProgress(0, 'Unknown status');
            }
        } catch (error) {
            handleError(error);
        }
    }

    function handleProcessingSteps(data) {
        // Map processing steps to progress percentage
        const stepToProgress = {
            'extracting_audio': 20,
            'transcribing': 40,
            'segmenting': 60,
            'scoring': 80,
            'generating_summary': 90
        };

        const step = data.current_step || 'processing';
        const progress = stepToProgress[step] || 30;
        const message = data.message || `Processing: ${step.replace('_', ' ')}...`;

        updateProgress(progress, message);
    }

    function completeTask(data) {
        clearInterval(checkStatusInterval);
        updateProgress(100, 'Processing complete!');

        setTimeout(() => {
            processingInfo.classList.add('d-none');
            displayResult(data.result_url);
        }, 1000);
    }

    function displayResult(videoUrl) {
        videoSource.src = videoUrl;
        outputVideo.load();
        downloadBtn.href = videoUrl;
        resultContainer.classList.remove('d-none');
    }

    // UI Helpers
    function updateProgress(percent, message) {
        progressBar.style.width = `${percent}%`;
        statusMessage.textContent = message;
    }

    function handleError(error) {
        console.error('Error:', error);
        clearInterval(checkStatusInterval);
        updateProgress(0, `Error: ${error.message}`);

        setTimeout(() => {
            alert(`An error occurred: ${error.message}`);
            resetUI();
        }, 1000);
    }

    function resetUI() {
        // Reset all UI elements to initial state
        handleRemoveFile();
        processingInfo.classList.add('d-none');
        resultContainer.classList.add('d-none');
        updateProgress(0, 'Initializing...');

        if (checkStatusInterval) {
            clearInterval(checkStatusInterval);
        }
    }
});
