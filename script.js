const initBtn = document.getElementById('initBtn');
const stopBtn = document.getElementById('stopBtn');
const statusText = document.getElementById('statusText');

stopBtn.disabled = true;

initBtn.addEventListener('click', handleStart);
stopBtn.addEventListener('click', handleStop);

async function handleStart() {
    initBtn.disabled = true;
    showStatus('Starting...', 'var(--text-light)');

    try {
        const res = await fetch('/api/start', { method: 'POST' });
        const data = await res.json();

        if (!res.ok) throw new Error(data.message || 'Failed to start');

        if (data.status === 'already_running') {
            showStatus('App is already running', 'var(--blue)');
        } else {
            showStatus('App is starting..', 'var(--green)');
        }

        stopBtn.disabled = false;
    } catch (err) {
        showStatus('Failed to start: ' + err.message, 'var(--red)');
        initBtn.disabled = false;
    }
}

async function handleStop() {
    stopBtn.disabled = true;

    try {
        await fetch('/api/stop', { method: 'POST' });
        showStatus('App stopped', 'var(--text-light)');
        initBtn.disabled = false;
    } catch (err) {
        showStatus('Failed to stop', 'var(--red)');
        stopBtn.disabled = false;
    }
}

function showStatus(text, color) {
    statusText.textContent = text;
    statusText.style.color = color;
    statusText.style.display = 'block';
}
