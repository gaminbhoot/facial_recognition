const API_BASE = ''; // Same origin
const logsContainer = document.getElementById('logs-container');
const logCountEl = document.getElementById('log-count');
const fpsValueEl = document.getElementById('fps-value');
const privacyStatusEl = document.getElementById('privacy-status');

// Nav Tabs
const navItems = document.querySelectorAll('#sidebar-nav .nav-item');
const tabContents = document.querySelectorAll('.tab-content');

// Settings Elements
const matchThresholdInput = document.getElementById('matching-threshold');
const matchThresholdVal = document.getElementById('matching-value');
const livenessThresholdInput = document.getElementById('liveness-threshold');
const livenessThresholdVal = document.getElementById('liveness-value');
const saveSettingsBtn = document.getElementById('save-settings');
const settingsAlert = document.getElementById('settings-alert');
const privacyCheckbox = document.getElementById('toggle-privacy-checkbox');

// Telemetry Elements
const latDetEl = document.getElementById('lat-det');
const latExtEl = document.getElementById('lat-ext');
const latClsEl = document.getElementById('lat-cls');
const latTotEl = document.getElementById('lat-tot');

// Enrollment Elements
const enrollmentForm = document.getElementById('enrollment-form');
const enrollNameInput = document.getElementById('enroll-name');
const btnEnroll = document.getElementById('btn-enroll');
const enrollmentAlert = document.getElementById('enrollment-alert');

// Users Database Elements
const usersTableBody = document.getElementById('users-table-body');

let lastSeenNames = new Set();
let lastLoggedTimes = {};
let logCount = 0;
let statusIntervalId = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupTabs();
    loadSettings();
    loadUsers();
    
    // Poll status every 500ms
    statusIntervalId = setInterval(updateStatus, 500);
});

// 1. Tab Switching Setup
function setupTabs() {
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetTabId = item.getAttribute('data-tab');
            
            // Toggle active sidebar items
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
            
            // Toggle active tab contents
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === targetTabId) {
                    content.classList.add('active');
                }
            });
            
            // If switched to database tab, reload users
            if (targetTabId === 'database-tab') {
                loadUsers();
            }
        });
    });
}

// 2. Poll Status & Update Telemetry
async function updateStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const data = await response.json();

        // Update basic stats
        fpsValueEl.textContent = data.fps.toFixed(1);
        privacyStatusEl.textContent = data.privacy_mode ? 'ON' : 'OFF';
        privacyStatusEl.style.color = data.privacy_mode ? 'var(--success)' : 'var(--text-secondary)';
        privacyCheckbox.checked = data.privacy_mode;

        // Update Detailed Latency Telemetry
        if (data.latency) {
            const getMS = (stage) => data.latency[stage] ? `${data.latency[stage].avg.toFixed(1)} ms` : '-- ms';
            latDetEl.textContent = getMS('detection');
            latExtEl.textContent = getMS('extraction');
            latClsEl.textContent = getMS('classification');
            latTotEl.textContent = getMS('total_frame');
        }

        // Update Bounding Box Results & Logs
        if (data.last_seen && data.last_seen.length > 0) {
            // Remove placeholder if it exists
            const placeholder = document.querySelector('.log-placeholder');
            if (placeholder) placeholder.remove();

            const now = Date.now();
            data.last_seen.forEach(face => {
                const name = face.name;
                const lastLogged = lastLoggedTimes[name] || 0;
                
                // Only log new detection sessions to avoid clutter, or log all Unknown/Spoofs with a 5s rate-limit
                if (!lastSeenNames.has(name) || ((name === 'Unknown' || name === 'Spoof Attack') && now - lastLogged > 5000)) {
                    addLog(name, face.conf);
                    lastSeenNames.add(name);
                    lastLoggedTimes[name] = now;
                }
            });
        } else {
            // Clear memory when nobody is in frame
            lastSeenNames.clear();
        }
    } catch (err) {
        console.error('Failed to fetch status:', err);
    }
}

// Render log item in sidebar list
function addLog(name, confidence) {
    const logItem = document.createElement('div');
    logItem.className = 'log-item';
    
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    
    let labelClass = 'unknown';
    if (name === 'Spoof Attack') {
        labelClass = 'spoofed';
    } else if (name !== 'Unknown') {
        labelClass = 'live';
    }
    
    let confDisplay = `${(confidence * 100).toFixed(0)}%`;
    if (name === 'Spoof Attack') {
        confDisplay = `Score: ${confidence.toFixed(1)}`;
    }
    
    logItem.innerHTML = `
        <span class="log-name ${labelClass}">${name}</span>
        <span class="log-time">${time} | ${confDisplay}</span>
    `;
    
    logsContainer.prepend(logItem);
    logCount++;
    logCountEl.textContent = `${logCount} Detected`;

    // Keep list length capped at 50
    if (logsContainer.children.length > 50) {
        logsContainer.lastElementChild.remove();
    }
}

// 3. User Enrollment Form Handler
enrollmentForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const name = enrollNameInput.value.trim();
    if (!name) return;
    
    showAlert(enrollmentAlert, 'info', 'Preparing biometric capture... Please face the camera.');
    btnEnroll.disabled = true;
    
    try {
        const response = await fetch(`${API_BASE}/register_user`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        const data = await response.json();
        
        if (data.status === 'success') {
            showAlert(enrollmentAlert, 'success', data.message);
            enrollNameInput.value = '';
            loadUsers(); // Reload table
        } else {
            showAlert(enrollmentAlert, 'error', data.message);
        }
    } catch (err) {
        showAlert(enrollmentAlert, 'error', 'Network error. Capture failed.');
        console.error(err);
    } finally {
        btnEnroll.disabled = false;
    }
});

// 4. Manage Enrolled Users List
async function loadUsers() {
    try {
        const response = await fetch(`${API_BASE}/users`);
        const users = await response.json();
        
        usersTableBody.innerHTML = '';
        if (users.length === 0) {
            usersTableBody.innerHTML = `
                <tr>
                    <td colspan="4" class="table-placeholder">No enrolled identities found in database.</td>
                </tr>
            `;
            return;
        }
        
        users.forEach(user => {
            const tr = document.createElement('tr');
            
            // Format Timestamp
            const enrollDate = user.last_seen ? new Date(user.last_seen).toLocaleDateString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : 'N/A';
            
            tr.innerHTML = `
                <td><strong>${user.name}</strong></td>
                <td><span class="setting-val">${user.embedding_count} face</span></td>
                <td style="color: var(--text-secondary);">${enrollDate}</td>
                <td>
                    <button class="btn-delete" onclick="deleteUser('${user.name}')">Remove</button>
                </td>
            `;
            usersTableBody.appendChild(tr);
        });
    } catch (err) {
        console.error('Failed to load users:', err);
    }
}

window.deleteUser = async function(name) {
    if (!confirm(`Are you sure you want to permanently delete '${name}' from the biometric database?`)) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/delete_user`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        const data = await response.json();
        if (data.status === 'success') {
            loadUsers();
        } else {
            alert(data.message);
        }
    } catch (err) {
        console.error(err);
        alert('Failed to delete user due to a network error.');
    }
};

// 5. Settings Configuration Management
async function loadSettings() {
    try {
        const response = await fetch(`${API_BASE}/settings`);
        const settings = await response.json();
        
        matchThresholdInput.value = settings.recognition_threshold;
        matchThresholdVal.textContent = settings.recognition_threshold.toFixed(2);
        
        livenessThresholdInput.value = settings.liveness_threshold;
        livenessThresholdVal.textContent = settings.liveness_threshold.toFixed(1);
    } catch (err) {
        console.error('Failed to load settings:', err);
    }
}

// Live range values updates
matchThresholdInput.addEventListener('input', (e) => {
    matchThresholdVal.textContent = parseFloat(e.target.value).toFixed(2);
});

livenessThresholdInput.addEventListener('input', (e) => {
    livenessThresholdVal.textContent = parseFloat(e.target.value).toFixed(1);
});

// Save settings to backend
saveSettingsBtn.addEventListener('click', async () => {
    const recognition_threshold = parseFloat(matchThresholdInput.value);
    const liveness_threshold = parseFloat(livenessThresholdInput.value);
    
    try {
        const response = await fetch(`${API_BASE}/update_settings`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ recognition_threshold, liveness_threshold })
        });
        const data = await response.json();
        
        if (data.status === 'success') {
            showAlert(settingsAlert, 'success', 'System thresholds updated successfully.');
        } else {
            showAlert(settingsAlert, 'error', 'Failed to update configuration.');
        }
    } catch (err) {
        showAlert(settingsAlert, 'error', 'Network error. Save failed.');
    }
});

// Privacy Mode Sync
privacyCheckbox.addEventListener('change', async () => {
    try {
        const response = await fetch(`${API_BASE}/toggle_privacy`, { method: 'POST' });
        const data = await response.json();
        privacyStatusEl.textContent = data.privacy_mode ? 'ON' : 'OFF';
        privacyStatusEl.style.color = data.privacy_mode ? 'var(--success)' : 'var(--text-secondary)';
    } catch (err) {
        console.error('Failed to toggle privacy:', err);
    }
});

// Shutdown Server Click Handler
const shutdownBtn = document.getElementById('btn-shutdown');
if (shutdownBtn) {
    shutdownBtn.addEventListener('click', async () => {
        if (confirm("Are you sure you want to shut down the face biometric server? The dashboard will disconnect.")) {
            try {
                const response = await fetch(`${API_BASE}/shutdown`, { method: 'POST' });
                const data = await response.json();
                alert(data.message);
                
                // Disconnect UI and show offline screen
                clearInterval(statusIntervalId);
                document.body.innerHTML = `
                    <div style="height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; font-family: 'Inter', sans-serif; background-color: #0b0d11; color: #fff; text-align: center; padding: 20px;">
                        <h1 style="font-size: 2.5rem; margin-bottom: 12px; color: var(--danger, #ff3b30);">System Offline</h1>
                        <p style="color: #8a8d93; font-size: 1.1rem; max-width: 500px; line-height: 1.6;">The biometric server has shut down successfully. You can close this browser tab safely.</p>
                    </div>
                `;
            } catch (err) {
                alert("Failed to send shutdown command. Server might already be offline.");
            }
        }
    });
}

// Helper for UI alerts
function showAlert(el, type, message) {
    if (el.timeoutId) {
        clearTimeout(el.timeoutId);
    }
    el.className = `alert ${type}`;
    el.textContent = message;
    
    // Auto fade alert out after 5s
    el.timeoutId = setTimeout(() => {
        el.className = 'alert hidden';
        delete el.timeoutId;
    }, 5000);
}
