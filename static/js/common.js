// ==================== Shared Utilities ====================

let currentModel = 'deepseek-chat';
let ganEnabled = false;

const LOCAL_MODELS = new Set(['v2a', 'v3a', 'v3a_baseline', 'v3b', 'v3c']);

function isLocalModel() {
    return LOCAL_MODELS.has(currentModel);
}

function escapeHtml(str) {
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
}

// ---- Toast ----
function showToast(msg, type) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = 'toast toast-' + type;
    toast.textContent = msg;
    toast.onclick = () => toast.remove();
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ---- API Helper ----
async function apiCall(endpoint, body, method) {
    method = method || 'POST';
    const opts = { method: method };
    if (body) {
        if (body instanceof FormData) {
            opts.body = body;
        } else {
            opts.body = body;
        }
    }
    const resp = await fetch(endpoint, opts);
    return resp.json();
}

// ---- Tab Navigation ----
let currentTab = 'single';

function setTab(name) {
    currentTab = name;
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    const tabEl = document.getElementById('tab-' + name);
    if (tabEl) tabEl.classList.add('active');
    const paneEl = document.getElementById('pane-' + name);
    if (paneEl) {
        paneEl.classList.add('active');
        if (name === 'stats') loadStats();
        if (name === 'compare') loadCompareHistory();
    }
}

// ---- Date Formatting ----
function formatDate(iso) {
    if (!iso) return '-';
    const d = new Date(iso);
    return d.toLocaleString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

// ---- Model Select ----
function initModelSelect() {
    const sel = document.getElementById('model-select');
    if (!sel) return;
    sel.onchange = function () {
        currentModel = this.value;
        const defenseGroup = document.getElementById('defense-mode-group');
        const defenseInfo = document.getElementById('defense-info');
        if (isLocalModel()) {
            if (defenseGroup) {
                defenseGroup.style.opacity = '0.4';
                defenseGroup.style.pointerEvents = 'none';
            }
            if (defenseInfo) {
                if (this.value === 'v3a_baseline') {
                    defenseInfo.innerHTML = '<span class="badge badge-danger">无防御</span> 基线版·无prompt防护';
                } else {
                    defenseInfo.innerHTML = '<span class="badge badge-success">StruQ</span> 本地模型内置三层防御';
                }
            }
        } else {
            if (defenseGroup) {
                defenseGroup.style.opacity = '1';
                defenseGroup.style.pointerEvents = 'auto';
            }
            if (typeof updateDefenseInfo === 'function') updateDefenseInfo();
        }
        // Update batch model selector if visible
        const bms = document.getElementById('batch-model-select');
        if (bms) bms.value = this.value;
    };
}

// ---- Init ----
document.addEventListener('DOMContentLoaded', function () {
    initModelSelect();
    // Preload data for tabs so they're ready when user switches
    if (typeof loadCompareHistory === 'function') loadCompareHistory();
    // Health check
    (async function () {
        try {
            const resp = await fetch('/api/health');
            const data = await resp.json();
            document.getElementById('status-text').textContent = 'API 连接正常';
            document.getElementById('status-dot').style.background = 'var(--success)';
            const localDot = document.getElementById('local-status-dot');
            if (data.local_model_available) {
                localDot.style.display = 'inline-block';
                localDot.style.background = 'var(--success)';
                localDot.style.boxShadow = '0 0 8px var(--success)';
                localDot.title = '本地 Qwen2.5-7B 模型已就绪';
            } else if (data.local_model_error) {
                localDot.style.display = 'inline-block';
                localDot.style.background = 'var(--warn)';
                localDot.title = '本地模型加载失败: ' + data.local_model_error;
            }
        } catch (e) {
            document.getElementById('status-text').textContent = 'API 离线';
            document.getElementById('status-dot').style.background = 'var(--danger)';
            showToast('后端服务未连接，请启动 review_app.py', 'danger');
        }
    })();
});
