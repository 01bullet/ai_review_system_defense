// ==================== Single Review ====================

let paperText = '';
let rawContent = '';
let paperFilename = '';
let currentPaperId = null;
let currentDefenseMode = 'combined';
let hasAttack = false;
let lastScanData = null;
let cleaningApplied = false;

// ---- Toggle GAN ----
function toggleGan(enabled) {
    ganEnabled = enabled;
    const statusEl = document.getElementById('gan-status');
    if (enabled) {
        statusEl.textContent = '已开启（需要下载 distilbert 模型）';
        statusEl.style.color = 'var(--warn)';
    } else {
        statusEl.textContent = '已关闭（模型未下载）';
        statusEl.style.color = 'var(--danger)';
    }
    if (!enabled && currentDefenseMode === 'gan_only') {
        setDefenseMode('rule_only', document.querySelector('#defense-mode-group .btn[data-mode="rule_only"]'));
    }
    updateDefenseInfo();
}

// ---- Stepper ----
function updateStepper(activeStep) {
    const steps = ['upload', 'scan', 'clean', 'review'];
    steps.forEach((s, i) => {
        const el = document.getElementById('step-' + s);
        if (!el) return;
        el.classList.remove('done', 'active');
        if (i < activeStep) el.classList.add('done');
        if (i === activeStep) el.classList.add('active');
    });
}

// ---- Upload ----
const uploadZone = document.getElementById('upload-zone');
if (uploadZone) {
    ['dragover'].forEach(ev => uploadZone.addEventListener(ev, e => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    }));
    ['dragleave', 'drop'].forEach(ev => uploadZone.addEventListener(ev, e => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
    }));
    uploadZone.addEventListener('drop', e => {
        const file = e.dataTransfer.files[0];
        if (file) uploadFile(file);
    });
}

function handleFile(e) {
    const file = e.target.files[0];
    if (file) uploadFile(file);
}

async function uploadFile(file) {
    const form = new FormData();
    form.append('file', file);
    document.getElementById('file-info').innerHTML =
        '<span style="color:var(--text2)">正在上传并扫描...</span>';
    updateStepper(0);

    const uploadHeaders = ganEnabled ? {} : { 'X-GAN-Enabled': 'false' };
    const resp = await fetch('/api/upload', {
        method: 'POST', body: form, headers: uploadHeaders
    });
    const data = await resp.json();

    if (data.error) {
        document.getElementById('file-info').innerHTML =
            '<span style="color:var(--danger)">上传失败: ' + data.error + '</span>';
        showToast('上传失败: ' + data.error, 'danger');
        return;
    }

    paperText = data.text;
    rawContent = data.raw_text || data.text;
    paperFilename = data.filename;
    currentPaperId = data.paper_id || null;
    hasAttack = data.has_attack;
    lastScanData = data;
    cleaningApplied = false;

    document.getElementById('file-info').innerHTML =
        '<span class="file-chip">&#128196; ' + data.filename + ' — ' +
        (data.text_length / 1024).toFixed(1) + ' KB</span>';

    document.getElementById('btn-review').disabled = false;
    document.getElementById('btn-clean').disabled = !hasAttack;

    const preview = document.getElementById('paper-preview');
    preview.textContent = data.text.substring(0, 6000);
    preview.style.display = 'block';
    document.getElementById('preview-card').hidden = false;
    document.getElementById('preview-len').textContent =
        '前 ' + Math.min(6000, data.text_length).toLocaleString() + ' 字符';

    displayScanResults(data);
    updateStepper(1);

    if (hasAttack) {
        showAttackModal(data);
    } else {
        showToast('未检测到攻击，论文安全', 'success');
    }
    updateDefenseInfo();
}

// ---- Scan Display ----
function displayScanResults(data) {
    const card = document.getElementById('scan-card');
    const div = document.getElementById('scan-result');
    card.hidden = false;

    const scan = data.scan;
    const severity = data.severity;
    const severityLabel = { safe: '安全', medium: '中等', high: '高风险', critical: '严重' };
    const severityColor = { safe: 'var(--success)', medium: 'var(--warn)', high: 'var(--danger)', critical: 'var(--danger)' };

    let html = '';
    const scorePct = Math.round(scan.score * 100);
    const scoreColor = scan.score > 0.7 ? 'var(--danger)' :
        scan.score > 0.5 ? 'var(--warn)' : 'var(--success)';
    html += '<div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">';
    html += '<div class="gauge-value" style="color:' + scoreColor + ';min-width:60px;">' + scorePct + '%</div>';
    html += '<div style="flex:1;"><div class="score-bar"><div class="score-bar-fill" style="width:' +
        scorePct + '%;background:' + scoreColor + '"></div></div>';
    html += '<div style="display:flex;justify-content:space-between;font-size:0.7rem;color:var(--text2);margin-top:2px;">' +
        '<span>GAN 检测分数</span><span>阈值: ' + Math.round(scan.threshold * 100) + '%</span></div></div>';
    html += '</div>';

    html += '<div style="margin-bottom:8px;"><span class="badge badge-' +
        (severity === 'safe' ? 'success' : severity === 'medium' ? 'warn' : 'danger') +
        '">风险等级: ' + severityLabel[severity] + '</span>';
    if (scan.flagged) html += ' <span class="badge badge-danger">GAN 标记: 异常</span>';
    html += '</div>';

    if (data.injection_patterns.length > 0) {
        html += '<div style="margin-top:6px;font-size:0.78rem;color:var(--danger);">检测到注入模式：</div>';
        html += '<div style="margin-top:4px;">' +
            data.injection_patterns.map(p =>
                '<span class="badge badge-danger">' + p + '</span>').join(' ') + '</div>';
    }

    if (data.clean_result && data.clean_result.findings.length > 0) {
        html += '<div style="margin-top:8px;font-size:0.78rem;color:var(--warn);">LaTeX 隐藏内容: ' +
            data.clean_result.findings.join(', ') + ' (' + data.clean_result.removed_bytes + ' bytes)</div>';
    }

    if (data.chunk_scan && data.chunk_scan.chunk_scores && data.chunk_scan.chunk_scores.length > 0) {
        html += '<div style="margin-top:10px;font-size:0.75rem;color:var(--text2);">段落风险分布：</div>';
        data.chunk_scan.chunk_scores.slice(0, 8).forEach(c => {
            const cscore = Math.round(c.score * 100);
            const ccol = c.score > 0.7 ? 'var(--danger)' :
                c.score > 0.5 ? 'var(--warn)' : 'var(--success)';
            html += '<div class="chunk-bar"><span style="width:100px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' +
                escapeHtml(c.text_preview) + '</span><div class="bar-bg"><div class="bar-fill" style="width:' +
                cscore + '%;background:' + ccol + '"></div></div><span style="min-width:36px;text-align:right;">' +
                cscore + '%</span></div>';
        });
    }

    div.innerHTML = html;
}

// ---- Attack Modal ----
function showAttackModal(data) {
    const overlay = document.getElementById('attack-modal-overlay');
    const subtitle = document.getElementById('modal-subtitle');
    const details = document.getElementById('modal-details');

    const scan = data.scan;
    const severity = data.severity;
    const severityLabel = { safe: '安全', medium: '中等', high: '高风险', critical: '严重' };

    subtitle.textContent = '系统在论文中发现了针对大语言模型审稿系统的隐蔽攻击注入，风险等级: ' +
        severityLabel[severity];

    let dhtml = '';
    dhtml += '<div class="detail-row"><span class="dl">GAN 检测分数</span><span class="dv" style="color:' +
        (scan.score > 0.5 ? 'var(--danger)' : 'var(--success)') + '">' +
        (scan.score * 100).toFixed(1) + '%</span></div>';
    dhtml += '<div class="detail-row"><span class="dl">防御升级级别</span><span class="dv">' +
        scan.escalation + '</span></div>';
    dhtml += '<div class="detail-row"><span class="dl">检测阈值</span><span class="dv">' +
        (scan.threshold * 100).toFixed(0) + '%</span></div>';

    if (data.injection_patterns.length > 0) {
        dhtml += '<div style="margin-top:12px;font-size:0.85rem;color:var(--danger);">已识别攻击模式：</div>';
        dhtml += '<div style="margin-top:4px;">' +
            data.injection_patterns.map(p =>
                '<span class="badge badge-danger">' + p + '</span>').join(' ') + '</div>';
    }

    if (data.clean_result && data.clean_result.findings.length > 0) {
        dhtml += '<div style="margin-top:12px;font-size:0.85rem;color:var(--warn);">可清除的隐藏内容：</div>';
        dhtml += '<div style="margin-top:4px;">' +
            data.clean_result.findings.map(f =>
                '<span class="badge badge-warn">' + f + '</span>').join(' ') + '</div>';
    }

    details.innerHTML = dhtml;
    overlay.classList.add('show');
    showToast('⚠️ 检测到攻击注入！请选择处理方式', 'danger');
}

function closeAttackModal() {
    document.getElementById('attack-modal-overlay').classList.remove('show');
}

function closeAttackModalAndReview() {
    closeAttackModal();
    runReview();
}

async function cleanPaperFromModal() {
    closeAttackModal();
    await cleanPaper();
}

// ---- Defense Mode ----
function setDefenseMode(mode, btn) {
    currentDefenseMode = mode;
    document.querySelectorAll('#defense-mode-group .btn').forEach(b =>
        b.classList.remove('selected'));
    if (btn) btn.classList.add('selected');
    updateDefenseInfo();
}

function updateDefenseInfo() {
    const div = document.getElementById('defense-info');
    if (!div) return;
    const ganNote = ganEnabled
        ? ' <span style="font-size:0.72rem;color:var(--success);">[GAN ON]</span>'
        : ' <span style="font-size:0.72rem;color:var(--danger);">[GAN OFF]</span>';
    const desc = {
        'combined': '<span class="badge badge-success">综合防护</span> 规则清洗 + StruQ + GAN 对抗检测' + ganNote,
        'rule_only': '<span class="badge badge-info">规则防御</span> LaTeX 清洗 + StruQ',
        'gan_only': '<span class="badge badge-warn">GAN 检测</span> 仅 GAN 检测' +
            (ganEnabled ? '' : ' <span style="color:var(--danger);font-size:0.72rem;">— GAN 已关闭</span>'),
        'no_defense': '<span class="badge badge-danger">无防御</span> 原始流程（易受攻击）',
    };
    div.innerHTML = desc[currentDefenseMode] || '';
}

// ---- Clean ----
async function cleanPaper() {
    if (!rawContent) return;
    const form = new FormData();
    form.append('text', rawContent);
    form.append('filename', paperFilename);

    const btn = document.getElementById('btn-clean');
    btn.disabled = true;
    btn.textContent = '清洗中...';

    const resp = await fetch('/api/clean', { method: 'POST', body: form });
    const data = await resp.json();

    btn.textContent = '清除攻击';
    if (data.error) {
        showToast('清洗失败: ' + data.error, 'danger');
        btn.disabled = false;
        return;
    }

    if (data.cleaned && data.clean_text) {
        paperText = data.clean_text;
        rawContent = data.clean_text;
        cleaningApplied = true;
        hasAttack = false;
        document.getElementById('btn-clean').disabled = true;
        document.getElementById('paper-preview').textContent = paperText.substring(0, 6000);

        const cleanCard = document.getElementById('clean-card');
        cleanCard.hidden = false;
        document.getElementById('clean-result').innerHTML =
            '<div class="alert alert-success">已成功清除攻击内容<br>' +
            '<small>移除了 ' + data.removed_bytes + ' bytes | ' +
            (data.findings.join(', ') || '无具体发现') + '</small></div>' +
            (data.patterns_detected.length > 0 ? '<div style="margin-top:6px;">' +
                data.patterns_detected.map(p =>
                    '<span class="badge badge-warn">' + p + '</span>').join(' ') + '</div>' : '');

        updateStepper(2);
        document.getElementById('scan-card').hidden = true;
        showToast('攻击内容已清除！移除了 ' + data.removed_bytes + ' bytes 隐藏文本', 'success');
    } else {
        btn.disabled = false;
        showToast('未发现需要清除的隐藏内容', 'warn');
    }
}

// ---- Review ----
async function runReview() {
    if (!paperText) return;

    const btn = document.getElementById('btn-review');
    const spinner = document.getElementById('review-spinner');
    btn.disabled = true;
    spinner.style.display = 'inline-block';
    const label = isLocalModel() ? '本地模型推理中...' : 'AI 正在审稿...';
    btn.innerHTML = '<span class="spinner" style="display:inline-block;border-color:transparent;border-top-color:#fff;"></span> ' + label;

    document.getElementById('empty-state').style.display = 'none';
    document.getElementById('review-card').style.display = 'none';

    let endpoint, formData;
    if (isLocalModel()) {
        endpoint = '/api/review-local';
        formData = new FormData();
        formData.append('text', paperText);
        formData.append('temperature', '0.1');
        formData.append('model_version', currentModel);
        if (currentPaperId) formData.append('paper_id', currentPaperId);
    } else {
        endpoint = '/api/review';
        formData = new FormData();
        formData.append('text', paperText);
        formData.append('defense_mode', currentDefenseMode);
        formData.append('auto_clean', cleaningApplied ? 'true' : 'false');
        formData.append('model', currentModel);
        formData.append('gan_enabled', ganEnabled ? 'true' : 'false');
        if (currentPaperId) formData.append('paper_id', currentPaperId);
    }

    try {
        const resp = await fetch(endpoint, { method: 'POST', body: formData });
        const data = await resp.json();

        if (data.error) {
            showToast('审稿失败: ' + data.error, 'danger');
            document.getElementById('empty-state').style.display = '';
            return;
        }

        if (data.source === 'local') {
            displayLocalReview(data);
        } else {
            displayReview(data);
        }
        updateStepper(3);
        showToast('审稿完成！', 'success');

        // Refresh batch/compare tabs if they were waiting for reviews
        if (typeof loadStats === 'function') {
            // Stats may need refresh after new review
        }
    } catch (e) {
        showToast('网络错误: ' + e.message, 'danger');
    } finally {
        btn.disabled = false;
        spinner.style.display = 'none';
        btn.innerHTML = '<span class="spinner" id="review-spinner"></span>开始审稿';
    }
}

// ---- Review Display ----
const CRITERIA = {
    'Originality': { max: 4, desc: '任务或方法的新颖程度', weights: ['low', 'medium', 'high', 'very high'] },
    'Quality': { max: 4, desc: '技术完备性、实验支撑', weights: ['low', 'medium', 'high', 'very high'] },
    'Clarity': { max: 4, desc: '写作清晰度、组织结构', weights: ['low', 'medium', 'high', 'very high'] },
    'Significance': { max: 4, desc: '结果的重要性、影响', weights: ['low', 'medium', 'high', 'very high'] },
    'Soundness': { max: 4, desc: '技术正确性、实验方法论', weights: ['poor', 'fair', 'good', 'excellent'] },
    'Presentation': { max: 4, desc: '写作质量、图表、引用', weights: ['poor', 'fair', 'good', 'excellent'] },
    'Contribution': { max: 4, desc: '整体贡献', weights: ['poor', 'fair', 'good', 'excellent'] },
};

const OVERALL_LABELS = {
    10: 'Award Quality — 技术完美，突破性影响',
    9: 'Very Strong Accept — 技术优秀',
    8: 'Strong Accept — 技术坚实',
    7: 'Accept — 扎实贡献',
    6: 'Weak Accept — 有一定价值',
    5: 'Borderline — 优劣参半',
    4: 'Borderline Reject — 不足较多',
    3: 'Reject — 技术缺陷',
    2: 'Strong Reject — 重大缺陷',
    1: 'Very Strong Reject — 无实质贡献',
};

function displayReview(data) {
    document.getElementById('empty-state').style.display = 'none';
    document.getElementById('review-card').style.display = '';

    const badge = document.getElementById('decision-badge');
    const isAccept = data.decision && /accept/i.test(data.decision);
    badge.textContent = (data.decision || 'Unknown').toUpperCase();
    badge.className = 'decision-badge ' + (isAccept ? 'decision-accept' : 'decision-reject');

    const overall = data.overall || 0;
    document.getElementById('score-overall').textContent = overall + '/10';
    document.getElementById('score-overall-label').textContent = OVERALL_LABELS[overall] || '';
    document.getElementById('score-confidence').textContent = (data.confidence || 0) + '/5';

    const scoreDetails = document.getElementById('score-details');
    let sdHtml = '';
    for (const [name, info] of Object.entries(CRITERIA)) {
        const val = (data.scores || {})[name] || 0;
        const pct = (val / info.max * 100);
        const color = pct >= 75 ? 'var(--success)' : pct >= 50 ? 'var(--accent)' :
            pct >= 30 ? 'var(--warn)' : 'var(--danger)';
        const level = info.weights[Math.max(0, val - 1)] || '-';
        sdHtml += '<div class="score-item">';
        sdHtml += '<div class="score-header"><span class="name">' + name + '</span>' +
            '<span class="criterion">' + info.desc + '</span>' +
            '<span class="val" style="color:' + color + '">' + val + '/' + info.max + ' (' + level + ')</span></div>';
        sdHtml += '<div class="score-bar"><div class="score-bar-fill" style="width:' +
            pct + '%;background:' + color + '"></div></div></div>';
    }
    scoreDetails.innerHTML = sdHtml;
    drawRadarChart(data.scores || {});

    document.getElementById('review-summary').textContent = data.summary || '暂无摘要';

    document.getElementById('strengths-list').innerHTML = (data.strengths || []).length
        ? data.strengths.map(s => '<li class="strength">' + escapeHtml(s) + '</li>').join('')
        : '<li style="color:var(--text2);font-size:0.8rem;">无</li>';

    document.getElementById('weaknesses-list').innerHTML = (data.weaknesses || []).length
        ? data.weaknesses.map(w => '<li class="weakness">' + escapeHtml(w) + '</li>').join('')
        : '<li style="color:var(--text2);font-size:0.8rem;">无</li>';

    document.getElementById('questions-list').innerHTML = (data.questions || []).length
        ? data.questions.map(q => '<li style="font-size:0.8rem;padding:6px 10px;color:var(--text);">' + escapeHtml(q) + '</li>').join('')
        : '<li style="color:var(--text2);font-size:0.8rem;">无</li>';

    document.getElementById('limitations-list').innerHTML = (data.limitations || []).length
        ? data.limitations.map(l => '<li style="font-size:0.8rem;padding:6px 10px;color:var(--text);">' + escapeHtml(l) + '</li>').join('')
        : '<li style="color:var(--text2);font-size:0.8rem;">无</li>';

    const def = data.defense_applied || {};
    document.getElementById('applied-defense').innerHTML =
        '<strong>防护状态:</strong> 模式=' + def.mode + ' | 级别=' + def.defense_level +
        ' | 规则防御=' + (def.rule_defense ? '<span style="color:var(--success)">ON</span>' : '<span style="color:var(--danger)">OFF</span>') +
        ' | GAN防御=' + (def.gan_defense ? '<span style="color:var(--success)">ON</span>' : '<span style="color:var(--danger)">OFF</span>');

    document.getElementById('review-card').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---- Radar Chart ----
function drawRadarChart(scores) {
    const svg = document.getElementById('radar-chart');
    const cx = 100, cy = 95, r = 70;
    const keys = ['Originality', 'Quality', 'Clarity', 'Significance', 'Soundness', 'Presentation', 'Contribution'];
    const n = keys.length, maxVal = 4;
    let html = '';

    for (let lvl = 1; lvl <= 4; lvl++) {
        let pts = [];
        for (let i = 0; i < n; i++) {
            const ang = (Math.PI * 2 * i / n) - Math.PI / 2;
            const rr = r * lvl / maxVal;
            pts.push((cx + rr * Math.cos(ang)).toFixed(1) + ',' + (cy + rr * Math.sin(ang)).toFixed(1));
        }
        html += '<polygon points="' + pts.join(' ') + '" fill="none" stroke="var(--border)" stroke-width="0.5"/>';
    }
    for (let i = 0; i < n; i++) {
        const ang = (Math.PI * 2 * i / n) - Math.PI / 2;
        html += '<line x1="' + cx + '" y1="' + cy + '" x2="' + (cx + r * Math.cos(ang)).toFixed(1) +
            '" y2="' + (cy + r * Math.sin(ang)).toFixed(1) + '" stroke="var(--border)" stroke-width="0.5"/>';
    }
    let dataPts = [];
    for (let i = 0; i < n; i++) {
        const val = scores[keys[i]] || 0;
        const ang = (Math.PI * 2 * i / n) - Math.PI / 2;
        const rr = r * val / maxVal;
        dataPts.push((cx + rr * Math.cos(ang)).toFixed(1) + ',' + (cy + rr * Math.sin(ang)).toFixed(1));
    }
    html += '<polygon points="' + dataPts.join(' ') + '" fill="rgba(59,130,246,0.2)" stroke="var(--accent)" stroke-width="1.5"/>';

    for (let i = 0; i < n; i++) {
        const val = scores[keys[i]] || 0;
        const ang = (Math.PI * 2 * i / n) - Math.PI / 2;
        const rr = r * val / maxVal;
        const dx = cx + rr * Math.cos(ang), dy = cy + rr * Math.sin(ang);
        html += '<circle cx="' + dx.toFixed(1) + '" cy="' + dy.toFixed(1) + '" r="3" fill="var(--accent)"/>';
        const lr = r + 16;
        const lx = cx + lr * Math.cos(ang), ly = cy + lr * Math.sin(ang);
        html += '<text x="' + lx.toFixed(1) + '" y="' + ly.toFixed(1) +
            '" text-anchor="middle" dominant-baseline="central" fill="var(--text2)" font-size="8px" font-weight="600">' +
            keys[i].substring(0, 4) + '</text>';
    }
    svg.innerHTML = html;
}

// ---- Local Review Display ----
const LOCAL_CRITERIA = {
    'Novelty': { max: 10, desc: '原始性与创新程度' },
    'Soundness': { max: 10, desc: '方法与实验的严谨性' },
    'Presentation': { max: 10, desc: '写作清晰度与组织结构' },
};

function displayLocalReview(data) {
    document.getElementById('empty-state').style.display = 'none';
    document.getElementById('review-card').style.display = '';

    const badge = document.getElementById('decision-badge');
    const isAccept = data.decision && /accept/i.test(data.decision);
    badge.textContent = (data.decision || 'Unknown').toUpperCase();
    badge.className = 'decision-badge ' + (isAccept ? 'decision-accept' : 'decision-reject');

    const overall = data.overall || 0;
    document.getElementById('score-overall').textContent = overall > 0 ? overall + '/10' : '?';
    document.getElementById('score-overall-label').textContent = OVERALL_LABELS[overall] || '';
    document.getElementById('score-confidence').textContent = 'N/A';
    document.getElementById('score-confidence').style.color = 'var(--text2)';

    const scoreDetails = document.getElementById('score-details');
    const scores = data.scores || {};
    const hasScores = (scores.Novelty || 0) + (scores.Soundness || 0) + (scores.Presentation || 0) > 0;
    if (hasScores) {
        let sdHtml = '';
        for (const [name, info] of Object.entries(LOCAL_CRITERIA)) {
            const val = scores[name] || 0;
            const pct = (val / info.max * 100);
            const color = pct >= 70 ? 'var(--success)' : pct >= 40 ? 'var(--accent)' :
                pct >= 20 ? 'var(--warn)' : 'var(--danger)';
            sdHtml += '<div class="score-item">';
            sdHtml += '<div class="score-header"><span class="name">' + name + '</span>' +
                '<span class="criterion">' + info.desc + '</span>' +
                '<span class="val" style="color:' + color + '">' + val + '/' + info.max + '</span></div>';
            sdHtml += '<div class="score-bar"><div class="score-bar-fill" style="width:' +
                pct + '%;background:' + color + '"></div></div></div>';
        }
        scoreDetails.innerHTML = sdHtml;
        drawLocalRadarChart(scores);
    } else {
        scoreDetails.innerHTML =
            '<p style="font-size:0.8rem;color:var(--text2);">评分未能从模型输出中解析，请查看下方审稿意见原文。</p>';
        document.getElementById('radar-chart').innerHTML = '';
    }

    const rawText = data.raw_response || data.review_text || '';
    const summaryEl = document.getElementById('review-summary');
    if (rawText) {
        summaryEl.textContent = rawText;
        summaryEl.style.whiteSpace = 'pre-wrap';
        summaryEl.style.fontFamily = 'var(--font-mono)';
        summaryEl.style.fontSize = '0.82rem';
        summaryEl.style.lineHeight = '1.7';
        summaryEl.style.color = 'var(--text)';
    } else {
        summaryEl.textContent = '模型未返回审稿意见';
        summaryEl.style.color = 'var(--text2)';
    }

    document.getElementById('strengths-list').innerHTML = hasScores
        ? '<li style="color:var(--text2);font-size:0.8rem;">详见审稿意见原文</li>'
        : '<li style="color:var(--text2);font-size:0.8rem;">模型提供综合叙述性审稿</li>';
    document.getElementById('weaknesses-list').innerHTML =
        '<li style="color:var(--text2);font-size:0.8rem;">详见审稿意见原文</li>';
    document.getElementById('questions-list').innerHTML =
        '<li style="color:var(--text2);font-size:0.8rem;">N/A</li>';
    document.getElementById('limitations-list').innerHTML =
        '<li style="color:var(--text2);font-size:0.8rem;">N/A</li>';

    const def = data.defense_applied || {};
    document.getElementById('applied-defense').innerHTML =
        '<strong>防护状态:</strong> <span class="badge badge-success">StruQ 结构化查询防御</span>' +
        ' | 方法=' + (def.method || 'struq_structured_query') +
        ' | 模式=' + def.mode +
        ' | 内置防护=' + (def.struq_defense ?
            '<span style="color:var(--success)">ACTIVE</span>' :
            '<span style="color:var(--danger)">OFF</span>');

    document.getElementById('review-card').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---- Local Radar Chart ----
function drawLocalRadarChart(scores) {
    const svg = document.getElementById('radar-chart');
    const cx = 100, cy = 95, r = 70;
    const keys = ['Novelty', 'Soundness', 'Presentation'];
    const n = keys.length, maxVal = 10;
    let html = '';

    for (let lvl = 2; lvl <= 10; lvl += 2) {
        let pts = [];
        for (let i = 0; i < n; i++) {
            const ang = (Math.PI * 2 * i / n) - Math.PI / 2;
            const rr = r * lvl / maxVal;
            pts.push((cx + rr * Math.cos(ang)).toFixed(1) + ',' + (cy + rr * Math.sin(ang)).toFixed(1));
        }
        html += '<polygon points="' + pts.join(' ') + '" fill="none" stroke="var(--border)" stroke-width="0.5"/>';
    }
    for (let i = 0; i < n; i++) {
        const ang = (Math.PI * 2 * i / n) - Math.PI / 2;
        html += '<line x1="' + cx + '" y1="' + cy + '" x2="' + (cx + r * Math.cos(ang)).toFixed(1) +
            '" y2="' + (cy + r * Math.sin(ang)).toFixed(1) + '" stroke="var(--border)" stroke-width="0.5"/>';
    }
    let dataPts = [];
    for (let i = 0; i < n; i++) {
        const val = scores[keys[i]] || 0;
        const ang = (Math.PI * 2 * i / n) - Math.PI / 2;
        const rr = r * Math.min(val, maxVal) / maxVal;
        dataPts.push((cx + rr * Math.cos(ang)).toFixed(1) + ',' + (cy + rr * Math.sin(ang)).toFixed(1));
    }
    html += '<polygon points="' + dataPts.join(' ') + '" fill="rgba(34,197,94,0.2)" stroke="var(--success)" stroke-width="1.5"/>';

    for (let i = 0; i < n; i++) {
        const val = scores[keys[i]] || 0;
        const ang = (Math.PI * 2 * i / n) - Math.PI / 2;
        const rr = r * Math.min(val, maxVal) / maxVal;
        const dx = cx + rr * Math.cos(ang), dy = cy + rr * Math.sin(ang);
        html += '<circle cx="' + dx.toFixed(1) + '" cy="' + dy.toFixed(1) + '" r="4" fill="var(--success)"/>';
        const lr = r + 18;
        const lx = cx + lr * Math.cos(ang), ly = cy + lr * Math.sin(ang);
        html += '<text x="' + lx.toFixed(1) + '" y="' + ly.toFixed(1) +
            '" text-anchor="middle" dominant-baseline="central" fill="var(--text2)" font-size="8px" font-weight="600">' +
            keys[i] + '</text>';
    }
    svg.innerHTML = html;
}

// ---- Init ----
setDefenseMode('combined', document.querySelector('#defense-mode-group .btn[data-mode="combined"]'));
