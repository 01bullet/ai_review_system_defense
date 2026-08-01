// ==================== Paper Compare ====================

let allPaperHistory = [];

// ---- Load History for Compare ----
async function loadCompareHistory() {
    const listDiv = document.getElementById('compare-history-list');
    if (!listDiv) return;

    // Don't overwrite if already loaded successfully
    const wasLoaded = allPaperHistory.length > 0;

    try {
        const resp = await fetch('/api/history?limit=100&with_review_only=true');
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const data = await resp.json();
        if (data.papers !== undefined) {
            allPaperHistory = data.papers;
            renderComparePaperSelects();
            renderCompareHistoryList(data.papers);
        } else if (data.error) {
            throw new Error(data.error);
        }
    } catch (e) {
        if (!wasLoaded) {
            renderComparePaperSelects();  // still render empty dropdowns
        }
        listDiv.innerHTML =
            '<p style="color:var(--danger);text-align:center;padding:16px;">'
            + '加载历史记录失败: ' + escapeHtml(e.message)
            + ' <a href="javascript:loadCompareHistory()" style="color:var(--accent);text-decoration:underline;">重试</a>'
            + '</p>';
    }
}

function renderComparePaperSelects() {
    const sel1 = document.getElementById('compare-paper-select-1');
    const sel2 = document.getElementById('compare-paper-select-2');
    if (!sel1 || !sel2) return;

    const options = allPaperHistory.map(p =>
        '<option value="' + p.id + '">#' + p.id + ' ' + escapeHtml(p.filename) +
        ' (' + (p.latest_review ? p.latest_review.decision + ' ' + p.latest_review.overall + '/10' : '未审稿') +
        ')</option>'
    ).join('');

    sel1.innerHTML = '<option value="">-- 选择论文 1 --</option>' + options;
    sel2.innerHTML = '<option value="">-- 选择论文 2 --</option>' + options;
}

// ---- Render History List ----
function renderCompareHistoryList(papers) {
    const div = document.getElementById('compare-history-list');
    if (!div) return;

    if (!papers || papers.length === 0) {
        div.innerHTML = '<p style="color:var(--text2);text-align:center;padding:20px;">'
            + '暂无历史审稿记录，请先在「单篇审稿」中提交论文并完成审稿 '
            + '<a href="javascript:loadCompareHistory()" style="color:var(--accent);text-decoration:underline;font-size:0.8rem;">刷新</a>'
            + '</p>';
        return;
    }

    let html = '<div class="data-table-wrap"><table class="data-table">';
    html += '<thead><tr><th>ID</th><th>文件名</th><th>模型</th><th>总分</th><th>决策</th><th>时间</th></tr></thead><tbody>';

    papers.forEach(p => {
        const r = p.latest_review;
        const isAccept = r && /accept/i.test(r.decision);
        html += '<tr>';
        html += '<td>' + p.id + '</td>';
        html += '<td>' + escapeHtml(p.filename) + '</td>';
        html += '<td>' + (r ? escapeHtml(r.model_type) : '-') + '</td>';
        html += '<td>' + (r ? r.overall : '-') + '</td>';
        html += '<td>' + (r
            ? '<span class="badge ' + (isAccept ? 'badge-success' : 'badge-danger') + '">' + r.decision + '</span>'
            : '<span class="badge badge-warn">未审稿</span>') + '</td>';
        html += '<td>' + (r ? formatDate(r.review_time) : formatDate(p.upload_time)) + '</td>';
        html += '</tr>';
    });

    html += '</tbody></table></div>';
    div.innerHTML = html;
}

// ---- Compare from History ----
async function compareFromHistory() {
    const id1 = parseInt(document.getElementById('compare-paper-select-1').value);
    const id2 = parseInt(document.getElementById('compare-paper-select-2').value);

    if (!id1 || !id2) {
        showToast('请选择两篇论文', 'warn');
        return;
    }
    if (id1 === id2) {
        showToast('请选择不同的论文', 'warn');
        return;
    }

    const form = new FormData();
    form.append('paper_id_1', id1);
    form.append('paper_id_2', id2);

    const btn = document.getElementById('compare-history-btn');
    btn.disabled = true;
    btn.textContent = '对比中...';

    try {
        const resp = await fetch('/api/compare-from-history', { method: 'POST', body: form });
        const data = await resp.json();

        if (data.error) {
            showToast('对比失败: ' + data.error, 'danger');
            return;
        }

        renderCompareResult(data);
    } catch (e) {
        showToast('对比错误: ' + e.message, 'danger');
    } finally {
        btn.disabled = false;
        btn.textContent = '开始对比';
    }
}

// ---- Render Compare Result ----
function renderCompareResult(data) {
    const div = document.getElementById('compare-results');
    div.style.display = '';

    const p1 = data.paper1, p2 = data.paper2;
    const diffs = data.diffs || {};
    const better = data.better || 'tie';
    const betterLabel = { paper1: '论文 1 更优', paper2: '论文 2 更优', tie: '持平' };

    let html = '';

    // Winner banner
    html += '<div class="alert ' + (better === 'paper2' ? 'alert-success' : better === 'paper1' ? 'alert-info' : 'alert-warn') + '">';
    html += '<strong>' + betterLabel[better] + '</strong>';
    if (diffs.overall !== undefined) {
        html += ' (总分差: ' + (diffs.overall > 0 ? '+' : '') + diffs.overall + ')';
    }
    html += '</div>';

    // Side-by-side comparison grid
    html += '<div class="compare-grid">';

    // Paper 1
    html += '<div class="compare-col">';
    html += '<h3 class="compare-title">' + escapeHtml(p1.filename || '论文 1') + '</h3>';
    html += '<div class="decision-badge ' +
        (/accept/i.test(p1.decision) ? 'decision-accept' : 'decision-reject') + '">' +
        (p1.decision || '?') + '</div>';
    html += '<div class="gauge-value">' + (p1.overall || '?') + '<span style="font-size:0.7rem;color:var(--text2);">/10</span></div>';
    if (p1.scores) {
        for (const [k, v] of Object.entries(p1.scores)) {
            html += '<div class="score-item"><div class="score-header">' +
                '<span class="name">' + escapeHtml(k) + '</span>' +
                '<span class="val">' + (v || 0) + '</span></div></div>';
        }
    }
    html += '</div>';

    // Delta column
    html += '<div class="compare-col compare-delta">';
    html += '<h3 class="compare-title">差异</h3>';
    html += '<div style="margin-top:50px;"></div>';
    if (diffs.overall !== undefined) {
        const d = diffs.overall;
        const dcol = d > 0 ? 'var(--success)' : d < 0 ? 'var(--danger)' : 'var(--text2)';
        html += '<div class="gauge-value" style="color:' + dcol + '">' +
            (d > 0 ? '+' : '') + d + '<span style="font-size:0.7rem;color:var(--text2);">pts</span></div>';
    }
    for (const [k, v] of Object.entries(diffs)) {
        if (k === 'overall') continue;
        const dcol = v > 0 ? 'var(--success)' : v < 0 ? 'var(--danger)' : 'var(--text2)';
        html += '<div class="score-item"><div class="score-header">' +
            '<span class="val" style="color:' + dcol + '">' + (v > 0 ? '+' : '') + v + '</span></div></div>';
    }
    html += '</div>';

    // Paper 2
    html += '<div class="compare-col">';
    html += '<h3 class="compare-title">' + escapeHtml(p2.filename || '论文 2') + '</h3>';
    html += '<div class="decision-badge ' +
        (/accept/i.test(p2.decision) ? 'decision-accept' : 'decision-reject') + '">' +
        (p2.decision || '?') + '</div>';
    html += '<div class="gauge-value">' + (p2.overall || '?') + '<span style="font-size:0.7rem;color:var(--text2);">/10</span></div>';
    if (p2.scores) {
        for (const [k, v] of Object.entries(p2.scores)) {
            html += '<div class="score-item"><div class="score-header">' +
                '<span class="name">' + escapeHtml(k) + '</span>' +
                '<span class="val">' + (v || 0) + '</span></div></div>';
        }
    }
    html += '</div>';

    html += '</div>'; // close compare-grid

    // Bar chart visualization
    html += '<div class="compare-bar-chart" style="margin-top:16px;">';
    const maxScore = 10;
    const dimensions = Object.keys(p1.scores || {}).filter(k => k !== undefined);
    if (!dimensions.length) dimensions.push('Overall');
    dimensions.forEach(dim => {
        const v1 = (p1.scores || {})[dim] || p1.overall || 0;
        const v2 = (p2.scores || {})[dim] || p2.overall || 0;
        const pct1 = Math.round(v1 / maxScore * 100);
        const pct2 = Math.round(v2 / maxScore * 100);
        html += '<div style="margin-bottom:10px;">';
        html += '<div style="display:flex;justify-content:space-between;font-size:0.75rem;margin-bottom:2px;">' +
            '<span>' + escapeHtml(dim) + '</span>' +
            '<span><span style="color:var(--accent);">' + v1 + '</span> vs <span style="color:var(--success);">' + v2 + '</span></span></div>';
        html += '<div class="score-bar" style="height:8px;position:relative;">';
        html += '<div class="score-bar-fill" style="width:' + pct1 + '%;background:var(--accent);position:absolute;top:0;left:0;opacity:0.7;"></div>';
        html += '<div class="score-bar-fill" style="width:' + pct2 + '%;background:var(--success);position:absolute;top:0;left:0;"></div>';
        html += '</div></div>';
    });
    html += '<div style="font-size:0.7rem;color:var(--text2);margin-top:4px;">' +
        '<span style="color:var(--accent);">&#9632; 论文 1</span> ' +
        '<span style="color:var(--success);">&#9632; 论文 2</span></div></div>';

    div.innerHTML = html;
    div.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
