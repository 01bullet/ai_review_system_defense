// ==================== Statistics Dashboard ====================

async function loadStats() {
    const div = document.getElementById('stats-content');
    if (!div) return;
    div.innerHTML = '<p style="color:var(--text2);text-align:center;padding:40px;">加载统计中...</p>';

    try {
        const resp = await fetch('/api/stats');
        const data = await resp.json();
        if (data.error) {
            div.innerHTML = '<p style="color:var(--danger);">加载失败: ' + data.error + '</p>';
            return;
        }
        renderStats(data);
    } catch (e) {
        div.innerHTML = '<p style="color:var(--danger);">加载失败: ' + e.message + '</p>';
    }
}

function renderStats(data) {
    const div = document.getElementById('stats-content');

    let html = '';

    // ---- Summary Cards ----
    html += '<div class="stats-summary">';
    html += _statCard('总论文数', data.total_papers || 0, 'var(--accent)');
    html += _statCard('总审稿数', data.total_reviews || 0, 'var(--success)');
    html += _statCard('API 审稿', data.api_reviews || 0, 'var(--info)');
    html += _statCard('本地审稿', data.local_reviews || 0, 'var(--warn)');
    html += '</div>';

    // ---- Decision Distribution ----
    html += '<div class="stats-grid">';

    html += '<div class="stats-card">';
    html += '<h3 class="stats-card-title">决策分布</h3>';
    const decisions = data.decisions || {};
    const accepts = decisions['Accept'] || 0;
    const rejects = decisions['Reject'] || 0;
    const others = Object.entries(decisions).reduce((s, [k, v]) => s + (k === 'Accept' || k === 'Reject' ? 0 : v), 0);
    const total = accepts + rejects + others || 1;
    html += '<div style="display:flex;align-items:center;gap:16px;">';
    html += '<div class="donut-chart" id="decision-donut"></div>';
    html += '<div>';
    html += '<div style="margin-bottom:6px;"><span style="color:var(--success);">&#9632;</span> Accept: ' + accepts + ' (' + Math.round(accepts / total * 100) + '%)</div>';
    html += '<div style="margin-bottom:6px;"><span style="color:var(--danger);">&#9632;</span> Reject: ' + rejects + ' (' + Math.round(rejects / total * 100) + '%)</div>';
    if (others) html += '<div><span style="color:var(--warn);">&#9632;</span> Other: ' + others + '</div>';
    html += '</div></div>';
    html += '</div>';

    // ---- Average Scores by Model ----
    html += '<div class="stats-card">';
    html += '<h3 class="stats-card-title">平均评分（按模型）</h3>';
    html += '<div class="data-table-wrap"><table class="data-table">';
    html += '<thead><tr><th>模型</th><th>Novelty</th><th>Soundness</th><th>Presentation</th><th>Overall</th><th>数量</th></tr></thead><tbody>';
    const avgScores = data.average_scores || [];
    if (avgScores.length) {
        avgScores.forEach(r => {
            html += '<tr>';
            html += '<td>' + escapeHtml(r.model_type) + '</td>';
            html += '<td>' + (r.avg_novelty || 0).toFixed(1) + '</td>';
            html += '<td>' + (r.avg_soundness || 0).toFixed(1) + '</td>';
            html += '<td>' + (r.avg_presentation || 0).toFixed(1) + '</td>';
            html += '<td>' + (r.avg_overall || 0).toFixed(1) + '</td>';
            html += '<td>' + r.c + '</td>';
            html += '</tr>';
        });
    } else {
        html += '<tr><td colspan="6" style="text-align:center;color:var(--text2);">暂无数据</td></tr>';
    }
    html += '</tbody></table></div>';
    html += '</div>';

    // ---- Score Distribution Bar Chart ----
    html += '<div class="stats-card" style="grid-column: span 2;">';
    html += '<h3 class="stats-card-title">总分分布</h3>';
    const scoreDist = data.score_distribution || {};
    if (Object.keys(scoreDist).length) {
        const maxCount = Math.max(1, ...Object.values(scoreDist));
        html += '<div class="bar-chart">';
        for (let s = 1; s <= 10; s++) {
            const cnt = scoreDist[s] || 0;
            const pct = Math.round(cnt / maxCount * 100);
            const color = s >= 7 ? 'var(--success)' : s >= 5 ? 'var(--warn)' : 'var(--danger)';
            html += '<div class="bar-chart-row">';
            html += '<span class="bar-chart-label">' + s + '</span>';
            html += '<div class="bar-chart-track"><div class="bar-chart-fill" style="width:' + pct + '%;background:' + color + ';"></div></div>';
            html += '<span class="bar-chart-value">' + cnt + '</span>';
            html += '</div>';
        }
        html += '</div>';
    } else {
        html += '<p style="color:var(--text2);">暂无数据</p>';
    }
    html += '</div>';

    // ---- Defense Stats ----
    html += '<div class="stats-card">';
    html += '<h3 class="stats-card-title">防御统计</h3>';
    html += '<div style="display:flex;gap:24px;">';
    html += '<div style="text-align:center;"><div style="font-size:2rem;font-weight:700;color:var(--success);">' + (data.defense_active || 0) + '</div><div style="font-size:0.75rem;color:var(--text2);">激活防御</div></div>';
    html += '<div style="text-align:center;"><div style="font-size:2rem;font-weight:700;color:var(--warn);">' + (data.defense_triggered || 0) + '</div><div style="font-size:0.75rem;color:var(--text2);">触发防御</div></div>';
    html += '</div>';
    html += '</div>';

    // ---- Recent Reviews ----
    html += '<div class="stats-card">';
    html += '<h3 class="stats-card-title">最近审稿</h3>';
    html += '<div class="data-table-wrap"><table class="data-table">';
    html += '<thead><tr><th>时间</th><th>模型</th><th>总分</th><th>决策</th></tr></thead><tbody>';
    const recent = data.recent || [];
    if (recent.length) {
        recent.forEach(r => {
            const isAccept = /accept/i.test(r.decision);
            html += '<tr>';
            html += '<td>' + formatDate(r.review_time) + '</td>';
            html += '<td>' + escapeHtml(r.model_type) + '</td>';
            html += '<td>' + (r.overall || '-') + '</td>';
            html += '<td><span class="badge ' + (isAccept ? 'badge-success' : 'badge-danger') + '">' + r.decision + '</span></td>';
            html += '</tr>';
        });
    } else {
        html += '<tr><td colspan="4" style="text-align:center;color:var(--text2);">暂无数据</td></tr>';
    }
    html += '</tbody></table></div>';
    html += '</div>';

    html += '</div>'; // close stats-grid

    div.innerHTML = html;

    // Draw donut chart for decisions
    renderDecisionDonut(accepts, rejects, others);

    // Refresh compare paper lists
    if (typeof loadCompareHistory === 'function') loadCompareHistory();
}

function _statCard(label, value, color) {
    return '<div class="stat-card">' +
        '<div class="stat-card-value" style="color:' + color + ';">' + value + '</div>' +
        '<div class="stat-card-label">' + label + '</div>' +
        '</div>';
}

// ---- Donut Chart ----
function renderDecisionDonut(accepts, rejects, others) {
    const svg = document.getElementById('decision-donut');
    if (!svg) return;

    const total = accepts + rejects + others || 1;
    const cx = 50, cy = 50, r = 40, sw = 12;

    function polarToCartesian(cx, cy, r, angleDeg) {
        const rad = (angleDeg - 90) * Math.PI / 180;
        return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
    }

    function describeArc(cx, cy, r, startAngle, endAngle) {
        const s = polarToCartesian(cx, cy, r, endAngle);
        const e = polarToCartesian(cx, cy, r, startAngle);
        const large = endAngle - startAngle > 180 ? 1 : 0;
        return 'M ' + s.x + ' ' + s.y + ' A ' + r + ' ' + r + ' 0 ' + large + ' 0 ' + e.x + ' ' + e.y;
    }

    const acceptPct = accepts / total * 360;
    const rejectPct = rejects / total * 360;
    const otherPct = others / total * 360;

    let html = '<svg width="120" height="120" viewBox="0 0 100 100">';
    let offset = 0;
    if (accepts > 0) {
        html += '<path d="' + describeArc(cx, cy, r, offset, offset + acceptPct) +
            '" fill="none" stroke="var(--success)" stroke-width="' + sw + '" stroke-linecap="round"/>';
        offset += acceptPct;
    }
    if (rejects > 0) {
        html += '<path d="' + describeArc(cx, cy, r, offset, offset + rejectPct) +
            '" fill="none" stroke="var(--danger)" stroke-width="' + sw + '" stroke-linecap="round"/>';
        offset += rejectPct;
    }
    if (others > 0) {
        html += '<path d="' + describeArc(cx, cy, r, offset, offset + otherPct) +
            '" fill="none" stroke="var(--warn)" stroke-width="' + sw + '" stroke-linecap="round"/>';
    }
    html += '<text x="50" y="50" text-anchor="middle" dominant-baseline="central" fill="var(--text)" font-size="14" font-weight="700">' + total + '</text>';
    html += '</svg>';
    svg.innerHTML = html;
}
