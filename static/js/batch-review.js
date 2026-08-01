// ==================== Batch Review ====================

let batchPapers = [];
let batchJobId = null;

// ---- Upload Zone ----
const batchUploadZone = document.getElementById('batch-upload-zone');
if (batchUploadZone) {
    batchUploadZone.addEventListener('click', () => document.getElementById('batch-file-input').click());
    ['dragover'].forEach(ev => batchUploadZone.addEventListener(ev, e => {
        e.preventDefault();
        batchUploadZone.classList.add('drag-over');
    }));
    ['dragleave', 'drop'].forEach(ev => batchUploadZone.addEventListener(ev, e => {
        e.preventDefault();
        batchUploadZone.classList.remove('drag-over');
    }));
    batchUploadZone.addEventListener('drop', e => {
        handleBatchFiles(e.dataTransfer.files);
    });
}

// ---- File Selection ----
function handleBatchInput(e) {
    handleBatchFiles(e.target.files);
}

async function handleBatchFiles(files) {
    if (!files || files.length === 0) return;

    const form = new FormData();
    for (const f of files) {
        form.append('files', f);
    }

    document.getElementById('batch-status').innerHTML =
        '<span style="color:var(--text2)">正在上传 ' + files.length + ' 个文件...</span>';
    document.getElementById('batch-upload-btn').disabled = true;

    try {
        const resp = await fetch('/api/upload-batch', { method: 'POST', body: form });
        const data = await resp.json();

        if (data.error) {
            document.getElementById('batch-status').innerHTML =
                '<span style="color:var(--danger)">上传失败: ' + data.error + '</span>';
            showToast('批量上传失败: ' + data.error, 'danger');
            return;
        }

        batchPapers = data.papers;
        renderBatchPaperList();
        document.getElementById('batch-status').innerHTML =
            '<span style="color:var(--success)">已上传 ' + data.total + ' 篇论文</span>';
        document.getElementById('batch-review-btn').disabled = data.total === 0;
        showToast('成功上传 ' + data.total + ' 篇论文', 'success');
    } catch (e) {
        showToast('上传错误: ' + e.message, 'danger');
    } finally {
        document.getElementById('batch-upload-btn').disabled = false;
    }
}

// ---- Render ----
function renderBatchPaperList() {
    const div = document.getElementById('batch-paper-list');
    if (!batchPapers.length) {
        div.innerHTML = '<div class="empty-state"><p style="color:var(--text2)">暂无论文</p></div>';
        return;
    }

    let html = '<div class="data-table-wrap"><table class="data-table">';
    html += '<thead><tr><th style="width:40px">#</th><th>文件名</th><th style="width:90px">大小</th><th style="width:100px">风险</th></tr></thead><tbody>';

    batchPapers.forEach((p, i) => {
        const sizeKB = (p.text_length / 1024).toFixed(1);
        const attackBadge = p.has_attack
            ? '<span class="badge badge-danger">攻击</span>'
            : '<span class="badge badge-success">安全</span>';
        const patterns = (p.injection_patterns || []).join(', ');
        html += '<tr>';
        html += '<td>' + (i + 1) + '</td>';
        html += '<td title="' + escapeHtml(patterns) + '">' + escapeHtml(p.filename) + '</td>';
        html += '<td>' + sizeKB + ' KB</td>';
        html += '<td>' + attackBadge + '</td>';
        html += '</tr>';
    });

    html += '</tbody></table></div>';
    div.innerHTML = html;
}

// ---- Run Batch Review ----
async function startBatchReview() {
    if (!batchPapers.length) {
        showToast('请先上传论文', 'warn');
        return;
    }

    const ids = batchPapers.map(p => p.id).join(',');
    const isLocal = isLocalModel();
    const endpoint = isLocal ? '/api/review-batch-local' : '/api/review-batch';

    const form = new FormData();
    form.append('paper_ids', ids);
    if (!isLocal) {
        form.append('model', currentModel);
        form.append('defense_mode', currentDefenseMode);
    } else {
        form.append('temperature', '0.1');
        form.append('model_version', currentModel);
    }

    const btn = document.getElementById('batch-review-btn');
    btn.disabled = true;
    btn.textContent = '审稿中...';
    document.getElementById('batch-progress').style.display = '';

    try {
        const resp = await fetch(endpoint, { method: 'POST', body: form });
        const data = await resp.json();

        if (data.error) {
            showToast('批量审稿失败: ' + data.error, 'danger');
            return;
        }

        batchJobId = data.batch_id;
        renderBatchResults(data);
        document.getElementById('batch-status').innerHTML =
            '<span style="color:var(--success)">审稿完成！共处理 ' + data.completed + ' 篇</span>';
        showToast('批量审稿完成！', 'success');
    } catch (e) {
        showToast('批量审稿错误: ' + e.message, 'danger');
    } finally {
        btn.disabled = false;
        btn.textContent = '开始批量审稿';
        document.getElementById('batch-progress').style.display = 'none';
    }
}

// ---- Results ----
function renderBatchResults(data) {
    const div = document.getElementById('batch-results');
    if (!data.results || !data.results.length) {
        div.innerHTML = '<p style="color:var(--text2)">无结果</p>';
        return;
    }

    let accepts = 0, rejects = 0, errors = 0;
    let html = '<div class="data-table-wrap"><table class="data-table">';
    html += '<thead><tr><th>#</th><th>论文 ID</th><th>决策</th><th>总分</th><th>状态</th></tr></thead><tbody>';

    data.results.forEach((r, i) => {
        let decisionHtml, statusHtml;
        if (r.error) {
            decisionHtml = '-';
            statusHtml = '<span class="badge badge-danger">失败</span>';
            errors++;
        } else {
            const isAccept = /accept/i.test(r.decision);
            if (isAccept) accepts++; else rejects++;
            decisionHtml = '<span class="badge ' + (isAccept ? 'badge-success' : 'badge-danger') + '">' +
                r.decision + '</span>';
            statusHtml = '<span class="badge badge-success">完成</span>';
        }
        html += '<tr>';
        html += '<td>' + (i + 1) + '</td>';
        html += '<td>' + (r.paper_id || '?') + '</td>';
        html += '<td>' + decisionHtml + '</td>';
        html += '<td>' + (r.overall || '-') + '</td>';
        html += '<td>' + statusHtml + '</td>';
        html += '</tr>';
    });

    html += '</tbody></table></div>';

    // Summary
    html += '<div class="compare-meta">';
    html += '<span class="badge badge-success">Accept: ' + accepts + '</span> ';
    html += '<span class="badge badge-danger">Reject: ' + rejects + '</span> ';
    if (errors) html += '<span class="badge badge-warn">Error: ' + errors + '</span>';
    html += '</div>';

    div.innerHTML = html;
    div.style.display = '';
}

// ---- Drag & Drop (initial) ----
document.getElementById('batch-file-input').onchange = handleBatchInput;
