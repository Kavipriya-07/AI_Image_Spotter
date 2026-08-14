const $ = (selector) => document.querySelector(selector);
let selectedFile = null;

function percent(value) { return `${Math.round(value * 100)}%`; }
function fileSize(bytes) { return bytes < 1024 * 1024 ? `${Math.round(bytes / 1024)} KB` : `${(bytes / 1024 / 1024).toFixed(2)} MB`; }
function toast(message, isError = false) { const el = $('#toast'); el.textContent = message; el.className = `show${isError ? ' error' : ''}`; setTimeout(() => el.className = '', 3500); }
function esc(text) { const div = document.createElement('div'); div.textContent = text; return div.innerHTML; }

function navigate(page) {
  document.querySelectorAll('.page').forEach(el => el.classList.toggle('active', el.id === page));
  document.querySelectorAll('.nav-link').forEach(el => el.classList.toggle('active', el.dataset.page === page));
  $('#pageTitle').textContent = ({dashboard:'Dashboard',analyze:'Analyze Image',history:'History',statistics:'Statistics',model:'Model Info',settings:'Settings'})[page];
  $('.sidebar').classList.remove('open');
  if (page === 'history') loadHistory();
  if (page === 'statistics') loadStatistics();
  if (page === 'model') loadModelInfo();
}

document.querySelectorAll('.nav-link').forEach(button => button.addEventListener('click', () => navigate(button.dataset.page)));
document.querySelectorAll('.go-analyze').forEach(button => button.addEventListener('click', () => navigate('analyze')));
document.querySelector('[data-page-target="history"]').addEventListener('click', () => navigate('history'));
$('.mobile-menu').addEventListener('click', () => $('.sidebar').classList.toggle('open'));

const fileInput = $('#fileInput'), dropZone = $('#dropZone');
$('#browseButton').addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => selectImage(fileInput.files[0]));
['dragenter','dragover'].forEach(event => dropZone.addEventListener(event, e => { e.preventDefault(); dropZone.classList.add('drag'); }));
['dragleave','drop'].forEach(event => dropZone.addEventListener(event, e => { e.preventDefault(); dropZone.classList.remove('drag'); }));
dropZone.addEventListener('drop', e => selectImage(e.dataTransfer.files[0]));
$('#removeImage').addEventListener('click', clearSelection);

function selectImage(file) {
  if (!file) return;
  const types = ['image/jpeg','image/png','image/webp'];
  if (!types.includes(file.type)) return toast('Use a JPG, JPEG, PNG, or WEBP image.', true);
  if (file.size > 10 * 1024 * 1024) return toast('Image must be 10 MB or smaller.', true);
  selectedFile = file;
  $('#previewImage').src = URL.createObjectURL(file);
  $('#previewName').textContent = file.name;
  $('#previewMeta').textContent = `${file.type.replace('image/','').toUpperCase()} · ${fileSize(file.size)}`;
  $('#uploadPrompt').classList.add('hidden'); $('#previewBox').classList.remove('hidden'); $('#analyzeButton').disabled = false;
}
function clearSelection() { selectedFile = null; fileInput.value = ''; $('#previewImage').src = ''; $('#uploadPrompt').classList.remove('hidden'); $('#previewBox').classList.add('hidden'); $('#analyzeButton').disabled = true; }

$('#analyzeButton').addEventListener('click', async () => {
  if (!selectedFile) return toast('Please select an image first.', true);
  $('#loading').classList.remove('hidden'); $('#results').classList.add('hidden'); $('#analyzeButton').disabled = true;
  const form = new FormData(); form.append('image', selectedFile);
  try {
    const response = await fetch('/api/predict', {method:'POST', body:form}); const data = await response.json();
    if (!response.ok || data.status !== 'success') throw new Error(data.message || 'Prediction failed.');
    renderResult(data); toast('Analysis complete.'); loadDashboard();
  } catch (error) { toast(error.message || 'Could not reach the server.', true); }
  finally { $('#loading').classList.add('hidden'); $('#analyzeButton').disabled = false; }
});

function renderResult(data) {
  const ai = percent(data.ai_probability), real = percent(data.real_probability);
  const indicators = data.demo_indicators.map(i => `<div class="indicator"><label><span>${i.label}</span><span>${i.value}%</span></label><div class="track"><i style="width:${i.value}%"></i></div></div>`).join('');
  $('#results').innerHTML = `<div class="result-grid"><section class="card result-hero"><p class="eyebrow">PREDICTION RESULT</p><span class="result-label">${data.model_used}</span><div class="prediction">${data.prediction}</div><div class="confidence-ring">${percent(data.confidence)}</div><span class="result-label">Model score</span><div class="bar-row"><div><span>AI-Generated</span><b>${ai}</b></div><div class="track"><i style="width:${ai}"></i></div></div><div class="bar-row"><div><span>Real</span><b>${real}</b></div><div class="track real"><i style="width:${real}"></i></div></div></section><section class="card"><p class="eyebrow">UPLOADED IMAGE</p><div class="preview-box"><img src="${data.image_url}" alt="Analyzed image"><div><strong>${esc(data.filename)}</strong><span>${data.file_type} · ${fileSize(data.file_size)}</span><span>${data.resolution}</span></div></div><div class="details"><p class="eyebrow">ANALYSIS DETAILS</p><dl><div><dt>File name</dt><dd>${esc(data.filename)}</dd></div><div><dt>Analysis time</dt><dd>${data.analysis_time}s</dd></div><div><dt>Model used</dt><dd>${data.model_used}</dd></div><div><dt>Prediction</dt><dd>${data.prediction} (${percent(data.confidence)})</dd></div></dl></div></section></div><section class="card details"><p class="eyebrow">MODEL PROBABILITY DETAILS</p><h3>Classifier output</h3><p class="form-note" style="text-align:left">Values are the classifier's raw output for this image, not calibrated certainty. The model scores 56.4% accuracy on held-out test data, so a high score does not mean a reliable verdict.</p>${indicators}</section>`;
  $('#results').classList.remove('hidden'); $('#results').scrollIntoView({behavior:'smooth',block:'start'});
}

function recordRow(item) { return `<div class="recent-row"><img src="${item.image_url}" alt=""><div class="file-grow"><strong>${esc(item.filename)}</strong><small>${new Date(item.analyzed_at).toLocaleString()}</small></div><span class="pill ${item.prediction === 'REAL' ? 'real':'ai'}">${item.prediction}</span><b>${percent(item.confidence)}</b></div>`; }
async function loadDashboard() { try { const data = await (await fetch('/api/statistics')).json(); $('#statTotal').textContent=data.total; $('#statReal').textContent=data.real; $('#statAI').textContent=data.ai_generated; $('#statConfidence').textContent=data.total ? percent(data.average_confidence) : '—'; $('#recentList').className='recent-list'; $('#recentList').innerHTML=data.recent.length ? data.recent.map(recordRow).join('') : 'No images analyzed yet. Start with an image check.'; if (!data.recent.length) $('#recentList').classList.add('empty'); } catch { toast('Unable to load dashboard data.',true); } }
async function loadHistory() { const data=await (await fetch('/api/history')).json(); $('#historyList').innerHTML=data.history.length ? data.history.map(item => `<div class="history-row"><img src="${item.image_url}" alt=""><div class="file-grow"><strong>${esc(item.filename)}</strong><small>${item.file_type} · ${item.resolution} · ${item.status}</small></div><span class="pill ${item.prediction === 'REAL' ? 'real':'ai'}">${item.prediction}</span><b>${percent(item.confidence)}</b><time>${new Date(item.analyzed_at).toLocaleString()}</time></div>`).join('') : '<div class="empty">No analyses in history yet.</div>'; }
async function loadStatistics() { const data=await (await fetch('/api/statistics')).json(); $('#statsTotal').textContent=data.total; $('#statsReal').textContent=data.real; $('#statsAI').textContent=data.ai_generated; $('#statsAverage').textContent=data.total ? percent(data.average_confidence) : '—'; $('#statsRecent').className='recent-list'; $('#statsRecent').innerHTML=data.recent.length ? data.recent.map(recordRow).join('') : 'No data available.'; if(!data.recent.length) $('#statsRecent').classList.add('empty'); }
async function loadModelInfo(){ try { await fetch('/api/model-info'); } catch { toast('Model information is unavailable.',true); } }
async function clearHistory(){ if(!confirm('Clear all analysis history?')) return; const response=await fetch('/api/history',{method:'DELETE'}); if(response.ok){toast('History cleared.'); loadDashboard(); loadHistory(); loadStatistics();}else toast('Could not clear history.',true); }
$('#clearHistory').addEventListener('click',clearHistory); $('.clear-history-alt').addEventListener('click',clearHistory);
$('#themeToggle').addEventListener('change', e => { document.body.classList.toggle('light',e.target.checked); localStorage.setItem('ais-theme',e.target.checked?'light':'dark'); });
if(localStorage.getItem('ais-theme')==='light') { document.body.classList.add('light'); $('#themeToggle').checked=true; }
loadDashboard();
