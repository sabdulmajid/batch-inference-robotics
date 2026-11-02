import {MAX_BUNDLE_BYTES, MAX_MEDIA_BYTES, sha256, parseBundle, reviewable, latestReview, appendReview, exportBundle, historyState, outcomeState} from './bundle.js';
import {inspectStoredDraft, restoreStoredDraft} from './recovery.js';

const $ = id => document.getElementById(id);
const video = $('video');
let bundle = null, payload = null, episodeIndex = 0, selectedIndex = 0;
let examples = [], trustedMedia = {}, localMedia = new Map(), loadedMedia = new Map();
let tabOnlyDraft = false;
let loadVersion = 0, bundleIntent = 0, mediaAbort = null, playingInterval = false, dirtyText = false, unsavedBundle = false, pendingDraft = null;
const episode = () => payload?.episodes[episodeIndex];
const selected = () => episode()?.records[selectedIndex];
const text = (id, value) => { $(id).textContent = value; };
const show = (id, visible) => $(id).classList.toggle('hidden', !visible);
function notice(message, error = false) { text('global-notice', message); $('global-notice').classList.toggle('error', error); }
function badge(id, label, tone = '') { text(id, label); $(id).className = `badge ${tone}`; }
const seconds = value => Number.isFinite(value) ? value.toFixed(2) + ' s' : 'Not recorded';
const number = value => typeof value === 'number' ? Number(value.toFixed(5)).toString() : 'Not recorded';
function safeName(value) { return value.replace(/[^A-Za-z0-9._-]+/g, '-').slice(0, 80) || 'rollout'; }
function beforeSwitch() {
  if (dirtyText && !confirm('This caption has unsaved text. Discard that text and continue?')) return false;
  dirtyText = false;
  return true;
}
function draftKey(hash) { return `bir-review-draft-v1:${hash}`; }
function persistDraft() {
  try {
    localStorage.setItem(draftKey(bundle.payload_sha256), JSON.stringify({schema_version:'bir_browser_draft_v1', payload_sha256:bundle.payload_sha256, reviews:bundle.reviews}));
    return true;
  } catch { return false; }
}
async function checkDraft() {
  const capturedBundle = bundle, payloadHash = capturedBundle.payload_sha256;
  pendingDraft = null; show('recovery', false);
  try {
    const raw = localStorage.getItem(draftKey(payloadHash));
    if (!raw) return;
    const candidate = await inspectStoredDraft(raw, capturedBundle, () => bundle);
    if (bundle !== capturedBundle || bundle.payload_sha256 !== payloadHash || !candidate) return;
    if (candidate.kind === 'identical') return;
    if (candidate.kind === 'conflict') {
      notice('The browser has a different or older review history for this payload. The imported history is retained. Download it before making changes.');
      return;
    }
    pendingDraft = candidate;
    text('recovery-copy', `${candidate.reviews.length - capturedBundle.reviews.length} additional saved correction event(s) are stored on this device for this exact payload. The imported file has not changed.`);
    show('recovery', true);
  } catch {
    if (bundle === capturedBundle && bundle.payload_sha256 === payloadHash) {
      notice('A local draft could not be verified. It was not applied. The imported bundle remains usable.', true);
    }
  }
}

async function limitedFetch(url, limit, signal) {
  const response = await fetch(url, {credentials:'omit', cache:'no-cache', redirect:'error', signal});
  if (!response.ok) throw Error(`Could not load the public example (${response.status}). You can import a compatible local bundle.`);
  const announced = Number(response.headers.get('content-length'));
  if (announced > limit) throw Error('The public asset exceeds the supported size.');
  const reader = response.body.getReader(), chunks = [];
  let length = 0;
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    length += value.length;
    if (length > limit) { await reader.cancel(); throw Error('The public asset exceeds the supported size.'); }
    chunks.push(value);
  }
  return new Blob(chunks, {type:response.headers.get('content-type') || 'application/octet-stream'});
}
function exampleUrl(path) {
  const root = new URL('./examples/', location.href), url = new URL(path, root);
  if (url.origin !== location.origin || !url.pathname.startsWith(root.pathname) || url.search || url.hash) throw Error('Public example assets must use this site’s examples directory.');
  return url;
}

async function openBundle(raw, {publicExample = null, intent = ++bundleIntent} = {}) {
  const parsed = await parseBundle(raw);
  if (intent !== bundleIntent) return;
  if (!beforeSwitch()) return;
  if (tabOnlyDraft && !confirm('These corrections exist only in this tab. Download the current bundle before opening another file. Discard this unsaved bundle and continue?')) return;
  loadVersion++; video.pause(); playingInterval = false;
  bundle = parsed.bundle; payload = parsed.payload; episodeIndex = 0; selectedIndex = 0;
  dirtyText = false; unsavedBundle = false; tabOnlyDraft = false;
  trustedMedia = publicExample?.media || {};
  $('episode-picker').replaceChildren(...payload.episodes.map((item, index) => {
    const option = document.createElement('option'); option.value = String(index); option.textContent = item.episode_id; return option;
  }));
  $('episode-picker').disabled = false; $('download-bundle').disabled = false;
  $('workspace').setAttribute('aria-busy', 'false');
  const attribution = payload.provenance?.attribution || payload.provenance?.license || publicExample?.attribution;
  text('attribution', attribution ? (typeof attribution === 'string' ? attribution : JSON.stringify(attribution)) : 'Keep source attribution with exported bundles. Own-file review stays in this browser.');
  text('save-state', 'Imported history verified. No new changes.');
  $('save-state').className = 'save-state';
  notice(publicExample ? 'Public precomputed example. No model ran in this browser. Your edits remain local.' : 'Bundle opened locally. No file was uploaded. Select matching media files to play their recorded video.');
  renderEpisode();
  await checkDraft();
}

async function openExample() {
  const chosen = examples[Number($('example-picker').value)];
  if (!chosen) return;
  const intent = ++bundleIntent;
  $('load-example').disabled = true;
  notice('Verifying the public example…');
  try {
    const file = await limitedFetch(exampleUrl(chosen.bundle), MAX_BUNDLE_BYTES);
    await openBundle(await file.text(), {publicExample:chosen, intent});
  } catch (error) { notice(error.message, true); }
  finally { $('load-example').disabled = false; $('workspace').setAttribute('aria-busy', 'false'); }
}

function mainMedia() { return episode()?.media.find(x => x.camera === 'main') || episode()?.media[0]; }
async function loadVideo() {
  const item = mainMedia(), version = ++loadVersion;
  mediaAbort?.abort(); mediaAbort = new AbortController();
  video.pause(); playingInterval = false; video.removeAttribute('src'); video.load();
  show('media-empty', true); text('media-state', 'Media unavailable');
  if (!item) { text('media-state', 'No source video is attached'); return; }
  let available = localMedia.get(item.source_sha256) || loadedMedia.get(item.source_sha256);
  const publicPath = trustedMedia[item.source_sha256];
  try {
    if (!available && typeof publicPath === 'string') {
      text('media-state', 'Loading and verifying public video…');
      const blob = await limitedFetch(exampleUrl(publicPath), MAX_MEDIA_BYTES, mediaAbort.signal);
      if (await sha256(await blob.arrayBuffer()) !== item.source_sha256) throw Error('Public media checksum failed. Playback is blocked.');
      available = {url:URL.createObjectURL(blob), origin:'public', filename:item.filename};
      loadedMedia.set(item.source_sha256, available);
      while (loadedMedia.size > 2) {
        const [oldHash, old] = loadedMedia.entries().next().value;
        loadedMedia.delete(oldHash); URL.revokeObjectURL(old.url);
      }
    }
    if (version !== loadVersion) return;
    if (!available) { text('media-state', `Choose matching local media · ${item.filename}`); return; }
    video.src = available.url;
    show('media-empty', false);
    text('media-state', `${available.origin === 'local' ? 'Local media' : 'Public example'} · source checksum verified`);
    $('play-interval').disabled = !Number.isFinite(selected()?.start_s);
  } catch (error) { if (version === loadVersion && error.name !== 'AbortError') { notice(error.message, true); text('media-state', 'Media verification failed'); } }
}

function renderEpisode() {
  const ep = episode(); if (!ep) return;
  $('episode-picker').value = String(episodeIndex);
  text('episode-title', ep.episode_id); text('task-description', ep.task ? `Task: ${ep.task}` : 'No task description is attached.');
  text('episode-source', ep.trajectory ? 'RECORDED ROBOT / SIMULATOR ROLLOUT' : 'VIDEO ANNOTATION EXAMPLE');
  const duration = mainMedia()?.duration_s;
  text('duration-label', seconds(duration));
  const all = ep.records, complete = all.filter(reviewable), precomputed = all.some(x => x.generation?.precomputed);
  badge('annotation-badge', !all.length ? 'Unavailable' : precomputed ? 'Precomputed' : `${complete.length}/${all.length} complete`, precomputed || complete.length ? 'blue' : 'warn');
  selectedIndex = Math.min(selectedIndex, Math.max(0, all.length - 1));
  renderIntervals(); renderSelected(); renderHistory(); renderTrajectory(); renderOutcome();
  text('training-state', ep.trajectory ? 'Recorded transitions are attached. CLI validation and an explicit reward source are still required for trainer export. Captions are not policy observations.' : 'This episode has no recorded actions. It supports annotation review, but cannot be exported as training transitions.');
  loadVideo();
}

function renderIntervals() {
  const ep = episode(); if (!ep) return;
  $('interval-list').replaceChildren(); $('timeline').replaceChildren();
  if (!ep.records.length) {
    const p = document.createElement('p'); p.className = 'empty-copy';
    p.textContent = 'No generated action captions are attached. The video and recorded trajectory can still be inspected. Source task text is not a FLORA prediction.';
    $('interval-list').append(p);
  }
  ep.records.forEach((row, i) => {
    const review = latestReview(bundle.reviews, row.record_id), usable = reviewable(row);
    const button = document.createElement('button'); button.className = `interval-item${i === selectedIndex ? ' selected' : ''}`;
    button.setAttribute('aria-current', String(i === selectedIndex));
    const clock = document.createElement('span'); clock.className = 'interval-clock'; clock.textContent = row.start_s === null ? 'No interval' : `${row.start_s.toFixed(2)}–${row.end_s.toFixed(2)} s`;
    const caption = document.createElement('span'); caption.className = 'interval-caption'; caption.textContent = usable && review?.reviewed_caption !== null && review?.reviewed_caption !== undefined ? review.reviewed_caption : row.generated_caption || 'Caption unavailable';
    const status = document.createElement('span'); status.className = 'interval-status'; status.textContent = usable && review?.reviewed_caption ? 'Reviewed' : row.inference_status;
    button.append(clock, caption, status); button.addEventListener('click', () => selectInterval(i, true)); $('interval-list').append(button);
    const duration = mainMedia()?.duration_s;
    if (duration && Number.isFinite(row.start_s)) {
      const segment = document.createElement('button'); segment.className = `timeline-segment${i === selectedIndex ? ' selected' : ''}`;
      segment.style.left = `${Math.max(0, row.start_s / duration * 100)}%`;
      segment.style.width = `${Math.min(100, (row.end_s - row.start_s) / duration * 100)}%`;
      segment.dataset.index = String(i); segment.title = `${clock.textContent} · ${row.generated_caption || row.inference_status}`;
      segment.setAttribute('aria-label', `Select interval ${i + 1}, ${clock.textContent}`);
      segment.addEventListener('click', () => selectInterval(i, true)); $('timeline').append(segment);
    }
  });
  text('interval-count', ep.records.length ? `${selectedIndex + 1} of ${ep.records.length}` : 'No caption intervals');
  $('previous-interval').disabled = selectedIndex <= 0;
  $('next-interval').disabled = selectedIndex >= ep.records.length - 1;
}

function renderSelected() {
  const row = selected(), usable = reviewable(row), review = row && latestReview(bundle.reviews, row.record_id);
  badge('record-state', !row ? 'Unavailable' : !usable ? row.inference_status : review?.reviewed_caption ? 'Reviewed' : row.generation?.precomputed ? 'Precomputed' : 'Generated', usable ? 'blue' : 'warn');
  text('selected-time', Number.isFinite(row?.start_s) ? `${seconds(row.start_s)} – ${seconds(row.end_s)}` : 'No completed caption selected.');
  text('generated-caption', row?.generated_caption || 'No generated caption is available.');
  text('record-warning', !usable ? 'New corrections require a completed generated caption. Historical edits are kept below.' : 'A plausible caption is not proof of visual accuracy. Check the video before saving.');
  $('reviewed-caption').value = usable ? (review?.reviewed_caption ?? row.generated_caption) : '';
  $('reviewed-caption').disabled = !usable; $('reviewer').disabled = !usable;
  $('save-correction').disabled = !usable; $('restore-generated').disabled = !usable || !review || review.reviewed_caption === null;
  $('play-interval').disabled = !Number.isFinite(row?.start_s) || !video.getAttribute('src');
  text('record-provenance', row ? JSON.stringify({record_id:row.record_id, request_id:row.request_id, source_sha256:row.source_sha256, generation:row.generation, inference_status:row.inference_status, failure:row.failure, latest_review:review}, null, 2) : 'No generated record.');
  dirtyText = false;
}

function seek(time) {
  if (!video.getAttribute('src') || !Number.isFinite(time)) return;
  const apply = () => { video.currentTime = Math.max(0, Math.min(time, video.duration || time)); };
  if (video.readyState >= 1) apply(); else video.addEventListener('loadedmetadata', apply, {once:true});
}
function selectInterval(index, seekToStart = false) {
  if (!beforeSwitch()) return;
  selectedIndex = Math.max(0, Math.min(episode().records.length - 1, index));
  playingInterval = false; video.pause(); renderIntervals(); renderSelected();
  if (seekToStart) seek(selected()?.start_s);
}

function renderHistory() {
  const entries = historyState(payload, bundle.reviews);
  text('history-count', String(entries.length)); $('review-history').replaceChildren();
  if (!entries.length) { const p = document.createElement('p'); p.className = 'empty-copy'; p.textContent = 'No saved corrections.'; $('review-history').append(p); }
  for (const {entry, applicable, record} of entries.slice().reverse()) {
    const card = document.createElement('article'); card.className = 'history-entry';
    const title = document.createElement('strong'); title.textContent = `#${entry.revision} · ${entry.reviewer}`;
    const status = document.createElement('p'); status.className = applicable ? '' : 'inactive';
    status.textContent = applicable ? `${record.episode_id} · ${entry.action === 'restore_generated' ? 'Restored generated text' : 'Caption correction'}` : 'Historical only · generated record is no longer reviewable';
    const content = document.createElement('p'); content.textContent = entry.reviewed_caption ?? 'Generated label restored.';
    const time = document.createElement('p'); time.textContent = new Date(entry.reviewed_utc).toLocaleString();
    card.append(title, status, content, time); $('review-history').append(card);
  }
}

function renderTrajectory() {
  const t = episode()?.trajectory;
  const present = Boolean(t && t.schema_version === 'bir_trajectory_v1' && Array.isArray(t.actions) && t.actions.length);
  show('trajectory-data', present); show('trajectory-empty', !present);
  badge('trajectory-badge', present ? 'Recorded actions' : 'Unavailable', present ? 'blue' : '');
  if (!present) { text('trajectory-summary', 'Video-only input can be annotated; training export needs a trajectory.'); return; }
  text('trajectory-summary', `${t.actions.length} recorded transitions · ${t.action_spec?.representation || 'See source action specification'}`);
  $('transition-slider').max = String(t.actions.length - 1); $('transition-slider').value = '0';
  renderTransition(0);
}
function renderTransition(index) {
  const t = episode()?.trajectory;
  if (!t?.actions?.length) return;
  index = Math.max(0, Math.min(t.actions.length - 1, index));
  $('transition-slider').value = String(index);
  text('transition-index', `${index + 1} / ${t.actions.length}`);
  text('transition-time', seconds(t.action_times_s?.[index]));
  text('transition-reward', number(t.environment_rewards?.[index]));
  const terminated = t.terminated?.[index], truncated = t.truncated?.[index];
  let ending = terminated === true ? 'Terminated' : truncated === true ? 'Time-limit truncation' : terminated === false && truncated === false ? 'Not terminal' : 'End type not recorded';
  if (index === t.actions.length - 1 && !terminated && !truncated) ending = `Recording end · ${terminated === null || truncated === null ? 'type unknown' : 'see source boundary'}`;
  text('transition-boundary', ending);
  $('action-values').replaceChildren();
  const values = Array.isArray(t.actions[index]) ? t.actions[index] : [t.actions[index]];
  values.forEach((value, axis) => {
    const cell = document.createElement('div'); cell.className = 'action-cell';
    const label = document.createElement('span'); label.textContent = `action[${axis}]`;
    const output = document.createElement('strong'); output.textContent = number(value); cell.append(label, output); $('action-values').append(cell);
  });
  const obs = {}, next = {};
  for (const [key, values] of Object.entries(t.observations || {})) obs[key] = values[index];
  for (const [key, values] of Object.entries(t.next_observations || {})) next[key] = values[index];
  text('trajectory-details', JSON.stringify({source:t.source, action_spec:t.action_spec, clock_mappings:t.clock_mappings, observation_time_s:t.observation_times_s?.[index], action_time_s:t.action_times_s?.[index], next_observation_time_s:t.next_observation_times_s?.[index], source_bootstrap_stop:t.source_bootstrap_stop?.[index], boundary:t.boundary, observations:obs, next_observations:next, environment_metadata:t.environment_metadata}, null, 2));
}
function trajectoryAtVideoTime(time) {
  const t = episode()?.trajectory, camera = mainMedia()?.camera;
  const mapping = t?.clock_mappings?.find(x => x.camera === camera);
  if (!mapping || !(mapping.scale > 0) || !Array.isArray(t.action_times_s)) return;
  const target = (time - mapping.offset_s) / mapping.scale;
  let index = 0;
  while (index + 1 < t.action_times_s.length && t.action_times_s[index + 1] <= target) index++;
  renderTransition(index);
}

function renderOutcome() {
  const outcome = episode()?.outcome;
  show('outcome-values', false); $('outcome-values').replaceChildren();
  show('outcome-details-wrap', Boolean(outcome));
  if (!outcome) {
    badge('outcome-badge', 'Unavailable');
    text('outcome-copy', 'No native FLORA Rewarder result is attached. Missing scores are not zero. The task description and environment reward are not FLORA predictions.');
    return;
  }
  const state = outcomeState(outcome);
  const complete = state === 'native_attached';
  const fixture = state === 'test_fixture';
  badge('outcome-badge', fixture ? 'Test fixture' : complete ? 'Native · attached' : state, fixture ? 'warn' : complete ? 'blue' : 'error');
  text('outcome-copy', fixture ? 'Synthetic test artifact. This is not a native FLORA prediction and cannot establish model execution.' : complete ? 'Attached terminal video judgment. Inspect its raw values and provenance. A derived training reward must use an explicit mapping; it is not an instantaneous causal signal.' : 'Scoring did not produce a valid complete outcome. This stays failed or unknown and must not become a zero reward.');
  text('outcome-details', JSON.stringify(outcome, null, 2));
}

async function saveCorrection(restore = false) {
  const row = selected(), current = bundle; if (!row || !current) return;
  $('save-correction').disabled = true;
  try {
    const updated = await appendReview(current, row, restore ? null : $('reviewed-caption').value, $('reviewer').value);
    if (current !== bundle) throw Error('The open bundle changed before the correction was saved. Reopen the intended record.');
    bundle = updated;
    dirtyText = false; unsavedBundle = true;
    const stored = persistDraft(); tabOnlyDraft = !stored;
    renderIntervals(); renderSelected(); renderHistory();
    text('save-state', stored ? 'Saved on this device. Download the bundle to keep or transfer this history.' : 'Saved in this open tab only. Browser storage is unavailable; download the bundle now.');
    $('save-state').className = stored ? 'save-state' : 'save-state error';
    show('recovery', false); pendingDraft = null;
  } catch (error) { text('save-state', error.message); $('save-state').className = 'save-state error'; }
  finally { $('save-correction').disabled = !reviewable(selected()); }
}

$('load-example').addEventListener('click', openExample);
$('bundle-file').addEventListener('change', async event => {
  const file = event.target.files[0]; if (!file) return;
  const intent = ++bundleIntent;
  try { if (file.size > MAX_BUNDLE_BYTES) throw Error('Choose a bundle of at most 64 MiB.'); await openBundle(await file.text(), {intent}); }
  catch (error) { notice(error.message, true); }
  finally { event.target.value = ''; }
});
$('media-file').addEventListener('change', async event => {
  const files = [...event.target.files]; if (!files.length) return;
  if (!bundle) { notice('Open a bundle before selecting its media.', true); event.target.value = ''; return; }
  const expected = new Set(payload.episodes.flatMap(e => e.media).map(m => m.source_sha256));
  let matched = 0, rejected = 0;
  try {
    for (const file of files) {
      if (file.size > MAX_MEDIA_BYTES) { rejected++; continue; }
      const hash = await sha256(await file.arrayBuffer());
      if (!expected.has(hash)) { rejected++; continue; }
      const prior = localMedia.get(hash); if (prior) URL.revokeObjectURL(prior.url);
      localMedia.set(hash, {url:URL.createObjectURL(file), origin:'local', filename:file.name}); matched++;
    }
    notice(`${matched} matching local video(s) verified. ${rejected ? rejected + ' unmatched or oversized file(s) were not used. ' : ''}No files were uploaded.`, Boolean(rejected));
    await loadVideo();
  } catch (error) { notice(error.message, true); }
  finally { event.target.value = ''; }
});
$('episode-picker').addEventListener('change', event => {
  if (!beforeSwitch()) { event.target.value = String(episodeIndex); return; }
  episodeIndex = Number(event.target.value); selectedIndex = 0; renderEpisode();
});
$('play-interval').addEventListener('click', () => {
  const row = selected(); if (!Number.isFinite(row?.start_s) || !video.getAttribute('src')) return;
  seek(row.start_s); playingInterval = true; video.play().catch(error => { notice(error.message, true); playingInterval = false; });
});
$('previous-interval').addEventListener('click', () => selectInterval(selectedIndex - 1, true));
$('next-interval').addEventListener('click', () => selectInterval(selectedIndex + 1, true));
video.addEventListener('timeupdate', () => {
  text('video-time', seconds(video.currentTime));
  const row = selected();
  if (playingInterval && row && video.currentTime >= row.end_s) { video.pause(); playingInterval = false; }
  document.querySelectorAll('.timeline-segment').forEach(item => {
    const record = episode()?.records[Number(item.dataset.index)];
    item.classList.toggle('playing', record && video.currentTime >= record.start_s && video.currentTime < record.end_s);
  });
  trajectoryAtVideoTime(video.currentTime);
});
video.addEventListener('error', () => { if (video.getAttribute('src')) notice('This browser could not play the source video. Its bytes were not changed. Use a compatible browser or local player.', true); });
$('transition-slider').addEventListener('input', event => {
  const i = Number(event.target.value), t = episode()?.trajectory;
  renderTransition(i);
  const mapping = t?.clock_mappings?.find(x => x.camera === mainMedia()?.camera);
  if (mapping && mapping.scale > 0) { playingInterval = false; video.pause(); seek(t.action_times_s[i] * mapping.scale + mapping.offset_s); }
});
$('reviewed-caption').addEventListener('input', () => { dirtyText = true; text('save-state', 'Unsaved text. Select Save correction to record it.'); $('save-state').className = 'save-state unsaved'; });
$('save-correction').addEventListener('click', () => saveCorrection(false));
$('restore-generated').addEventListener('click', () => saveCorrection(true));
$('download-bundle').addEventListener('click', () => {
  if (!bundle) return;
  if (dirtyText) { text('save-state', 'Save the correction before downloading. The text is not yet in the review history.'); $('save-state').className = 'save-state unsaved'; return; }
  const blob = new Blob([exportBundle(bundle)], {type:'application/json'}), url = URL.createObjectURL(blob);
  const link = document.createElement('a'); link.href = url;
  link.download = `${safeName(episode()?.episode_id || 'rollout')}-review-r${bundle.reviews.length}.bir.json`;
  document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 30000);
  unsavedBundle = false; tabOnlyDraft = false;
  text('save-state', 'Bundle download requested. Keep it with the matching source media; the full review history is included.');
});
$('restore-draft').addEventListener('click', async () => {
  if (!pendingDraft || !beforeSwitch()) return;
  const capturedBundle = bundle, candidate = pendingDraft;
  $('restore-draft').disabled = true;
  try {
    const restored = await restoreStoredDraft(candidate, capturedBundle, () => bundle);
    if (bundle !== capturedBundle || pendingDraft !== candidate) return;
    bundle = restored; pendingDraft = null; show('recovery', false); unsavedBundle = true;
    renderIntervals(); renderSelected(); renderHistory();
    text('save-state', 'Local draft restored for this exact payload. Download the bundle to keep this version.');
  } catch (error) {
    if (bundle === capturedBundle && pendingDraft === candidate) {
      pendingDraft = null; show('recovery', false); notice(error.message, true);
    }
  } finally { $('restore-draft').disabled = false; }
});
$('keep-imported').addEventListener('click', () => { pendingDraft = null; show('recovery', false); notice('Imported review history retained. The local draft has not been deleted.'); });
window.addEventListener('beforeunload', event => { if (dirtyText || unsavedBundle) { event.preventDefault(); event.returnValue = ''; } });

async function start() {
  const startupIntent = bundleIntent;
  try {
    if (!globalThis.crypto?.subtle) throw Error('This app needs HTTPS or localhost for source checksum verification.');
    const index = JSON.parse(await (await limitedFetch(new URL('./examples/index.json', location.href), 1024 * 1024)).text());
    if (!Array.isArray(index.examples) || !index.examples.length || index.examples.length > 20) throw Error('No public example is configured. Import a compatible local bundle to begin.');
    examples = index.examples;
    $('example-picker').replaceChildren(...examples.map((entry, i) => { const option = document.createElement('option'); option.value = String(i); option.textContent = entry.title || entry.id; return option; }));
    $('example-picker').disabled = false; $('load-example').disabled = false;
    if (startupIntent === bundleIntent && !bundle) await openExample();
  } catch (error) { notice(error.message, true); $('workspace').setAttribute('aria-busy', 'false'); }
}
start();
