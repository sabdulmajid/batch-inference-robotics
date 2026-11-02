import {scanCanonicalJSON, member, rawObject} from './raw-json.js';
/* Portable bundle primitives. No network or browser storage access. */
export const MAX_BUNDLE_BYTES = 64 * 1024 * 1024;
export const MAX_MEDIA_BYTES = 256 * 1024 * 1024;
const HEX = /^[a-f0-9]{64}$/;
const keysAre = (value, keys) => Object.keys(value).sort().join('\0') === keys.slice().sort().join('\0');
const validUnicode = value => !/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u.test(value);
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const assert = (test, message) => { if (!test) throw new Error(message); };

export async function sha256(value) {
  const bytes = typeof value === 'string' ? new TextEncoder().encode(value) : value;
  const result = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(result), x => x.toString(16).padStart(2, '0')).join('');
}

export function canonical(value) {
  if (typeof value === 'string') assert(validUnicode(value), 'Text contains an invalid Unicode character.');
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return JSON.stringify(value);
  if (typeof value === 'number') {
    assert(Number.isSafeInteger(value), 'Review journal numbers must be safe integers.');
    return String(value);
  }
  if (Array.isArray(value)) return '[' + value.map(canonical).join(',') + ']';
  assert(object(value), 'Invalid review journal value.');
  return '{' + Object.keys(value).sort().map(k => JSON.stringify(k) + ':' + canonical(value[k])).join(',') + '}';
}

function finiteTree(value, depth = 0) {
  assert(depth < 32, 'The bundle is nested too deeply.');
  if (typeof value === 'string') assert(validUnicode(value), 'Text contains an invalid Unicode character.');
  else if (typeof value === 'number') assert(Number.isFinite(value), 'The bundle contains a non-finite number.');
  else if (Array.isArray(value)) for (const item of value) finiteTree(item, depth + 1);
  else if (object(value)) for (const item of Object.values(value)) finiteTree(item, depth + 1);
}

async function validateTrajectory(t, episode, raw, rawNode) {
  if (t === null) return;
  assert(object(t) && t.schema_version === 'bir_trajectory_v1' && t.episode_id === episode.episode_id, 'Trajectory identity differs from its episode.');
  assert(object(t.source) && ['dataset','revision','demo_id'].every(key => typeof t.source[key] === 'string' && t.source[key].length > 0 && t.source[key].length <= 512) && HEX.test(t.source.sha256), 'Trajectory needs its original dataset, revision, source hash, and demonstration identity.');
  assert(object(t.environment_metadata) && typeof t.environment_metadata.env_name === 'string' && t.environment_metadata.env_name.length > 0 && typeof t.environment_metadata.env_version === 'string' && t.environment_metadata.env_version.length > 0 && object(t.environment_metadata.env_kwargs), 'Trajectory environment metadata is required.');
  assert(object(t.boundary) && typeof t.boundary.kind === 'string' && t.boundary.kind.length > 0, 'Trajectory episode boundary metadata is required.');
  assert(Array.isArray(t.actions) && t.actions.length > 0 && t.actions.length <= 10000, 'Trajectory actions must contain recorded transitions.');
  const n = t.actions.length;
  const dims = t.actions[0]?.length;
  assert(Number.isSafeInteger(dims) && dims > 0 && dims <= 1024 && t.actions.every(a => Array.isArray(a) && a.length === dims && a.every(x => typeof x === 'number')), 'Recorded action shapes are inconsistent.');
  assert(object(t.action_spec) && typeof t.action_spec.representation === 'string' && t.action_spec.representation.length > 0 && Array.isArray(t.action_spec.units) && t.action_spec.units.length === dims && t.action_spec.units.every(unit => typeof unit === 'string' && unit.length > 0) && object(t.action_spec.controller), 'Action representation, per-axis units, and controller metadata are required.');
  const bounds = t.action_spec.bounds;
  assert(Array.isArray(bounds) && bounds.length === 2 && bounds.every(x => typeof x === 'number') && bounds[0] < bounds[1] && t.actions.every(row => row.every(x => x >= bounds[0] - 1e-6 && x <= bounds[1] + 1e-6)), 'Actions must fit their explicit controller bounds.');
  assert(object(t.observations) && object(t.next_observations) && Object.keys(t.observations).length > 0 && Object.keys(t.observations).length <= 32, 'Recorded current and next observations are required.');
  assert(Object.keys(t.observations).sort().join('\0') === Object.keys(t.next_observations).sort().join('\0'), 'Current and next observation keys differ.');
  for (const name of Object.keys(t.observations)) {
    assert(/^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$/.test(name) && !['caption','reward','flora_reward'].includes(name), 'Unsupported policy observation key.');
    const current = t.observations[name], next = t.next_observations[name], width = current?.[0]?.length;
    assert(Array.isArray(current) && current.length === n && Array.isArray(next) && next.length === n, 'Observation counts differ from the recorded transitions.');
    assert(Number.isSafeInteger(width) && width > 0 && width <= 1024 && [...current, ...next].every(row => Array.isArray(row) && row.length === width && row.every(x => typeof x === 'number')), 'Observation shapes are inconsistent.');
  }
  for (const name of ['observation_times_s', 'action_times_s', 'next_observation_times_s']) {
    assert(Array.isArray(t[name]) && t[name].length === n && t[name].every((v, i) => typeof v === 'number' && (i === 0 || v > t[name][i - 1])), 'Trajectory clocks must be recorded, increasing, and aligned to transition count.');
  }
  assert(t.action_times_s.every((v, i) => t.observation_times_s[i] <= v && v < t.next_observation_times_s[i]), 'Action clocks must fall between current and next observations.');
  assert(t.next_observation_times_s.slice(0, -1).every((v, i) => Math.abs(v - t.observation_times_s[i + 1]) <= 1e-7), 'Disconnected transitions must be split into separate episodes.');
  for (const name of ['terminated', 'truncated']) {
    assert(Array.isArray(t[name]) && t[name].length === n && t[name].every(v => v === null || typeof v === 'boolean'), 'End flags must be explicit booleans or unknown.');
    assert(!t[name].slice(0, -1).includes(true), 'An episode cannot continue after an end flag.');
  }
  assert(t.terminated.every((v, i) => !(v === true && t.truncated[i] === true)), 'A transition cannot be both terminated and truncated.');
  assert(t.environment_rewards === null || (Array.isArray(t.environment_rewards) && t.environment_rewards.length === n && t.environment_rewards.every(v => typeof v === 'number')), 'Environment reward count differs from transitions.');
  const stop = t.source_bootstrap_stop ?? null;
  assert(stop === null || (Array.isArray(stop) && stop.length === n && stop.every(x => x === 0 || x === 1 || typeof x === 'boolean')), 'Invalid source bootstrap mask.');
  assert(stop === null || (typeof t.boundary.source_mask_semantics === 'string' && t.boundary.source_mask_semantics.length > 0), 'Source bootstrap masks need explicit semantics.');
  assert(Array.isArray(t.transition_ids) && t.transition_ids.length === n && t.transition_ids.every(x => HEX.test(x)) && new Set(t.transition_ids).size === n, 'Original transition identities are required and must be unique.');
  const rawTimes = member(rawNode, 'observation_times_s').items, rawNextTimes = member(rawNode, 'next_observation_times_s').items;
  for (let i = 0; i < n; i++) {
    const originalTime = raw.slice(rawTimes[i].start, rawTimes[i].end), nextTime = raw.slice(rawNextTimes[i].start, rawNextTimes[i].end);
    const tuple = `[${JSON.stringify(t.source.sha256)},${JSON.stringify(t.source.demo_id)},${i},${originalTime},${nextTime}]`;
    assert(await sha256(tuple) === t.transition_ids[i], 'Transition identity differs from its original source and clock.');
  }
  if (t.simulator_states !== undefined) {
    const width = t.simulator_states?.[0]?.length;
    assert(Array.isArray(t.simulator_states) && t.simulator_states.length === n && Number.isInteger(width) && width > 0 && width <= 1024 && t.simulator_states.every(row => Array.isArray(row) && row.length === width && row.every(x => typeof x === 'number')), 'Recorded simulator state shapes differ.');
  }
  const attributes = t.source_attributes ?? {};
  assert(object(attributes) && Object.keys(attributes).every(k => ['model_file','ep_meta','camera_info'].includes(k)) && Object.values(attributes).every(v => typeof v === 'string' && v.length <= 2 * 1024 * 1024), 'Unsupported source attributes.');
  const values = {actions:t.actions, rewards:t.environment_rewards, dones:stop, states:t.simulator_states ?? null};
  for (const [group, data] of [['obs', t.observations], ['next_obs', t.next_observations]]) for (const [key, value] of Object.entries(data)) values[`${group}/${key}`] = value;
  const dtypes = t.array_dtypes ?? {};
  assert(object(dtypes) && Object.keys(dtypes).every(key => Object.hasOwn(values, key)), 'Unknown source array dtype.');
  for (const [key, value] of Object.entries(values)) if (value !== null) validateDtype(value, dtypes[key] ?? '<f8', key);
  assert(Array.isArray(t.clock_mappings) && t.clock_mappings.length === episode.media.length, 'An explicit video clock mapping is required.');
  const cameras = new Set();
  for (const mapping of t.clock_mappings) {
    assert(object(mapping) && !cameras.has(mapping.camera) && episode.media.some(m => m.camera === mapping.camera) && typeof mapping.scale === 'number' && mapping.scale > 0 && typeof mapping.offset_s === 'number' && typeof mapping.source === 'string' && mapping.source.length > 0, 'Invalid or duplicate trajectory-to-video clock mapping.');
    const media = episode.media.find(m => m.camera === mapping.camera);
    assert(mapping.video_sha256 === media.source_sha256, 'Trajectory clock source differs from its video.');
    assert(t.observation_times_s[0] * mapping.scale + mapping.offset_s >= -1e-7 && t.next_observation_times_s.at(-1) * mapping.scale + mapping.offset_s <= media.duration_s + 0.1, 'Trajectory clock extends outside its video.');
    cameras.add(mapping.camera);
  }
}

function validateDtype(values, dtype, name) {
  const match = typeof dtype === 'string' && /^[<>=|]?([fiub])([1248])$/.exec(dtype);
  assert(match, `${name}: unsupported numeric dtype.`);
  const kind = match[1], size = Number(match[2]);
  assert((kind !== 'f' || [4,8].includes(size)) && (kind !== 'b' || size === 1), `${name}: unsupported numeric dtype.`);
  const visit = value => {
    if (Array.isArray(value)) { value.forEach(visit); return; }
    assert(typeof value === 'number' && Number.isFinite(value), `${name}: dtype requires finite numeric values.`);
    assert(!Number.isInteger(value) || Number.isSafeInteger(value), `${name}: this integer exceeds browser verification range; use CLI validation.`);
    if (kind === 'f') {
      assert(size === 8 || Math.fround(value) === value, `${name}: dtype would change a recorded value.`);
    } else {
      const low = kind === 'i' ? -(2 ** (size * 8 - 1)) : 0;
      const high = kind === 'b' ? 1 : kind === 'u' ? 2 ** (size * 8) - 1 : 2 ** (size * 8 - 1) - 1;
      assert(Number.isInteger(value) && value >= low && value <= high, `${name}: dtype would change a recorded value.`);
    }
  };
  visit(values);
}

export async function validateReviews(reviews) {
  assert(Array.isArray(reviews) && reviews.length <= 10000, 'Invalid or oversized review history.');
  let previous = null;
  for (let index = 0; index < reviews.length; index++) {
    const entry = reviews[index];
    assert(object(entry) && entry.schema_version === 'bir_review_v1', 'Unsupported review event.');
    assert(entry.revision === index + 1 && entry.previous_entry_sha256 === previous, 'Review history order or chain changed.');
    assert(HEX.test(entry.record_id) && entry.generated_record_sha256 === entry.record_id, 'Review record identity is invalid.');
    assert(typeof entry.reviewer === 'string' && entry.reviewer.trim().length > 0 && entry.reviewer.length <= 120, 'Review identifier is invalid.');
    assert(typeof entry.reviewed_utc === 'string' && Number.isFinite(Date.parse(entry.reviewed_utc)), 'Review time is invalid.');
    assert(entry.reviewed_caption === null || (typeof entry.reviewed_caption === 'string' && entry.reviewed_caption.trim().length > 0 && entry.reviewed_caption.length <= 10000), 'Review text is invalid.');
    assert(entry.action === (entry.reviewed_caption === null ? 'restore_generated' : 'caption_correction'), 'Review action and text disagree.');
    const {entry_sha256, ...body} = entry;
    assert(HEX.test(entry_sha256) && await sha256(canonical(body)) === entry_sha256, 'Review journal checksum failed.');
    previous = entry_sha256;
  }
}

export async function parseBundle(text) {
  assert(typeof text === 'string' && new TextEncoder().encode(text).length <= MAX_BUNDLE_BYTES, 'Choose a bundle of at most 64 MiB.');
  let bundle;
  try { bundle = JSON.parse(text); } catch { throw new Error('The selected file is not valid JSON.'); }
  assert(object(bundle) && bundle.schema_version === 'bir_rollout_bundle_v1' && keysAre(bundle, ['schema_version','payload_json','payload_sha256','reviews']), 'Expected a bir_rollout_bundle_v1 file. Export a compatible bundle with the CLI.');
  assert(typeof bundle.payload_json === 'string' && HEX.test(bundle.payload_sha256), 'The bundle has no valid immutable payload.');
  assert(await sha256(bundle.payload_json) === bundle.payload_sha256, 'Bundle payload checksum failed. No data was opened or changed.');
  const rawTree = scanCanonicalJSON(bundle.payload_json);
  let payload;
  try { payload = JSON.parse(bundle.payload_json); } catch { throw new Error('The immutable payload is not valid JSON.'); }
  finiteTree(payload);
  assert(object(payload) && payload.schema_version === 'bir_rollout_payload_v1' && keysAre(payload, ['schema_version','episodes','provenance']), 'Unsupported bundle payload version.');
  assert(Array.isArray(payload.episodes) && payload.episodes.length > 0 && payload.episodes.length <= 100, 'A bundle must contain 1–100 episodes.');
  const episodes = new Set(), records = new Set();
  for (const [episodeIndex, episode] of payload.episodes.entries()) {
    const rawEpisode = member(rawTree, 'episodes').items[episodeIndex];
    assert(object(episode) && typeof episode.episode_id === 'string' && episode.episode_id.length > 0 && episode.episode_id.length <= 200 && keysAre(episode, ['episode_id','task','media','records','trajectory','outcome']), 'Invalid episode identity.');
    assert(!episodes.has(episode.episode_id), 'Duplicate episode identity.'); episodes.add(episode.episode_id);
    assert(typeof episode.task === 'string' && episode.task.length <= 2000, 'Each episode needs an explicit task string.');
    assert(Array.isArray(episode.media) && episode.media.length <= 4, 'Invalid episode media list.');
    const cameras = new Set();
    for (const media of episode.media) {
      assert(object(media) && typeof media.camera === 'string' && !cameras.has(media.camera) && keysAre(media, ['camera','filename','source_sha256','duration_s']), 'Invalid or duplicate camera identity.'); cameras.add(media.camera);
      assert(HEX.test(media.source_sha256) && typeof media.filename === 'string' && media.filename.length > 0 && !/[\/\\]/.test(media.filename) && !['.','..'].includes(media.filename), 'Media needs a source hash and filename.');
      assert(typeof media.duration_s === 'number' && media.duration_s > 0 && media.duration_s <= 3600, 'Media duration must be positive.');
    }
    assert(Array.isArray(episode.records) && episode.records.length <= 10000, 'Invalid annotation records.');
    for (const [recordIndex, row] of episode.records.entries()) {
      const rawRecord = member(rawEpisode, 'records').items[recordIndex];
      assert(object(row) && row.schema_version === 'bir_label_v1' && row.episode_id === episode.episode_id, 'Annotation does not belong to its episode.');
      assert(HEX.test(row.record_id) && !records.has(row.record_id), 'Duplicate or invalid annotation record identity.'); records.add(row.record_id);
      assert(await sha256(rawObject(bundle.payload_json, rawRecord, ['record_id'])) === row.record_id, 'Generated record identity changed. Historical corrections were not applied.');
      const boundSources = Object.entries(row.source_sha256 || {}).sort();
      const actualSources = episode.media.map(m => [m.camera, m.source_sha256]).sort();
      assert(JSON.stringify(boundSources) === JSON.stringify(actualSources), 'Annotation sources differ from the attached cameras.');
      assert(typeof row.inference_status === 'string' && (row.generated_caption === null || typeof row.generated_caption === 'string'), 'Invalid caption state.');
      if (row.start_s !== null || row.end_s !== null) {
        assert(typeof row.start_s === 'number' && typeof row.end_s === 'number' && row.start_s >= 0 && row.end_s > row.start_s, 'Caption interval has invalid clocks.');
        assert(episode.media.length > 0 && episode.media.every(m => row.end_s <= m.duration_s + 0.1), 'Caption clock or source differs from its video.');
      }
    }
    await validateTrajectory(episode.trajectory, episode, bundle.payload_json, member(rawEpisode, 'trajectory'));
    if (episode.trajectory) {
      assert(HEX.test(episode.trajectory.trajectory_id) && await sha256(rawObject(bundle.payload_json, member(rawEpisode, 'trajectory'), ['trajectory_id'])) === episode.trajectory.trajectory_id, 'Trajectory content identity changed.');
    }
    assert(episode.outcome === null || object(episode.outcome), 'Invalid outcome attachment.');
    if (episode.outcome) await validateOutcome(episode.outcome, episode, bundle.payload_json, member(rawEpisode, 'outcome'));
  }
  await validateReviews(bundle.reviews);
  return {bundle, payload};
}

export function reviewable(record) {
  return Boolean(record && record.inference_status === 'complete' && typeof record.generated_caption === 'string' && record.generated_caption.trim() && Number.isFinite(record.start_s) && Number.isFinite(record.end_s));
}

export function latestReview(reviews, recordId) {
  for (let i = reviews.length - 1; i >= 0; i--) if (reviews[i].record_id === recordId) return reviews[i];
  return null;
}

export async function appendReview(bundle, record, caption, reviewer) {
  assert(reviewable(record), 'Only completed generated captions can receive new corrections.');
  reviewer = reviewer.trim();
  assert(reviewer.length > 0 && reviewer.length <= 120, 'Enter a reviewer identifier of 1–120 characters.');
  assert(caption === null || (typeof caption === 'string' && caption.trim().length > 0 && caption.length <= 10000), 'Enter a nonempty correction of at most 10,000 characters.');
  const entry = {
    schema_version:'bir_review_v1', record_id:record.record_id,
    revision:bundle.reviews.length + 1,
    previous_entry_sha256:bundle.reviews.at(-1)?.entry_sha256 ?? null,
    generated_record_sha256:record.record_id, reviewer,
    reviewed_utc:new Date().toISOString(), reviewed_caption:caption,
    action:caption === null ? 'restore_generated' : 'caption_correction'
  };
  entry.entry_sha256 = await sha256(canonical(entry));
  // Return a new wrapper. The original prediction payload never changes.
  return {...bundle, reviews:[...bundle.reviews, entry]};
}

export function exportBundle(bundle) { return JSON.stringify(bundle, null, 2) + '\n'; }

export function historyState(payload, reviews) {
  const current = new Map(payload.episodes.flatMap(e => e.records).map(r => [r.record_id, r]));
  return reviews.map(entry => ({entry, applicable:reviewable(current.get(entry.record_id)), record:current.get(entry.record_id) ?? null}));
}

async function validateOutcome(outcome, episode, raw, node) {
  assert(outcome.schema_version === 'bir_rewarder_result_v1' && HEX.test(outcome.result_digest) &&
    await sha256(rawObject(raw, node, ['result_digest'], true)) === outcome.result_digest, 'Outcome artifact checksum changed.');
  const request = outcome.request;
  const requestNode = member(node, 'request');
  assert(object(request) && keysAre(request, ['schema_version','episode_id','task','whole_episode','camera_order','clock_offsets_s','sources','config']) && request.schema_version === 'bir_rewarder_request_v1' && HEX.test(outcome.request_id) &&
    await sha256(rawObject(raw, requestNode, [], true)) === outcome.request_id, 'Outcome request identity changed.');
  // Digest of configs/rewarder-terminal.json, Python sorted compact ASCII JSON.
  assert(await sha256(rawObject(raw, member(requestNode, 'config'), [], true)) === 'd7e8e5eba9f931ee2eefb9cee098ec9a5f4c1b7abce19efd48e95c2255376c6e', 'Unsupported outcome model or runtime configuration.');
  assert(Array.isArray(request.sources) && request.sources.every(source => object(source) && keysAre(source, ['camera','sha256','size_bytes']) && Number.isSafeInteger(source.size_bytes) && source.size_bytes > 0), 'Invalid outcome source metadata.');
  assert(request.episode_id === episode.episode_id && request.task === episode.task && request.whole_episode === true, 'Outcome task or episode differs from this rollout.');
  assert(Array.isArray(request.camera_order) && Array.isArray(request.sources) &&
    JSON.stringify(request.camera_order) === JSON.stringify(episode.media.map(m => m.camera)) &&
    JSON.stringify(request.sources.map(m => [m.camera, m.sha256])) === JSON.stringify(episode.media.map(m => [m.camera, m.source_sha256])), 'Outcome camera order or source hashes differ from this rollout.');
  assert(Array.isArray(request.clock_offsets_s) && request.clock_offsets_s.length === episode.media.length && request.clock_offsets_s.every(v => v === 0), 'Unsupported outcome camera clock offsets.');
  assert(['complete','failed','interrupted','pending','unknown','unavailable'].includes(outcome.status) && typeof outcome.parse_valid === 'boolean' &&
    ['native','test_fixture','not_executed'].includes(outcome.execution?.kind), 'Unsupported outcome execution state.');
  if (outcome.status !== 'complete') {
    assert(!outcome.parse_valid && ['score','expected_score','progress'].every(key => outcome[key] === null), 'Incomplete outcome cannot contain a valid training value.');
    return;
  }
  assert(outcome.parse_valid && outcome.execution.kind !== 'not_executed', 'Unexecuted or invalid outcome cannot be complete.');
  const r = outcome.raw_record;
  assert(object(r) && keysAre(r, ['id','task','score','expected_score','progress','probabilities','rationale','duration_s','sampled_frames','output']) && r.id === episode.episode_id && r.task === episode.task, 'Native outcome record belongs to a different task or episode.');
  assert(Number.isInteger(r.score) && r.score >= 1 && r.score <= 5 && typeof r.output === 'string' && /^Score:\s*[1-5]\s*$/.test(r.output.trim()) && Number(r.output.trim().at(-1)) === r.score && r.rationale === null, 'Invalid native outcome answer.');
  assert(object(r.probabilities) && keysAre(r.probabilities, ['1','2','3','4','5']) && Object.values(r.probabilities).every(v => typeof v === 'number' && v >= 0 && v <= 1), 'Invalid outcome probabilities.');
  assert(Math.abs(Object.values(r.probabilities).reduce((a,b) => a+b, 0)-1) <= 0.000251, 'Outcome probabilities do not sum to one.');
  const expected = Object.entries(r.probabilities).reduce((a,[k,v]) => a+Number(k)*v,0);
  assert(typeof r.expected_score === 'number' && r.expected_score >= 1 && r.expected_score <= 5 && Math.abs(expected-r.expected_score) <= 0.000801 && typeof r.progress === 'number' && r.progress >= 0 && r.progress <= 1 && Math.abs(r.progress-(r.expected_score-1)/4) <= 0.000064, 'Outcome score and probabilities disagree.');
  assert(typeof r.duration_s === 'number' && r.duration_s > 0 && Number.isInteger(r.sampled_frames) && r.sampled_frames >= 2 && r.sampled_frames <= 64 && r.sampled_frames % 2 === 0, 'Invalid outcome source duration or sampled frames.');
  assert(['score','expected_score','progress'].every(key => outcome[key] === r[key]), 'Outcome summary differs from its native record.');
}

export function outcomeState(outcome) {
  if (!outcome) return 'unavailable';
  if (outcome.execution?.kind === 'test_fixture' || outcome.synthetic === true || outcome.test_fixture === true) return 'test_fixture';
  if (outcome.schema_version === 'bir_rewarder_result_v1' && outcome.status === 'complete' && outcome.parse_valid === true && outcome.execution?.kind === 'native') return 'native_attached';
  return ['failed','interrupted','pending','unknown','unavailable'].includes(outcome.status) ? outcome.status : 'unknown';
}
