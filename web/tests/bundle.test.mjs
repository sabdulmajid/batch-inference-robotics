import {test} from 'node:test';
import assert from 'node:assert/strict';
import {webcrypto} from 'node:crypto';
import {readFile} from 'node:fs/promises';
if (!globalThis.crypto) Object.defineProperty(globalThis, 'crypto', {value:webcrypto});
import {parseBundle, sha256, appendReview, exportBundle, historyState, validateReviews, reviewable} from '../bundle.js';

const jsonCanonical = v => v === null || typeof v !== 'object' ? JSON.stringify(v) : Array.isArray(v) ? '['+v.map(jsonCanonical).join(',')+']' : '{'+Object.keys(v).sort().map(k=>JSON.stringify(k)+':'+jsonCanonical(v[k])).join(',')+'}';
const source = 'a'.repeat(64);
let recordId;
const record = () => ({schema_version:'bir_label_v1', episode_id:'demo_0', record_id:recordId,
  request_id:'c'.repeat(64),source_sha256:{main:source},profile:'human',interval_index:0,
  start_s:0,end_s:1,generated_caption:'Pick up the object.',inference_status:'complete',
  failure:null,generation:{precomputed:true}});
{const {record_id, ...body} = record(); recordId = await sha256(jsonCanonical(body));}
const payload = () => ({schema_version:'bir_rollout_payload_v1', episodes:[{episode_id:'demo_0',
  task:'Lift the object.',media:[{camera:'main',filename:'demo.mp4',source_sha256:source,duration_s:2}],
  records:[record()],trajectory:null,outcome:null}],provenance:{fixture:true}});
async function wrap(value = payload()) {
  for (const e of value.episodes) {
    for (const r of e.records) {const {record_id,...body}=r;r.record_id=await sha256(jsonCanonical(body));}
    if(e.trajectory) {const {trajectory_id,...body}=e.trajectory;e.trajectory.trajectory_id=await sha256(jsonCanonical(body));}
  }
  const payload_json = jsonCanonical(value);
  return {schema_version:'bir_rollout_bundle_v1',payload_json,payload_sha256:await sha256(payload_json),reviews:[]};
}
const parse = value => parseBundle(JSON.stringify(value));

test('a reviewed bundle round trip keeps exact prediction bytes and full history', async () => {
  const original = await wrap();
  const corrected = await appendReview(original, record(), 'Lift the red object.', 'reviewer-1');
  const restored = await appendReview(corrected, record(), null, 'reviewer-2');
  const reopened = await parseBundle(exportBundle(restored));
  assert.equal(reopened.bundle.payload_json, original.payload_json);
  assert.equal(reopened.bundle.payload_sha256, original.payload_sha256);
  assert.equal(reopened.bundle.reviews.length, 2);
  assert.equal(reopened.bundle.reviews[0].reviewed_caption, 'Lift the red object.');
  assert.equal(reopened.bundle.reviews[1].action, 'restore_generated');
  assert.equal(original.reviews.length, 0);
});

test('Python-style float spelling in immutable payload survives browser round trip', async () => {
  const original = await wrap();
  original.payload_json = original.payload_json.replace('"start_s":0', '"start_s":0.0');
  const {scanCanonicalJSON, member, rawObject}=await import('../raw-json.js');
  const tree=scanCanonicalJSON(original.payload_json);
  const row=member(member(tree,'episodes').items[0],'records').items[0];
  const newId=await sha256(rawObject(original.payload_json,row,['record_id']));
  original.payload_json=original.payload_json.replace(recordId,newId);
  original.payload_sha256 = await sha256(original.payload_json);
  const {bundle} = await parse(original);
  const edited = await appendReview(bundle, record(), 'Correction', 'x');
  assert.equal((await parseBundle(exportBundle(edited))).bundle.payload_json, original.payload_json);
});

test('changed prediction payload fails its checksum', async () => {
  const original = await wrap(); original.payload_json = original.payload_json.replace('Pick up', 'Drop');
  await assert.rejects(parse(original), /checksum failed/);
});

test('tampered correction and broken chain fail verification', async () => {
  const edited = await appendReview(await wrap(), record(), 'Correction', 'x');
  const changed = structuredClone(edited); changed.reviews[0].reviewed_caption = 'Another caption';
  await assert.rejects(parse(changed), /checksum failed/);
  const broken = structuredClone(edited); broken.reviews[0].previous_entry_sha256 = source;
  await assert.rejects(parse(broken), /chain changed/);
});

test('orphaned historical review remains visible and does not bind to new output', async () => {
  const edited = await appendReview(await wrap(), record(), 'Correction', 'x');
  const changed = payload(); changed.episodes[0].records[0].generated_caption = 'A changed generated caption.';
  const reopened = await parse({...await wrap(changed),reviews:edited.reviews});
  const history = historyState(reopened.payload, reopened.bundle.reviews);
  assert.equal(history.length, 1); assert.equal(history[0].applicable, false);
  assert.equal(history[0].record, null);
});

test('pending, failed and missing captions cannot receive new corrections', async () => {
  for (const change of [{inference_status:'pending'}, {inference_status:'failed'}, {generated_caption:null}]) {
    const row = {...record(),...change}; assert.equal(reviewable(row), false);
    await assert.rejects(appendReview(await wrap(),row,'Correction','x'), /completed generated captions/);
  }
});

test('invalid interval bounds and source mismatches are rejected', async () => {
  for (const change of [{end_s:-1}, {end_s:10}, {source_sha256:{main:'e'.repeat(64)}}]) {
    const changed = payload(); Object.assign(changed.episodes[0].records[0], change);
    await assert.rejects(parse(await wrap(changed)), /clock|source/i);
  }
});

test('duplicate records and episode IDs are rejected', async () => {
  const p1=payload();p1.episodes[0].records.push(record());
  await assert.rejects(parse(await wrap(p1)), /Duplicate/);
  const p2=payload();p2.episodes.push(structuredClone(p2.episodes[0]));
  await assert.rejects(parse(await wrap(p2)), /Duplicate/);
});

test('unknown and failed outcome evidence stays raw without a zero score', async () => {
  const p=payload();p.episodes[0].outcome=await failedOutcome();
  const parsed=await parse(await wrap(p));assert.deepEqual(parsed.payload.episodes[0].outcome,p.episodes[0].outcome);
  assert.equal(parsed.payload.episodes[0].outcome.score,null);
});

test('new review events reject empty identity and empty text', async () => {
  await assert.rejects(appendReview(await wrap(),record(),'Correction','  '), /reviewer/);
  await assert.rejects(appendReview(await wrap(),record(),'   ','x'), /nonempty/);
});

test('journal revision numbers are not silently coerced', async () => {
  const b=await appendReview(await wrap(),record(),'Correction','x');b.reviews[0].revision='1';
  await assert.rejects(validateReviews(b.reviews), /order/);
});

test('trajectory observations, actions and clocks must agree', async () => {
  const p=payload();p.episodes[0].trajectory=await trajectory(2);
  await parse(await wrap(p));
  p.episodes[0].trajectory.action_times_s[1]=.3;
  await assert.rejects(parse(await wrap(p)), /Action clocks/);
});

test('fixture outcome cannot be presented as native evidence', async () => {
  const {outcomeState}=await import('../bundle.js');
  const common={schema_version:'bir_rewarder_result_v1',status:'complete',parse_valid:true};
  assert.equal(outcomeState({...common,execution:{kind:'test_fixture'}}),'test_fixture');
  assert.equal(outcomeState({...common,execution:{kind:'not_executed'}}),'unknown');
  assert.equal(outcomeState({...common,execution:{kind:'native'}}),'native_attached');
  assert.equal(outcomeState({...common,parse_valid:false,execution:{kind:'native'}}),'unknown');
  assert.equal(outcomeState({status:'failed',score:null}),'failed');
  assert.equal(outcomeState(null),'unavailable');
});

test('review text with unpaired Unicode surrogates is rejected before journal creation', async () => {
  await assert.rejects(appendReview(await wrap(),record(),'bad\ud800','x'), /Unicode/);
});

test('unsupported outer fields cannot survive into a portable bundle', async () => {
  await assert.rejects(parse({...await wrap(),remote_media_url:'https://invalid.example/video'}), /bir_rollout_bundle_v1/);
});

async function failedOutcome() {
  const request={schema_version:'bir_rewarder_request_v1',episode_id:'demo_0',task:'Lift the object.',whole_episode:true,camera_order:['main'],clock_offsets_s:[0],sources:[{camera:'main',sha256:source,size_bytes:10}],config:JSON.parse(await readFile(new URL('../../configs/rewarder-terminal.json',import.meta.url),'utf8'))};
  const result={schema_version:'bir_rewarder_result_v1',request,request_id:await sha256(jsonCanonical(request)),status:'failed',parse_valid:false,score:null,expected_score:null,progress:null,execution:{kind:'not_executed'},error:{message:'Invalid parse'}};
  result.result_digest=await sha256(jsonCanonical(result));return result;
}

test('changed generated content cannot reuse a reviewed identity even with a new payload hash', async () => {
  const edited=await appendReview(await wrap(),record(),'A correction','x');
  edited.payload_json=edited.payload_json.replace('Pick up the object.','Drop the object.');
  edited.payload_sha256=await sha256(edited.payload_json);
  await assert.rejects(parse(edited), /Generated record identity changed/);
});

test('changed trajectory data cannot retain the prior trajectory identity', async () => {
  const p=payload();p.episodes[0].records=[];
  p.episodes[0].trajectory=await trajectory(1);
  const b=await wrap(p);b.payload_json=b.payload_json.replace('"actions":[[0.1]]','"actions":[[0.2]]');b.payload_sha256=await sha256(b.payload_json);
  await assert.rejects(parse(b), /Trajectory content identity changed/);
});

test('changed outcome and mismatched task are rejected', async () => {
  const p=payload();p.episodes[0].outcome=await failedOutcome();
  const b=await wrap(p);b.payload_json=b.payload_json.replace('Invalid parse','Other failure');b.payload_sha256=await sha256(b.payload_json);
  await assert.rejects(parse(b), /Outcome artifact checksum changed/);
  const changed=payload();changed.episodes[0].outcome=await failedOutcome();changed.episodes[0].task='Different task';
  await assert.rejects(parse(await wrap(changed)), /Outcome task or episode/);
});

test('unsupported JSON key order is rejected instead of reserializing source floats', async () => {
  const b=await wrap();b.payload_json=JSON.stringify(JSON.parse(b.payload_json),null,2);b.payload_sha256=await sha256(b.payload_json);
  await assert.rejects(parse(b), /canonical JSON/);
});

test('Python-generated Unicode and floating-point identities validate without rewriting', async () => {
  const {outcomeState}=await import('../bundle.js');
  const file=await readFile(new URL('./fixtures/python-canonical.bir.json',import.meta.url),'utf8');
  const {bundle,payload}=await parseBundle(file);
  assert.match(bundle.payload_json,/"start_s":0\.0/);
  assert.equal(outcomeState(payload.episodes[0].outcome),'test_fixture');
  const corrected=await appendReview(bundle,payload.episodes[0].records[0],'Check the café object.','cross-language');
  assert.equal((await parseBundle(exportBundle(corrected))).bundle.payload_json,bundle.payload_json);
});

async function trajectory(n=2) {
  const times=Array.from({length:n},(_,i)=>i/10), next=times.map((_,i)=>(i+1)/10);
  const t={schema_version:'bir_trajectory_v1',episode_id:'demo_0',source:{dataset:'synthetic-test-only',revision:'fixture-v1',sha256:'d'.repeat(64),demo_id:'demo_0'},
    actions:times.map(()=>[.1]),observations:{state:times.map((_,i)=>[i])},next_observations:{state:times.map((_,i)=>[i+1])},
    observation_times_s:times,action_times_s:times,next_observation_times_s:next,
    terminated:times.map(()=>null),truncated:times.map(()=>null),environment_rewards:times.map(()=>0),source_bootstrap_stop:times.map(()=>0),
    boundary:{kind:'recording_end',source_mask_semantics:'Synthetic fixture; no physical terminal inference'},
    action_spec:{representation:'Synthetic normalized action',units:['normalized command'],controller:{type:'test_fixture'},bounds:[-1,1]},
    environment_metadata:{env_name:'SyntheticFixture',env_version:'test',env_kwargs:{}},
    clock_mappings:[{camera:'main',scale:1,offset_s:0,source:'Synthetic recorded clock',video_sha256:source}]};
  t.transition_ids=await Promise.all(times.map((time,i)=>sha256(jsonCanonical([t.source.sha256,t.source.demo_id,i,time,next[i]]))));
  return t;
}

test('a claimed trajectory must supply the complete supported data contract', async () => {
  for (const key of ['source','transition_ids','action_spec','environment_metadata','boundary','observations','next_observations','terminated','truncated']) {
    const p=payload();p.episodes[0].trajectory=await trajectory();delete p.episodes[0].trajectory[key];
    await assert.rejects(parse(await wrap(p)), undefined, `missing ${key} must fail`);
  }
});

test('transition IDs bind the exact source, original clock and next-observation clock', async () => {
  const p=payload();p.episodes[0].trajectory=await trajectory();p.episodes[0].trajectory.transition_ids[0]='e'.repeat(64);
  await assert.rejects(parse(await wrap(p)), /Transition identity differs/);
});

test('controller bounds, per-axis units and lossy source dtypes are rejected', async () => {
  for (const change of [t=>{t.action_spec.bounds=[0,.05];},t=>{t.action_spec.units=[];},t=>{t.array_dtypes={actions:'<i4'};},t=>{t.array_dtypes={actions:'<f4'};},t=>{t.array_dtypes={unknown:'<f8'};}]) {
    const p=payload();p.episodes[0].trajectory=await trajectory();change(p.episodes[0].trajectory);
    await assert.rejects(parse(await wrap(p)), /bounds|units|dtype/);
  }
});
