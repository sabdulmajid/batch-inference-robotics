import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {parseBundle, appendReview, validateReviews} from '../bundle.js';
import {inspectStoredDraft, restoreStoredDraft} from '../recovery.js';

async function fixture() {
  const {bundle,payload}=await parseBundle(await readFile(new URL('./fixtures/python-canonical.bir.json',import.meta.url),'utf8'));
  const reviewed=await appendReview(bundle,payload.episodes[0].records[0],'Checked correction.','recovery-test');
  const raw=JSON.stringify({schema_version:'bir_browser_draft_v1',payload_sha256:bundle.payload_sha256,reviews:reviewed.reviews});
  return {bundle,reviewed,raw,record:payload.episodes[0].records[0]};
}
function pausedValidation() {
  let release;
  const pause=new Promise(resolve=>{release=resolve;});
  return {release,validate:async reviews=>{await pause;await validateReviews(reviews);}};
}

test('opening another payload while recovery is being verified cannot publish the old draft', async () => {
  const {bundle,raw}=await fixture();let current=bundle;
  const barrier=pausedValidation();
  const result=inspectStoredDraft(raw,bundle,()=>current,barrier.validate);
  current={...bundle,payload_sha256:'f'.repeat(64)};
  barrier.release();
  assert.equal(await result,null);
  assert.equal(current.reviews.length,0);
});

test('a verified draft restores only its original payload and unchanged base history', async () => {
  const {bundle,raw,record}=await fixture();let current=bundle;
  const pending=await inspectStoredDraft(raw,bundle,()=>current);
  assert.equal(pending.payload_sha256,bundle.payload_sha256);
  const restored=await restoreStoredDraft(pending,bundle,()=>current);
  assert.equal(restored.payload_json,bundle.payload_json);
  assert.equal(restored.reviews.length,1);
  await assert.rejects(restoreStoredDraft(pending,{...bundle,payload_sha256:'e'.repeat(64)},()=>current),/changed/);
  current=await appendReview(bundle,record,'A different correction.','other-reviewer');
  await assert.rejects(restoreStoredDraft(pending,current,()=>current),/changed/);
});

test('a switch during Restore validation cannot attach a draft to the newly opened bundle', async () => {
  const {bundle,raw}=await fixture();let current=bundle;
  const pending=await inspectStoredDraft(raw,bundle,()=>current);
  const barrier=pausedValidation();
  const result=restoreStoredDraft(pending,bundle,()=>current,barrier.validate);
  current={...bundle,payload_sha256:'d'.repeat(64)};
  barrier.release();
  await assert.rejects(result,/changed/);
  assert.equal(current.reviews.length,0);
});

test('competing histories are not offered as a recoverable extension', async () => {
  const {bundle,raw,record}=await fixture();
  const current=await appendReview(bundle,record,'A competing correction.','other');
  assert.deepEqual(await inspectStoredDraft(raw,current,()=>current),{kind:'conflict'});
});
