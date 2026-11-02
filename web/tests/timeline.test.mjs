import {test} from 'node:test';
import assert from 'node:assert/strict';
import {transitionIndexAtVideoTime, PLAYBACK_CLOCK_TOLERANCE_S} from '../timeline.js';

const times=Array.from({length:59},(_,i)=>i/20);
const mapping={scale:1,offset_s:0};

test('rounded final-frame seek keeps the selected 59th recorded transition', () => {
  const original=times.slice();
  assert.equal(transitionIndexAtVideoTime(times,mapping,2.899999),58);
  assert.equal(transitionIndexAtVideoTime(times,mapping,2.9),58);
  assert.deepEqual(times,original);
});

test('sub-millisecond tolerance does not advance a genuinely earlier playback time', () => {
  assert(PLAYBACK_CLOCK_TOLERANCE_S<0.001);
  assert.equal(transitionIndexAtVideoTime(times,mapping,2.899),57);
  assert.equal(transitionIndexAtVideoTime(times,mapping,2.875),57);
  assert.equal(transitionIndexAtVideoTime(times,mapping,0),0);
});

test('tolerance stays in video seconds when the trajectory clock has scale and offset', () => {
  const shifted={scale:0.5,offset_s:10};
  const end=times.at(-1)*shifted.scale+shifted.offset_s;
  assert.equal(transitionIndexAtVideoTime(times,shifted,end-0.000001),58);
  assert.equal(transitionIndexAtVideoTime(times,shifted,end-0.001),57);
});

test('unavailable or invalid playback mappings do not select a transition', () => {
  assert.equal(transitionIndexAtVideoTime([],mapping,0),null);
  assert.equal(transitionIndexAtVideoTime(times,{scale:0,offset_s:0},1),null);
  assert.equal(transitionIndexAtVideoTime(times,mapping,NaN),null);
});
