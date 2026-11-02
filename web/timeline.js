/* Playback clocks can round a requested seek by a few microseconds. This
   tolerance affects displayed transition selection only, never source clocks. */
export const PLAYBACK_CLOCK_TOLERANCE_S = 0.0001;

export function transitionIndexAtVideoTime(actionTimes, mapping, videoTime) {
  if (!Array.isArray(actionTimes) || !actionTimes.length || !Number.isFinite(videoTime) ||
      !Number.isFinite(mapping?.scale) || mapping.scale <= 0 || !Number.isFinite(mapping?.offset_s)) return null;
  const target = (videoTime + PLAYBACK_CLOCK_TOLERANCE_S - mapping.offset_s) / mapping.scale;
  let low = 0, high = actionTimes.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (actionTimes[middle] <= target) low = middle + 1;
    else high = middle;
  }
  return Math.max(0, low - 1);
}
