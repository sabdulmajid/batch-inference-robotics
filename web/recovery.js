/* Recovery journals always bind to the payload and history observed before await. */
import {validateReviews} from './bundle.js';

const head = reviews => reviews.at(-1)?.entry_sha256 ?? null;
function baseMatches(bundle, pending) {
  return bundle?.payload_sha256 === pending.payload_sha256 &&
    bundle.reviews.length === pending.base_revision && head(bundle.reviews) === pending.base_head;
}
function extendsHistory(bundle, reviews) {
  return reviews.length > bundle.reviews.length &&
    bundle.reviews.every((entry, i) => entry.entry_sha256 === reviews[i]?.entry_sha256);
}

export async function inspectStoredDraft(raw, capturedBundle, getCurrentBundle, validate = validateReviews) {
  const payloadHash = capturedBundle.payload_sha256;
  const draft = JSON.parse(raw);
  if (draft.schema_version !== 'bir_browser_draft_v1' || draft.payload_sha256 !== payloadHash) {
    throw Error('Local draft identity changed.');
  }
  await validate(draft.reviews);
  if (getCurrentBundle() !== capturedBundle || getCurrentBundle()?.payload_sha256 !== payloadHash) return null;
  if (JSON.stringify(draft.reviews) === JSON.stringify(capturedBundle.reviews)) return {kind:'identical'};
  if (!extendsHistory(capturedBundle, draft.reviews)) return {kind:'conflict'};
  return {
    kind:'available', payload_sha256:payloadHash,
    base_revision:capturedBundle.reviews.length, base_head:head(capturedBundle.reviews),
    reviews:draft.reviews,
  };
}

export async function restoreStoredDraft(pending, capturedBundle, getCurrentBundle, validate = validateReviews) {
  if (!pending || pending.kind !== 'available' || !baseMatches(capturedBundle, pending)) {
    throw Error('The open payload or review history changed. The local draft was not applied.');
  }
  await validate(pending.reviews);
  if (getCurrentBundle() !== capturedBundle || !baseMatches(getCurrentBundle(), pending) ||
      !extendsHistory(capturedBundle, pending.reviews)) {
    throw Error('The open payload or review history changed. The local draft was not applied.');
  }
  return {...capturedBundle, reviews:pending.reviews};
}
