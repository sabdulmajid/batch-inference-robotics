/* Read canonical JSON spans without rewriting numeric lexemes.
   Python may serialize 0.0 where JavaScript serializes 0. Hash original bytes. */
export function scanCanonicalJSON(source) {
  let position = 0;
  const fail = () => { throw Error('Bundle payload must use canonical JSON from the CLI; unsupported spelling, duplicate keys, or key order.'); };
  const codepointOrder = (a, b) => {
    const left = Array.from(a), right = Array.from(b);
    for (let i = 0; i < Math.min(left.length, right.length); i++) {
      const delta = left[i].codePointAt(0) - right[i].codePointAt(0);
      if (delta) return delta;
    }
    return left.length - right.length;
  };
  function string() {
    const start = position++;
    while (position < source.length) {
      const char = source[position++];
      if (char === '\\') { position++; continue; }
      if (char === '"') {
        const raw = source.slice(start, position);
        let value;
        try { value = JSON.parse(raw); } catch { fail(); }
        if (JSON.stringify(value) !== raw) fail();
        return value;
      }
    }
    fail();
  }
  function keep(path) {
    return path.length === 0 || (path[0] === 'episodes' && (
      path.length <= 2 ||
      (path[2] === 'records' && path.length <= 4) ||
      (['trajectory', 'outcome'].includes(path[2]) && path.length === 3) ||
      (path[2] === 'trajectory' && path.length === 4 && ['observation_times_s', 'next_observation_times_s'].includes(path[3])) ||
      (path[2] === 'outcome' && path[3] === 'request' && (path.length === 4 || (path[4] === 'config' && path.length === 5)))
    ));
  }
  function value(path, depth) {
    if (depth > 32) fail();
    const start = position, char = source[position], retained = keep(path);
    let members, items;
    if (char === '{') {
      position++; if (retained) members = [];
      let previous = null;
      if (source[position] !== '}') while (true) {
        if (source[position] !== '"') fail();
        const keyStart = position, key = string();
        if (previous !== null && codepointOrder(previous, key) >= 0) fail();
        previous = key;
        if (source[position++] !== ':') fail();
        const child = value([...path, key], depth + 1);
        if (retained) members.push({key, keyStart, node:child});
        if (source[position] !== ',') break;
        position++;
      }
      if (source[position++] !== '}') fail();
    } else if (char === '[') {
      position++; if (retained) items = [];
      let index = 0;
      if (source[position] !== ']') while (true) {
        const child = value([...path, index++], depth + 1);
        if (retained) items.push(child);
        if (source[position] !== ',') break;
        position++;
      }
      if (source[position++] !== ']') fail();
    } else if (char === '"') string();
    else {
      const match = /^(?:null|true|false|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)/.exec(source.slice(position));
      if (!match) fail();
      position += match[0].length;
    }
    return {start, end:position, members, items};
  }
  const root = value([], 0);
  if (position !== source.length) fail();
  return root;
}

export const member = (node, key) => node?.members?.find(item => item.key === key)?.node;

export function rawObject(source, node, excluded = [], ascii = false) {
  if (!node?.members) throw Error('Expected a canonical object for an identity check.');
  let value = '{' + node.members.filter(item => !excluded.includes(item.key))
    .map(item => source.slice(item.keyStart, item.node.end)).join(',') + '}';
  // The independent Rewarder hashes JSON with Python ensure_ascii=True.
  if (ascii) value = value.replace(/[\u007f-\uffff]/g, char => '\\u' + char.charCodeAt(0).toString(16).padStart(4, '0'));
  return value;
}
