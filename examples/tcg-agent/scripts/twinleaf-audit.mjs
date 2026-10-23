import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve, relative } from 'node:path';
import { projectRoot, twinleafPin, verifyPublicSource } from '../integrations/twinleaf/public-engine.mjs';

const root = verifyPublicSource();
const rows = [];
const walk = directory => {
  for (const entry of readdirSync(directory, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
    const path = resolve(directory, entry.name);
    if (entry.isDirectory()) { walk(path); continue; }
    if (!entry.name.endsWith('.ts') || entry.name.endsWith('.spec.ts')) continue;
    const source = readFileSync(path, 'utf8');
    const classes = [...source.matchAll(/export class\s+(\w+)\s+extends\s+(\w+)/g)];
    for (let index = 0; index < classes.length; index++) {
      const item = classes[index];
      const body = source.slice(item.index, classes[index + 1]?.index ?? source.length);
      const literal = name => {
        const match = body.match(new RegExp(`public\\s+${name}(?:\\s*:\\s*string)?\\s*=\\s*(['"])(.*?)\\1`));
        return match?.[2] ?? null;
      };
      rows.push({ path: 'ptcg-server/src/' + relative(root, path).replaceAll('\\', '/'),
        sha256: createHash('sha256').update(source).digest('hex'), className: item[1], baseClass: item[2],
        set: literal('set'), setNumber: literal('setNumber'), fullName: literal('fullName'),
        regulationMark: literal('regulationMark'), status: 'script-present-unverified' });
    }
  }
};
walk(resolve(root, 'sets'));
const report = { schemaVersion: 1, upstream: twinleafPin.repository, commit: twinleafPin.commit,
  completeness: 'Static exported-class inventory only. Inherited or computed metadata remains null. No legality or correctness inference.',
  classes: rows.length, literalCardIdentities: rows.filter(row => row.set && row.setNumber).length, rows };
const output = resolve(projectRoot, process.argv[2] ?? 'integrations/twinleaf/source-inventory.json');
writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ output, classes: report.classes, literalCardIdentities: report.literalCardIdentities }));
