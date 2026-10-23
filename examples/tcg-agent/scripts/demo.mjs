import { mkdirSync, writeFileSync, readFileSync } from 'node:fs';
import { Environment, replay } from '../src/environment.mjs';
import { greedyPolicy, randomPolicy, runEpisode } from '../src/policies.mjs';

if (process.argv[2] === '--replay') {
  const record = JSON.parse(readFileSync(process.argv[3], 'utf8'));
  console.log(JSON.stringify({ replayVerified: true, ...replay(record).outcome }, null, 2));
} else {
  const env = new Environment({ seed: 20261019 });
  const result = runEpisode(env, [greedyPolicy, randomPolicy(429)]);
  mkdirSync('runs', { recursive: true });
  writeFileSync('runs/demo-replay.json', JSON.stringify(env.replayRecord(), null, 2));
  console.log(JSON.stringify({ ...result, replay: 'runs/demo-replay.json', profile: 'basic-duel-v1', competitiveStrength: 'not evaluated' }, null, 2));
}
