import { execFileSync, execSync } from 'node:child_process';
import { mkdirSync, existsSync, copyFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { projectRoot, twinleafPin, verifyPublicSource } from '../integrations/twinleaf/public-engine.mjs';

const source = resolve(projectRoot, '.cache/twinleaf');
const tooling = resolve(projectRoot, '.cache/twinleaf-tooling');
mkdirSync(resolve(projectRoot, '.cache'), { recursive: true });
if (!existsSync(source)) {
  execFileSync('git', ['clone', '--depth=1', '--no-checkout', '--filter=blob:none', '--sparse', twinleafPin.repository, source], { stdio: 'inherit' });
  const fetchedHead = execFileSync('git', ['-C', source, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
  if (fetchedHead !== twinleafPin.commit) {
    execFileSync('git', ['-C', source, 'fetch', '--depth=1', 'origin', twinleafPin.commit], { stdio: 'inherit' });
  }
  execFileSync('git', ['-C', source, 'sparse-checkout', 'set', 'ptcg-server/src'], { stdio: 'inherit' });
  execFileSync('git', ['-C', source, 'checkout', '--detach', twinleafPin.commit], { stdio: 'inherit' });
}
verifyPublicSource(source);
mkdirSync(tooling, { recursive: true });
copyFileSync(resolve(projectRoot, 'integrations/twinleaf/tooling-package.json'), resolve(tooling, 'package.json'));
copyFileSync(resolve(projectRoot, 'integrations/twinleaf/tooling-package-lock.json'), resolve(tooling, 'package-lock.json'));
if (process.platform === 'win32') {
  execSync('npm.cmd ci --ignore-scripts --no-audit --no-fund', { cwd: tooling, stdio: 'inherit' });
} else {
  execFileSync('npm', ['ci', '--ignore-scripts', '--no-audit', '--no-fund'], { cwd: tooling, stdio: 'inherit' });
}
console.log(`Prepared public Twinleaf ${twinleafPin.commit}. No server, database, or client started.`);
