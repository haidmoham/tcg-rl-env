import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';
import { resolve, relative, isAbsolute } from 'node:path';
import { fileURLToPath } from 'node:url';

export const projectRoot = fileURLToPath(new URL('../../', import.meta.url));
export const twinleafPin = JSON.parse(readFileSync(new URL('./pin.json', import.meta.url), 'utf8'));

export function verifyPublicSource(sourceRoot = resolve(projectRoot, '.cache/twinleaf')) {
  const git = (...args) => execFileSync('git', ['-C', sourceRoot, ...args], { encoding: 'utf8' }).trim();
  if (git('rev-parse', 'HEAD') !== twinleafPin.commit) throw new Error('TWINLEAF_PIN_MISMATCH');
  if (git('remote', 'get-url', 'origin') !== twinleafPin.repository) throw new Error('TWINLEAF_ORIGIN_MISMATCH');
  if (git('status', '--porcelain', '--untracked-files=all')) throw new Error('TWINLEAF_SOURCE_DIRTY');
  return resolve(sourceRoot, 'ptcg-server/src');
}

export function assertStandardAdmission() {
  throw new Error('UNVERIFIED_STANDARD_COVERAGE: card presence is not rules conformance or legal-action completeness');
}

export function loadPublicRules() {
  const root = verifyPublicSource();
  const require = createRequire(import.meta.url);
  const tooling = resolve(projectRoot, '.cache/twinleaf-tooling/node_modules');
  const ts = require(resolve(tooling, 'typescript'));
  if (ts.version !== twinleafPin.typescript) throw new Error('TYPESCRIPT_PIN_MISMATCH');
  const pako = require(resolve(tooling, '@progress/pako-esm/package.json'));
  if (pako.version !== twinleafPin.pako) throw new Error('PAKO_PIN_MISMATCH');
  const Module = require('node:module');
  const originalResolve = Module._resolveFilename;
  const originalExtension = Module._extensions['.ts'];
  Module._resolveFilename = function(request, parent, ...rest) {
    if (request === '@progress/pako-esm') request = resolve(tooling, '@progress/pako-esm');
    return originalResolve.call(this, request, parent, ...rest);
  };
  Module._extensions['.ts'] = (module, filename) => {
    const pathFromRoot = relative(root, filename);
    if (pathFromRoot.startsWith('..') || isAbsolute(pathFromRoot)) throw new Error('UNPINNED_TYPESCRIPT_SOURCE');
    module._compile(ts.transpileModule(readFileSync(filename, 'utf8'), {
      compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2019 }
    }).outputText, filename);
  };
  return {
    load(path) {
      const file = resolve(root, path);
      const pathFromRoot = relative(root, file);
      if (pathFromRoot.startsWith('..') || isAbsolute(pathFromRoot)) throw new Error('INVALID_RULES_PATH');
      return require(file);
    },
    close() {
      Module._resolveFilename = originalResolve;
      if (originalExtension) Module._extensions['.ts'] = originalExtension;
      else delete Module._extensions['.ts'];
    }
  };
}
