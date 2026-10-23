import { spawn } from 'node:child_process';
import { isAbsolute, parse, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { isDeepStrictEqual } from 'node:util';
import { cardActionFeatureContract, cardActionContractSha256, encodeCardActionDecision,
  encodeCardChoiceDecision, indexedCardActionDecision, validatePrintingAliases } from '../../learning/card-action-features.mjs';

export class CardActionModelClient {
  constructor({ executable, checkpoint, modelRoot, seed = 1729, device = 'cpu', printingAliases = {} }) {
    if (!isAbsolute(executable ?? '') || !isAbsolute(checkpoint ?? '') || !['cpu', 'cuda'].includes(device)
        || !Number.isSafeInteger(seed) || seed < 0 || seed > 0xffffffff) throw new Error('explicit local card/action model configuration required');
    this.printingAliases = structuredClone(validatePrintingAliases(printingAliases));
    if (modelRoot !== undefined && (typeof modelRoot !== 'string' || !isAbsolute(modelRoot)
        || resolve(modelRoot) === parse(resolve(modelRoot)).root)) throw new Error('absolute dedicated modelRoot required');
    const childEnv = { ...process.env, ...(modelRoot === undefined ? {} : { PTCG_AGENT_MODEL_ROOT: resolve(modelRoot) }) };
    this.checkpoint = resolve(checkpoint); this.device = device;
    this.pending = new Map(); this.buffer = ''; this.closed = false; this.sequence = 0;
    this.requests = 0; this.inferenceMilliseconds = 0;
    this.child = spawn(executable, [fileURLToPath(new URL('../../learning/card_action_torch.py', import.meta.url)),
      '--checkpoint', checkpoint, '--seed', String(seed), '--device', device],
    { windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'], env: childEnv });
    this.exited = new Promise(accept => { this.child.once('exit', accept); this.child.once('error', accept); });
    this.child.stdout.setEncoding('utf8');
    this.child.stdout.on('data', chunk => this.read(chunk));
    this.child.stderr.on('data', () => {});
    this.child.once('exit', () => this.close(new Error('card/action model stopped')));
    this.child.once('error', () => this.close(new Error('card/action model unavailable')));
    this.child.stdin.on('error', () => this.close(new Error('card/action model input closed')));
  }

  request(op, payload = {}) {
    if (this.closed) return Promise.reject(new Error('card/action model closed'));
    const id = ++this.sequence;
    return new Promise((accept, reject) => {
      const timer = setTimeout(() => this.close(new Error('card/action model reply timed out')), 30000);
      this.pending.set(id, { accept, reject, timer });
      this.child.stdin.write(JSON.stringify({ ...payload, op, id }) + '\n');
    });
  }

  read(chunk) {
    this.buffer += chunk;
    if (Buffer.byteLength(this.buffer) > 4 * 1024 * 1024) return this.close(new Error('card/action response exceeded budget'));
    let end;
    while ((end = this.buffer.indexOf('\n')) >= 0) {
      const line = this.buffer.slice(0, end); this.buffer = this.buffer.slice(end + 1);
      let response;
      try { response = JSON.parse(line); } catch { return this.close(new Error('invalid card/action model response')); }
      if (!response || typeof response !== 'object' || Array.isArray(response) || !Number.isSafeInteger(response.id)
          || typeof response.ok !== 'boolean' || !this.pending.has(response.id)) return this.close(new Error('uncorrelated card/action response'));
      const pending = this.pending.get(response.id); this.pending.delete(response.id); clearTimeout(pending.timer);
      if (response.ok) pending.accept(response); else pending.reject(new Error(response.error ?? 'card/action request rejected'));
    }
  }

  async verify() {
    const info = await this.request('ping');
    if (info.schema !== cardActionFeatureContract.schema || info.stateDim !== 224 || info.candidateDim !== 191
        || info.contractSha256 !== cardActionContractSha256 || resolve(info.checkpoint ?? '') !== this.checkpoint
        || info.device !== this.device || !isDeepStrictEqual(info.printingAliases, this.printingAliases)) {
      this.close(); throw new Error('card/action checkpoint feature or alias contract mismatch');
    }
    return info;
  }

  async decide({ view }) {
    const started = performance.now();
    const result = await this.request('infer', encodeCardActionDecision(view, null, { printingAliases: this.printingAliases }));
    this.requests++; this.inferenceMilliseconds += performance.now() - started;
    return indexedCardActionDecision(view, result.index);
  }

  async rankChoices({ view, prompt = view.prompts[0] }) {
    const started = performance.now();
    const result = await this.request('infer', encodeCardChoiceDecision(view, prompt, { printingAliases: this.printingAliases }));
    this.requests++; this.inferenceMilliseconds += performance.now() - started;
    if (!Array.isArray(result.ranking) || result.ranking.length !== prompt.choices.length
        || new Set(result.ranking).size !== prompt.choices.length
        || result.ranking.some(index => !Number.isSafeInteger(index) || index < 0 || index >= prompt.choices.length))
      throw new Error('model choice ranking must cover the complete current prompt');
    return result.ranking;
  }

  close(error = new Error('card/action model closed')) {
    if (this.closed) return;
    this.closed = true;
    for (const pending of this.pending.values()) { clearTimeout(pending.timer); pending.reject(error); }
    this.pending.clear();
    if (this.child.exitCode === null) this.child.kill();
  }

  async shutdown() {
    if (!this.closed) { try { await this.request('close'); } catch {} }
    this.close(); await this.exited;
  }
}
