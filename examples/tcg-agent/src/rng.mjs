export class Random {
  #state;
  constructor(seed) {
    if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw new TypeError('seed must be uint32');
    this.#state = seed;
  }
  next() {
    this.#state = (this.#state + 0x9e3779b9) >>> 0;
    let x = this.#state;
    x = Math.imul(x ^ (x >>> 16), 0x21f0aaad);
    x = Math.imul(x ^ (x >>> 15), 0x735a2d97);
    return ((x ^ (x >>> 15)) >>> 0) / 0x100000000;
  }
  int(n) {
    if (!Number.isSafeInteger(n) || n < 1) throw new TypeError('random bound must be positive');
    return Math.floor(this.next() * n);
  }
  shuffle(items) {
    const result = [...items];
    for (let i = result.length - 1; i > 0; i--) {
      const j = this.int(i + 1);
      [result[i], result[j]] = [result[j], result[i]];
    }
    return result;
  }
}
