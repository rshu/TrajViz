// Fixed evaluator: failed assertions are grades; crashes are ungraded.
const core = require('./core.js');
const checks = [
  ['lower_bound', () => core.clamp(-2, 0, 10) === 0],
  ['upper_bound', () => core.clamp(12, 0, 10) === 10],
  ['inside_interval', () => core.clamp(5, 0, 10) === 5],
  ['addition', () => core.add(2, 3) === 5],
];
const tests = checks.map(([id, check]) => ({ id, passed: check() }));
console.log(JSON.stringify({ tests, resolved: tests.every(t => t.passed) }));
