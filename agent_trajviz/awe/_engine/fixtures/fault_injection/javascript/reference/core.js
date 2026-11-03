function clamp(value, lower, upper) {
  if (value < lower) return lower;
  if (value > upper) return upper;
  return value;
}

function add(a, b) {
  return a + b;
}

module.exports = { clamp, add };
