const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const indicatorSource = fs.readFileSync(path.join(root, 'app/static/js/fps.js'), 'utf8');
const context = { window: {} };
vm.runInNewContext(indicatorSource, context);
const indicator = context.window.DetectionFpsIndicator;

test('calculates processing FPS from measured processing duration', () => {
  const element = { textContent: '' };
  indicator.update(element, 250);
  assert.equal(element.textContent, 'FPS: 4.0');
});

test('shows zero FPS for invalid or non-positive processing durations', () => {
  for (const duration of [0, -1, Number.NaN, Number.POSITIVE_INFINITY, undefined]) {
    const element = { textContent: '' };
    indicator.update(element, duration);
    assert.equal(element.textContent, 'FPS: 0');
  }
});

test('reset displays zero FPS', () => {
  const element = { textContent: 'FPS: 8.0' };
  indicator.reset(element);
  assert.equal(element.textContent, 'FPS: 0');
});

test('Live Detection wires the indicator to WebSocket results and stop events', () => {
  const app = fs.readFileSync(path.join(root, 'app/static/js/app.js'), 'utf8');
  const html = fs.readFileSync(path.join(root, 'app/templates/index.html'), 'utf8');

  assert.match(html, /id="liveFps"[^>]*>FPS: 0/);
  assert.match(html, /static\/js\/fps\.js/);
  assert.match(app, /DetectionFpsIndicator\.update\(fpsIndicator, payload\.inference_ms\)/);
  assert.match(app, /function stopCamera\([\s\S]*?DetectionFpsIndicator\.reset\(fpsIndicator\)/);
  assert.match(app, /ws\.onclose = \(event\) => \{[\s\S]*?DetectionFpsIndicator\.reset\(fpsIndicator\)/);
});
