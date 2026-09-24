const { test } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { Predictor, normalizeInput } = require('../dist-electron/predictor');

test('与现有模型的 Python ONNX Runtime 预测一致', async () => {
  const model = new Predictor();
  await model.load(path.resolve(__dirname, '../../线性回归/models/lr_torch_1f.onnx'));
  // 来自 machine_learning 环境中相同 ONNX 文件的 float32 输出。
  const expected = [[0, 5.020588397979736], [1, 7.299826622009277], [3, 11.85830307006836], [10, 27.81296730041504], [14.4, 37.84161376953125]];
  for (const [distance, value] of expected) {
    const { fare } = await model.predict({ distance, unit: 'mi' });
    assert.ok(Math.abs(fare - value) <= 1e-5 + Math.abs(value) * 1e-5);
  }
  const km = await model.predict({ distance: 1.609344, unit: 'km' });
  const mi = await model.predict({ distance: 1, unit: 'mi' });
  assert.equal(km.fare, mi.fare);
  await assert.rejects(model.predict({ distance: 3e38, unit: 'mi' }), /数值范围/);
});

test('拒绝非法参数，允许零', () => {
  for (const input of [null, {}, { distance: '', unit: 'mi' }, { distance: -1, unit: 'mi' },
    { distance: Infinity, unit: 'mi' }, { distance: NaN, unit: 'mi' },
    { distance: 1e40, unit: 'mi' }, { distance: 1, unit: 'bad' }]) {
    assert.throws(() => normalizeInput(input));
  }
  assert.equal(normalizeInput({ distance: 0, unit: 'mi' }), 0);
});

test('模型缺失及尚未加载时返回错误', async () => {
  const model = new Predictor();
  await assert.rejects(model.load(path.join(__dirname, 'missing.onnx')));
  await assert.rejects(model.predict({ distance: 1, unit: 'mi' }), /准备好/);
});
