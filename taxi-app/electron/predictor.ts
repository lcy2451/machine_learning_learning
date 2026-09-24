import * as ort from 'onnxruntime-node';

export type Prediction = { fare: number; miles: number };

// 即使界面已经检查，主进程仍需验证来自 IPC 的数据。
export function normalizeInput(input: unknown): number {
  if (!input || typeof input !== 'object') throw new Error('请输入有效里程。');
  const { distance, unit } = input as { distance?: unknown; unit?: unknown };
  if (typeof distance !== 'number' || !Number.isFinite(distance) || distance < 0)
    throw new Error('里程必须是大于或等于零的有限数字。');
  if (unit !== 'mi' && unit !== 'km') throw new Error('请选择英里或公里。');
  const miles = unit === 'km' ? distance / 1.609344 : distance;
  if (!Number.isFinite(Math.fround(miles))) throw new Error('里程太大，请输入较小的数值。');
  return miles;
}

export class Predictor {
  private session?: ort.InferenceSession;

  async load(path: string): Promise<void> {
    this.session = await ort.InferenceSession.create(path, { executionProviders: ['cpu'] });
    if (!this.session.inputNames.includes('TRIP_MILES') || !this.session.outputNames.includes('FARE'))
      throw new Error('模型输入输出与应用不匹配。');
  }

  async predict(input: unknown): Promise<Prediction> {
    const miles = normalizeInput(input);
    if (!this.session) throw new Error('模型尚未准备好。');
    // 一个张量包含一条行程、一个特征，与 Python 导出时的 [1, 1] 一致。
    const tensor = new ort.Tensor('float32', new Float32Array([miles]), [1, 1]);
    const outputs = await this.session.run({ TRIP_MILES: tensor });
    const fare = Number(outputs.FARE?.data[0]);
    if (!Number.isFinite(fare)) throw new Error('预测结果超出数值范围，请减小里程。');
    return { fare, miles };
  }
}
