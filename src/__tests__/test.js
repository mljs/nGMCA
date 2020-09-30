import { toBeDeepCloseTo } from 'jest-matcher-deep-close-to';
import { Matrix } from 'ml-matrix';

import { ngmca } from '../ngmca';

expect.extend({ toBeDeepCloseTo });

describe('NMF test', () => {
  it('use case 1', async () => {
    let w = new Matrix([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    let h = new Matrix([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);

    let v = w.mmul(h);

    const options = {
      maximumIteration: 100,
      phaseRatio: 0.4,
    };

    const result = ngmca(v, 2, options);
    const w0 = result.A;
    const h0 = result.S;
    expect(w0.mmul(h0).to2DArray()).toBeDeepCloseTo(v.to2DArray(), 1);
  });
});
