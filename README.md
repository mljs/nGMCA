# nGMCA - Non-negative Generalized Morphological Component Analysis
<p align="center">
  A tool for Non-negative matrix factorization.
</p>

## Instalation
`$ npm install ml-ngmca `

## [API Documentation](https://mljs.github.io/nGMCA/)

This algorithm is based on the article [Jérémy Rapin, Jérôme Bobin, Anthony Larue, Jean-Luc Starck. Sparse and Non-negative BSS for Noisy Data, IEEE Transactions on Signal Processing, 2013.IEEE Transactions on Signal Processing, vol. 61, issue 22, p. 5620-5632, 2013.](https://arxiv.org/pdf/1308.5546.pdf)

In order to get a general idea of the problem you could also check the [Wikipedia article](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization).

## Usage

```js
import { Matrix } from 'ml-matrix';
import { ngmca } from 'ml-ngmca';

let A = new Matrix([
  [1, 2, 3],
  [4, 5, 6],
]);
let S = new Matrix([
  [1, 2],
  [3, 4],
  [5, 6],
]);

let v = A.mmul(S);
const options = {
  maximumIteration: 200,
  phaseRatio: 0.4,
};
const result = ngmca(v, 2, options);
// result has properties A and S, the estimated matrices
```
## License
  [MIT](./LICENSE)
