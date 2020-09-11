# nGMCA - Non-negative Generalized Morphological Component Analysis
<p align="center">
  A tool for Non-negative matrix factorization.
</p>

## Instalation
`$ npm install ml-ngmca `

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
