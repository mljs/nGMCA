import { Matrix } from 'ml-matrix';

import {
  checkMatrixS,
  initialize,
  normalize,
  updateLambda,
  updateMatrixA,
  updateMatrixS,
} from './stages';

export function ngmca(originalMatrix, rank, options = {}) {
  const {
    maximumIteration = 500,
    maxFBIteration = 80,
    maxInitFBIteration = 50,
    toleranceFBInit = 0,
    toleranceFB = 0.00001,
    phaseRatio = 0.8,
    randGenerator = Math.random,
    tauMAD = 1,
    useTranspose = false,
  } = options;

  let { normConstrained = false } = options;
  originalMatrix = Matrix.checkMatrix(originalMatrix);
  if (useTranspose) originalMatrix = originalMatrix.transpose();
  let refinementBeginning = Math.floor(phaseRatio * maximumIteration);

  let data = initialize(originalMatrix, {
    rank,
    randGenerator,
    maxInitFBIteration,
    toleranceFBInit,
    maxFBIteration,
    toleranceFB,
  });

  data = normalize(data, { normOnA: true });
  data.lambda = data.A.transpose()
    .mmul(data.A.mmul(data.S).sub(originalMatrix))
    .abs()
    .max();

  for (let iter = 0; iter < maximumIteration; iter++) {
    data.iteration = iter;
    data.S = updateMatrixS(
      data.A,
      data.S,
      originalMatrix,
      data.lambda,
      options,
    );
    data = checkMatrixS(data, originalMatrix);
    data = normalize(data, { normOnA: false });

    if (iter > refinementBeginning) normConstrained = true;

    data.A = updateMatrixA(data.A, data.S, originalMatrix, {
      maxFBIteration,
      toleranceFB,
      normConstrained,
      lambda: 0,
    });

    data = normalize(data, { normOnA: true });

    data.lambda = updateLambda(data, originalMatrix, {
      refinementBeginning,
      tauMAD,
    });
  }

  if (useTranspose) {
    let temp = data.A.transpose();
    data.A = data.S.transpose();
    data.S = temp;
  }
  return data;
}
