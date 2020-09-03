import { Matrix, solve, EVD } from 'ml-matrix';
import median from 'median-quickselect';

const defaultOptions = {
  maximumIteration: 500,
  maximumFBIIteration: 80,
  FBRelativeDifferenceTolerance: 0.00001,
  phaseRatio: 0.8,
  tauMAD: 1,
};

export function ngmca(Y, r, options = {}) {
  options = Object.assign({}, defaultOptions, options);
  const {
    maximumIteration = 500,
    phaseRatio = 0.8,
    useTranspose = false,
  } = options;
  Y = new Matrix(Y);
  if (useTranspose) Y = Y.transpose();
  let refinementBeginning = Math.floor(phaseRatio * maximumIteration);
  let data = warmInitNMF(Y, r);
  data = normilizeA(data, 1);
  data.lambda = data.A.transpose()
    .mmul(data.A.clone().mmul(data.S).sub(Y))
    .abs()
    .max();

  for (let iter = 0; iter < maximumIteration; iter++) {
    data.iteration = iter;
    data.S = nonNegSparseUpdateS(data.A, data.S, Y, data.lambda, options);
    data = reinitializeS(data, Y, options);
    data = normilizeA(data, false);

    if (iter > refinementBeginning) options.normConstrained = true;

    data.A = nonNegSparseUpdateA(data.A, data.S, Y, 0, options);
    data = normilizeA(data, true);
    data.lambda = updateLambda(data, Y, options);
  }

  if (useTranspose) {
    let temp = data.A.transpose();
    data.A = data.S.transpose();
    data.S = temp;
  }
  return data;
}

function updateLambda(data, Y, options = {}) {
  let { phaseRatio, maximumIteration, tauMAD } = options;
  let { iteration, lambda, A, S } = data;
  let refinementBeginning = Math.floor(phaseRatio * maximumIteration);

  if (refinementBeginning <= iteration) return lambda;

  let sigmaResidue;
  if (options.lambdaInf !== undefined) {
    sigmaResidue = options.lambdaInf / options.tauMAD;
  } else if (options.addStd !== undefined) {
    sigmaResidue = options.addStd;
  } else {
    let alY = Matrix.sub(Y, A.mmul(S)).to1DArray();
    let result = dimMADstd(Matrix.from1DArray(1, alY.length, alY), 'row');
    sigmaResidue = result.get(0, 0);
  }
  let nextLambda = Math.max(
    tauMAD * sigmaResidue,
    lambda - 1 / (refinementBeginning - iteration),
  );
  return nextLambda;
}

function dimMADstd(X, dim) {
  let medians = getMedians(X, dim);
  let matrix = X.clone();
  matrix =
    dim === 'column'
      ? matrix.subRowVector(medians.to1DArray())
      : matrix.subColumnVector(medians.to1DArray());
  return Matrix.mul(getMedians(matrix.abs(), dim), 1.4826);
}

function getMedians(X, dim) {
  let medians = [];
  let rows = X.rows;
  let columns = X.columns;
  switch (dim) {
    case 'column':
      for (let i = 0; i < columns; i++) {
        medians.push(median(X.getColumn(i)));
      }
      medians = Matrix.from1DArray(1, columns, medians);
      break;
    default:
      for (let i = 0; i < rows; i++) {
        medians.push(median(X.getRow(i)));
      }
      medians = Matrix.from1DArray(rows, 1, medians);
  }
  return medians;
}

function normilizeA(data, normOnA) {
  let DS = dimNorm(data.S.transpose(), 'column');
  let DA = dimNorm(data.A, 'column');
  let D = Matrix.mul(DS, DA);
  let onS, onA;
  if (normOnA) {
    onS = (index, c) =>
      (data.S.get(index, c) * D.get(0, index)) / DS.get(0, index);
    onA = (index, r) => data.A.get(r, index) / DA.get(0, index);
  } else {
    onS = (index, c) => data.S.get(index, c) / DS.get(0, index);
    onA = (index, r) =>
      (data.A.get(r, index) * D.get(0, index)) / DA.get(0, index);
  }
  for (let index = 0; index < D.columns; index++) {
    let valueForS, valueForA;
    if (D.get(0, index) > 0) {
      valueForS = onS;
      valueForA = onA;
    } else {
      valueForA = () => 0;
      valueForS = () => 0;
    }
    for (let c = 0; c < data.S.columns; c++) {
      data.S.set(index, c, valueForS(index, c));
    }
    for (let r = 0; r < data.A.rows; r++) {
      data.A.set(r, index, valueForA(index, r));
    }
  }
  return data;
}

function warmInitNMF(Y, k) {
  let result = {};
  let r = Y.rows;
  let c = Y.columns;
  result.A = Matrix.rand(r, k, { random: Math.random });
  let options = {
    maximumFBIIteration: 50,
    FBRelativeDifferenceTolerance: 0,
  };
  for (let iter = 0; iter < 1; iter++) {
    //select columns with sum positive from A
    let sumC = result.A.sum('column');
    let positiveSumColumnA = [];
    for (let i = 0; i < sumC.length; i++) {
      if (sumC[i] > 0) positiveSumColumnA.push(result.A.getColumn(i));
    }
    positiveSumColumnA = new Matrix(positiveSumColumnA);
    //resolve the system of equation Lx = Y for x, then select just non negative values;
    let candidateS = getMax(solve(positiveSumColumnA.transpose(), Y), 0);
    //set the S matrix with dimensions k x c
    let rIndex = 0;
    result.S = Matrix.zeros(k, c);
    for (let i = 0; i < sumC.length; i++) {
      if (sumC[i] > 0) {
        let rowCandidate = candidateS.getRow(rIndex++);
        for (let j = 0; j < c; j++) {
          result.S.set(i, j, rowCandidate[j]);
        }
      }
    }
    //select rows with positive sum by row
    let sumR = result.S.sum('row');
    let positiveSumRowS = [];
    for (let i = 0; i < sumR.length; i++) {
      if (sumR[i] > 0) positiveSumRowS.push(result.S.getRow(i));
    }

    positiveSumRowS = new Matrix(positiveSumRowS);

    // solve the system of linear equation xY = S

    let candidateA = solve(positiveSumRowS.transpose(), Y.transpose());
    candidateA = getMax(candidateA.transpose(), 0);
    let cIndex = 0;

    result.A = Matrix.zeros(r, k);
    for (let i = 0; i < sumR.length; i++) {
      if (sumR[i] > 0) {
        let colCandidate = candidateA.getColumn(cIndex++);
        for (let j = 0; j < r; j++) {
          result.A.set(j, i, colCandidate[j]);
        }
      }
    }

    result.S = nonNegSparseUpdateS(result.A, result.S, Y, 0, options);
    result = reinitializeS(result, Y, 0);
    result.A = nonNegSparseUpdateA(result.A, result.S, Y, 0, options);
  }
  return result;
}

function reinitializeS(data, Y) {
  let { A, S } = data;
  //check if is there at least one element cero
  let indices = [];
  let sum = S.sum('row');

  for (let i = 0; i < sum.length; i++) {
    if (sum[i] === 0) {
      indices.push(i);
      continue;
    } else {
      for (let j = 0; j < S.columns; j++) {
        if (isNaN(S.get(i, j))) {
          indices.push(i);
          break;
        }
      }
    }
  }
  // if there than just one zero of NaN element
  // run a NMF with the residual matrix Y - A*B
  if (indices.length > 0) {
    let temp = fastExtractNMF(
      Y.clone().mmul(A.clone().mmul(S)),
      indices.length,
    );
    for (let i = 0; i < indices.length; i++) {
      for (let j = 0; j < S.columns; j++) {
        S.set(indices[i], j, temp.S.get(i, j));
      }
      for (let j = 0; j < A.rows; j++) {
        A.set(j, indices[i], temp.A.get(j, i));
      }
    }
  }
  return Object.assign({}, data, { A, S });
}

function fastExtractNMF(residual, r) {
  if (r <= 0) return { A: [], S: {} };
  let columns = residual.columns;
  let rows = residual.rows;
  let A = Matrix.zeros(rows, r);
  let S = Matrix.zeros(r, columns);
  for (let i = 0; i < r; i++) {
    residual = getMax(residual, 0);
    if (residual.sum() === 0) continue;
    let res2 = Matrix.pow(residual, 2).sum('column');
    //find the max of the first column

    let maxIndex = 0;
    for (let j = 1; j < res2.length; j++) {
      if (res2[maxIndex] < res2[j]) maxIndex = j;
    }

    if (res2[maxIndex] > 0) {
      let sqrtMaxValue = Math.sqrt(res2[maxIndex]);
      for (let j = 0; j < rows; j++) {
        let value = residual.get(j, maxIndex) / sqrtMaxValue;
        A.set(j, i, value);
      }
      let temp = A.getColumnVector(i).transpose().mmul(residual);
      for (let j = 0; j < columns; j++) {
        S.set(i, j, Math.max(temp.get(0, j), 0));
      }
      let subtracting = A.getColumnVector(i).mmul(S.getRowVector(i));
      residual = residual.sub(subtracting);
    }
  }
  return { A, S };
}

function nonNegSparseUpdateS(A, Sinit, Y, lambda, options) {
  let { maximumFBIIteration, FBRelativeDifferenceTolerance } = options;
  let H = A.transpose().mmul(A);
  let AtY = A.transpose().mmul(Y);
  let evd = new EVD(H, { assumeSymmetric: true });
  let L = Math.max(...evd.realEigenvalues);
  let t = 1;
  let S = Sinit.clone();
  let prevS = S.clone();
  let gradient = (s) => H.clone().mmul(s).sub(AtY);
  let proximal = (x, threshold) => getMax(x.subS(threshold), 0);

  for (let i = 0; i < maximumFBIIteration; i++) {
    let tNext = (1 + Math.sqrt(1 + 4 * t * t)) / 2;
    let w = (t - 1) / tNext;
    t = tNext;
    let R = Matrix.mul(S, 1 + w).sub(Matrix.mul(prevS, w));
    prevS = S.clone();
    S = proximal(R.sub(gradient(R).divS(L)), lambda / L);
    if (
      Matrix.sub(prevS, S).norm() / S.norm() <
      FBRelativeDifferenceTolerance
    ) {
      break;
    }
  }
  return S;
}

function nonNegSparseUpdateA(Ainit, S, Y, lambda, options) {
  let {
    maximumFBIIteration,
    FBRelativeDifferenceTolerance,
    normConstrained = false,
  } = options;
  let H = S.mmul(S.transpose());
  let YSt = Y.clone().mmul(S.transpose());
  let evd = new EVD(H, { assumeSymmetric: true });
  let L = Math.max(...evd.realEigenvalues);
  let A = Ainit;
  let prevA = A.clone();
  let t = 1;

  let gradient = (a) => a.clone().mmul(H).sub(YSt);
  let proximal;
  if (normConstrained) {
    let normLimits = dimNorm(Ainit, 'column');
    proximal = (x, threshold) =>
      normProj(getMax(x.subS(threshold), 0), normLimits);
  } else {
    proximal = (x, threshold) => getMax(x.subS(threshold), 0);
  }

  for (let i = 0; i < maximumFBIIteration; i++) {
    let tNext = (1 + Math.sqrt(1 + 4 * t * t)) / 2;
    let w = (t - 1) / tNext;
    t = tNext;
    let B = Matrix.mul(A, w + 1).sub(Matrix.mul(prevA, w));
    prevA = A.clone();
    A = proximal(B.sub(gradient(B).divS(L)), lambda / L);
    if (
      Matrix.sub(prevA, A).norm() / A.norm() <
      FBRelativeDifferenceTolerance
    ) {
      break;
    }
  }
  return A;
}

function normProj(X, normLimits) {
  let norms;
  let r = X.rows;
  let c = X.columns;
  if (normLimits.rows === r) {
    norms = dimNorm(X, 'row');
    //select rows with norm > 0 then multiply twise by the min
    for (let i = 0; i < r; i++) {
      if (norms.get(i, 0) <= 0) continue;
      for (let j = 0; j < c; j++) {
        let value =
          X.get(i, j) *
          Math.min(norms.get(i, 0), normLimits.get(i, 0) / norms.get(i, 0));
        X.set(i, j, value);
      }
    }
  } else {
    norms = dimNorm(X, 'column');
    for (let i = 0; i < c; i++) {
      if (norms.get(0, i) <= 0) continue;
      for (let j = 0; j < r; j++) {
        let value =
          X.get(j, i) *
          Math.min(norms.get(0, i), normLimits.get(0, i) / norms.get(0, i));
        X.set(j, i, value);
      }
    }
  }
  return X;
}

function dimNorm(x, dim) {
  let norms = Matrix.mul(x, x).sum(dim);
  let length = norms.length;
  for (let i = 0; i < length; i++) {
    norms[i] = Math.sqrt(norms[i]);
  }
  return dim === 'row'
    ? Matrix.from1DArray(length, 1, norms)
    : Matrix.from1DArray(1, length, norms);
}

function getMax(X, value) {
  let rows = X.rows;
  let columns = X.columns;
  let newMatrix = new Matrix(X);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < columns; c++) {
      if (newMatrix.get(r, c) < value) {
        newMatrix.set(r, c, value);
      }
    }
  }
  return newMatrix;
}
