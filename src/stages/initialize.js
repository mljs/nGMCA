import { Matrix, solve } from 'ml-matrix';

import { zeroInsteanOfNegative } from '../util';

import { checkMatrixS } from './checkMatrixS';
import { updateMatrixA } from './updateMatrixA';
import { updateMatrixS } from './updateMatrixS';

export function initialize(originalMatrix, options = {}) {
  const {
    rank,
    randGenerator,
    maxInitFBIteration,
    toleranceFBInit,
    maxFBIteration,
    toleranceFB,
    normConstrained,
  } = options;

  let result = {};
  let rows = originalMatrix.rows;
  let columns = originalMatrix.columns;

  result.A = Matrix.rand(rows, rank, { random: randGenerator });

  for (let iter = 0; iter < maxInitFBIteration; iter++) {
    //select columns with sum positive from A
    let sumC = result.A.sum('column');
    let positiveSumColumnA = [];
    for (let i = 0; i < sumC.length; i++) {
      if (sumC[i] > 0) positiveSumColumnA.push(result.A.getColumn(i));
    }
    positiveSumColumnA = new Matrix(positiveSumColumnA);
    //resolve the system of equation Lx = originalMatrix for x, then select just non negative values;
    let candidateS = zeroInsteanOfNegative(
      solve(positiveSumColumnA.transpose(), originalMatrix),
    );
    //set the S matrix with dimensions k x c
    let rIndex = 0;
    result.S = Matrix.zeros(rank, columns);
    for (let i = 0; i < sumC.length; i++) {
      if (sumC[i] > 0) {
        let rowCandidate = candidateS.getRow(rIndex++);
        for (let j = 0; j < columns; j++) {
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
    let candidateA = solve(
      positiveSumRowS.transpose(),
      originalMatrix.transpose(),
    );

    candidateA = zeroInsteanOfNegative(candidateA.transpose());

    let cIndex = 0;
    result.A = Matrix.zeros(rows, rank);
    for (let i = 0; i < sumR.length; i++) {
      if (sumR[i] > 0) {
        let colCandidate = candidateA.getColumn(cIndex++);
        for (let j = 0; j < rows; j++) {
          result.A.set(j, i, colCandidate[j]);
        }
      }
    }

    let prevS = result.S.clone();
    result.S = updateMatrixS(result.A, result.S, originalMatrix, 0, {
      maxFBIteration,
      toleranceFB,
    });
    result = checkMatrixS(result, originalMatrix);
    result.A = updateMatrixA(result.A, result.S, originalMatrix, 0, {
      maxFBIteration,
      toleranceFB,
      normConstrained,
    });

    if (
      Matrix.sub(prevS, result.S).norm() / result.S.norm() <
      toleranceFBInit
    ) {
      break;
    } //checkIt
  }
  return result;
}
