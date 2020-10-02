import { Matrix, solve } from 'ml-matrix';

import { zeroInsteadOfNegative } from '../util/zeroInsteadOfNegative';

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
      while (sumC[i] === 0) {
        sumC[i] = 0;
        for (let j = 0; j < rows; j++) {
          result.A.set(j, i, randGenerator());
          sumC[i] += result.A.get(j, i);
        }
      }
    }

    //resolve the system of equation Lx = D for x, then select just non negative values;
    result.S = zeroInsteadOfNegative(solve(result.A, originalMatrix));

    //set the S matrix with dimensions k x c
    // console.log('sumC', sumC);
    // let rIndex = 0;
    // result.S = Matrix.zeros(rank, columns);
    // for (let i = 0; i < sumC.length; i++) {
    //   if (sumC[i] > 0) {
    //     let rowCandidate = candidateS.getRow(rIndex++);
    //     for (let j = 0; j < columns; j++) {
    //       result.S.set(i, j, rowCandidate[j]);
    //     }
    //   }
    // }
    console.log('result S', result.S);
    //select rows with positive sum by row
    let sumR = result.S.sum('row');
    let positiveSumRowIndexS = [];
    let positiveSumRowS = [];
    for (let i = 0; i < sumR.length; i++) {
      if (sumR[i] > 0) {
        positiveSumRowIndexS.push(i);
        positiveSumRowS.push(result.S.getRow(i));
      }
    }

    positiveSumRowS = Matrix.checkMatrix(positiveSumRowS);
    let tempOriginalMatrixTransposed = new Matrix(
      columns,
      positiveSumRowIndexS.length,
    );

    for (let r = 0; r < columns; r++) {
      for (let c = 0; c < positiveSumRowIndexS.length; c++) {
        tempOriginalMatrixTransposed.set(
          r,
          c,
          originalMatrix.get(positiveSumRowIndexS[c], r),
        );
      }
    }

    console.log('before solve', positiveSumRowS, tempOriginalMatrixTransposed)
    // solve the system of linear equation xL = D for x.
    let candidateA = solve(
      positiveSumRowS.transpose(),
      tempOriginalMatrixTransposed,
    );

    result.A = zeroInsteadOfNegative(candidateA.transpose());

    result = checkMatrixS(result, originalMatrix);
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
