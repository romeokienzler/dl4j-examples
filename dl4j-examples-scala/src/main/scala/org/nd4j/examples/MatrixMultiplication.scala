package org.nd4j.examples

import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray

/**
 * This basic example shows how to create two matrices and multiply them
 *
 * @author Romeo Kienzler 
 */

object MatrixMultiplication {
  def main(args: Array[String]): Unit = {
    var v: INDArray = Nd4j.create(Array(Array(1d, 2d, 3d), Array(4d, 5d, 6d)))
    var w: INDArray = Nd4j.create(Array(Array(1d, 2d), Array(3d, 4d), Array(5d, 6d)))
    print(v.mul(w))
  }
}