package main

import (
	"io/ioutil"
	"strings"

	"gonum.org/v1/gonum/mat"
)

func vecMulMat(vec *mat.VecDense, matrix *mat.Dense) *mat.VecDense {
	vectorPre := mat.NewDense(vec.Len(), vec.Len(), nil)
	vectorPre.SetRow(0, RawVector(vec))
	vectorPre.Mul(vectorPre, matrix)
	return mat.VecDenseCopyOf(vectorPre.ColView(0))
}

func readDoubleOptimizationProblem(input string, varNumber, conditionsNumber int) (*mat.VecDense, *mat.Dense, *mat.VecDense, *mat.VecDense) {
	str, err := ioutil.ReadFile(input)
	if err != nil {
		panic(err)
	}
	lines := strings.Split(string(str), "\n")
	scalesVector, freeVector, baselineIndexes := mat.NewVecDense(varNumber, nil), mat.NewVecDense(varNumber, nil), mat.NewVecDense(varNumber, nil)
	lines, scalesVector = readVector(lines, varNumber)
	conditionsMatrix := mat.NewDense(conditionsNumber, varNumber, nil)
	for i := 0; i < conditionsNumber; i++ {
		condition := mat.NewVecDense(varNumber, nil)
		lines, condition = readVector(lines, varNumber)
		conditionsMatrix.SetRow(i, RawVector(condition))
	}
	lines, freeVector = readVector(lines, conditionsNumber)
	lines, baselineIndexes = readVector(lines, conditionsNumber)
	return scalesVector, conditionsMatrix, freeVector, baselineIndexes
}
