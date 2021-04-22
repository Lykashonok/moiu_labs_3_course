package main

import (
	"fmt"
	"io/ioutil"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

func readVector(lines []string, number int) ([]string, *mat.VecDense) {
	vector := make([]float64, number)
	vectorValues := strings.Split(lines[0], " ")
	for i := 0; i < number; i++ {
		number, _ := strconv.ParseFloat(vectorValues[i], 64)
		vector[i] = number
	}
	return lines[1:], mat.NewVecDense(number, vector)
}

// RawVector - return mat.Vector as []float64
func RawVector(v mat.Vector) []float64 {
	n := v.Len()
	r := make([]float64, n)
	for i := 0; i < n; i++ {
		r[i] = v.AtVec(i)
	}
	return r
}

func vecMulMat(vec *mat.VecDense, matrix *mat.Dense) *mat.VecDense {
	vectorPre := mat.NewDense(vec.Len(), vec.Len(), nil)
	vectorPre.SetRow(0, RawVector(vec))
	vectorPre.Mul(vectorPre, matrix)
	return mat.VecDenseCopyOf(vectorPre.ColView(0))
}

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
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
