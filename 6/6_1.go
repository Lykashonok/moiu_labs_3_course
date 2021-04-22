package main

import (
	"fmt"
	"io/ioutil"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

func readSquareProblem(input string, varNumber, conditionsNumber int) (*mat.VecDense, *mat.Dense, *mat.Dense) {
	str, err := ioutil.ReadFile(input)
	if err != nil {
		panic(err)
	}
	lines := strings.Split(string(str), "\n")
	objectiveVector, semiDefiniteMatrix, conditionsMatrix := mat.NewVecDense(varNumber, nil), mat.NewDense(varNumber, varNumber, nil), mat.NewDense(conditionsNumber, varNumber, nil)
	lines, objectiveVector = readVector(lines, varNumber)
	lines, semiDefiniteMatrix = readMatrix(lines, varNumber, varNumber)
	lines, conditionsMatrix = readMatrix(lines, conditionsNumber, varNumber)
	return objectiveVector, semiDefiniteMatrix, conditionsMatrix
}

func readVector(lines []string, number int) ([]string, *mat.VecDense) {
	vector := make([]float64, number)
	vectorValues := strings.Split(lines[0], " ")
	for i := 0; i < number; i++ {
		number, _ := strconv.ParseFloat(vectorValues[i], 64)
		vector[i] = number
	}
	return lines[1:], mat.NewVecDense(number, vector)
}

func readMatrix(lines []string, r, c int) ([]string, *mat.Dense) {
	matrix := mat.NewDense(r, c, nil)
	currentVector := mat.NewVecDense(c, nil)
	for i := 0; i < r; i++ {
		lines, currentVector = readVector(lines, c)
		matrix.SetRow(i, RawVector(currentVector))
	}
	return lines, matrix
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

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func vecMulMat(vec *mat.VecDense, matrix *mat.Dense) *mat.VecDense {
	a := mat.NewDense(1, vec.Len(), vec.RawVector().Data)
	b := mat.NewDense(vec.Len(), matrix.RawMatrix().Cols, matrix.RawMatrix().Data)
	var m mat.Dense
	m.Mul(a, b)

	// vectorPre := mat.NewDense(vec.Len(), vec.Len(), nil)
	// vectorPre.SetRow(0, RawVector(vec))
	// vectorPre.Mul(vectorPre, matrix.T())
	return mat.VecDenseCopyOf(m.RowView(0))
}

// SubstractSets = (a - b) or (a \ b). For example {1 2 3} \ {2 3} = {1}
func SubstractSets(a, b []float64) []float64 {
	result := make([]float64, len(a))
	copy(result, a)
	for _, v := range b {
		if found, _ := Find(a, v); found {
			result, _ = RemoveByValue(result, v)
		}
	}
	return result
}

// Find - find element in array
func Find(array []float64, value float64) (bool, int) {
	for index, curr := range array {
		if curr == value {
			return true, index
		}
	}
	return false, -1
}

func RemoveByIndex(slice []float64, index int) []float64 {
	return append(slice[:index], slice[index+1:]...)
}

func RemoveByValue(slice []float64, value float64) ([]float64, bool) {
	for index, v := range slice {
		if v == value {
			return RemoveByIndex(slice, index), true
		}
	}
	return slice, true
}
