package main

import (
	"fmt"
	"io/ioutil"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// Find - find element in array
func Find(array []float64, value float64) bool {
	for _, curr := range array {
		if curr == value {
			return true
		}
	}
	return false
}

// Min - find min element in array
func Min(array []float64) float64 {
	min := array[0]
	for i := 0; i < len(array); i++ {
		if min > array[i] {
			min = array[i]
		}
	}
	return min
}

func matMulVec(m *mat.Dense, v *mat.VecDense) *mat.VecDense {
	r, c := m.Dims()
	sum := 0.0
	vector := mat.NewVecDense(c, nil)
	for i := 0; i < c; i++ {
		sum = 0
		for j := 0; j < r; j++ {
			sum += v.AtVec(i) * m.At(i, j)
		}
		vector.SetVec(i, sum)
	}
	return vector
}

// func vecMulMat(v *mat.VecDense, m *mat.Dense) *mat.VecDense {
// 	r, c := m.Dims()
// 	sum := 0.0
// 	vector := mat.NewVecDense(c, nil)
// 	for i := 0; i < c; i++ {
// 		sum = 0
// 		for j := 0; j < r; j++ {
// 			sum += v.AtVec(j) * m.At(j, i)
// 		}
// 		vector.SetVec(i, sum)
// 	}
// 	return vector
// }

// RawVector - return mat.Vector as []float64
func RawVector(v mat.Vector) []float64 {
	n := v.Len()
	r := make([]float64, n)
	for i := 0; i < n; i++ {
		r[i] = v.AtVec(i)
	}
	return r
}

func eye(n int) *mat.Dense {
	d := make([]float64, n*n)
	for i := 0; i < n*n; i += n + 1 {
		d[i] = 1
	}
	return mat.NewDense(n, n, d)
}

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func readMatrixFromStringArray(n, shift int, lines []string) *mat.Dense {
	var matrix []float64
	for i := shift * n; i < (shift+1)*n; i++ {
		numbersRaw := strings.Split(lines[i], " ")
		for j := 0; j < n; j++ {
			number, _ := strconv.ParseFloat(numbersRaw[j], 64)
			matrix = append(matrix, number)
		}
	}
	return mat.NewDense(n, n, matrix)
}

func readMatrixMatrixInvVectorIndex(input string) (*mat.Dense, *mat.Dense, *mat.VecDense, int) {
	str, err := ioutil.ReadFile(input)
	if err != nil {
		panic(err)
	}
	lines := strings.Split(string(str), "\n")
	n := len(strings.Split(lines[0], " "))
	// reading matrix
	matrix := readMatrixFromStringArray(n, 0, lines)
	// reading matrixInv
	matrixInv := readMatrixFromStringArray(n, 1, lines)
	// reading vector
	vector := make([]float64, n)
	for i := 2 * n; i < 3*n; i++ {
		number, _ := strconv.ParseFloat(lines[i], 64)
		vector[i%n] = number
	}
	// reading vector
	index, _ := strconv.Atoi(lines[len(lines)-1])
	return matrix, matrixInv, mat.NewVecDense(n, vector), index - 1
}

func mulOptimized(a, b *mat.Dense, index int) *mat.Dense {
	n, _ := a.Dims()
	result, subSum := mat.NewDense(n, n, nil), float64(0)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i != index {
				subSum = a.At(i, i)*b.At(i, j) + a.At(i, index)*b.At(index, j)
			} else {
				subSum = a.At(i, index) * b.At(index, j)
			}
			result.Set(i, j, subSum)
		}
	}
	return result
}

func invOptimized(matrix, matrixInv *mat.Dense, vector *mat.VecDense, index int) *mat.Dense {
	n := len(vector.RawVector().Data)
	// step 0
	matrix.SetCol(index, vector.RawVector().Data)

	// step 1
	AInv := mat.NewDense(n, n, make([]float64, n*n)) // clonning matrixInv
	AInv.CloneFrom(matrixInv)
	l := mat.VecDenseCopyOf(vector)
	l.MulVec(AInv, vector)
	if l.At(index, 0) == 0 {
		panic("Matrix is uninversable!!!")
	}
	// step 2
	storedNumber := l.At(index, 0)
	l.SetVec(index, -1)

	// step 3
	l.ScaleVec((-1 / storedNumber), l)

	// step 4
	Q := eye(n)
	Q.SetCol(index, l.RawVector().Data)

	// step 5
	result := mulOptimized(Q, AInv, index)
	return result
}
