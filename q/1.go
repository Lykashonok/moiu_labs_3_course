package main

import (
	"fmt"
	"io/ioutil"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

const debug = true

func mulVec(m *mat.Dense, v *mat.VecDense) *mat.VecDense {
	n, sum := len(v.RawVector().Data), float64(0)
	vector := mat.NewVecDense(n, nil)
	for i := 0; i < n; i++ {
		sum = 0
		for j := 0; j < n; j++ {
			sum += m.At(i, j) * v.At(i, 0)
		}
		vector.SetVec(i, sum)
	}
	return vector
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

func readMatrixMatrixInvVectorIndex() (*mat.Dense, *mat.Dense, *mat.VecDense, int) {
	str, err := ioutil.ReadFile("input.txt")
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
	if debug {
		fmt.Printf("Optimized multiplication\n")
		matPrint(a)
		fmt.Printf("on\n")
		matPrint(b)
		fmt.Printf("is\n")
	}
	for i := 0; i < n; i++ {
		if debug {
			fmt.Printf("%v row\n", i+1)
		}
		for j := 0; j < n; j++ {
			if debug {
				fmt.Printf("\t%v col\n", j+1)
			}
			if i != index {
				subSum = a.At(i, i)*b.At(i, j) + a.At(i, index)*b.At(index, j)
				if debug {
					fmt.Printf("\t\tc[%v][%v] = a[%v][%v] * b[%v][%v] + a[%v][%v] * b[%v][%v] = %.3f * %.3f + %.3f * %.3f = %.3f\n", i+1, j+1, i+1, i+1, i+1, j+1, i+1, index+1, index+1, j+1, a.At(i, i), b.At(i, j), a.At(i, index), b.At(index, j), subSum)
				}
			} else {
				subSum = a.At(i, index) * b.At(index, j)
				if debug {
					fmt.Printf("\t\tc[%v][%v] = a[%v][%v] * b[%v][%v] = %.3f * %.3f = %.3f\n", i+1, j+1, i+1, index+1, index+1, j+1, a.At(i, index), b.At(index, j), subSum)
				}
			}
			result.Set(i, j, subSum)
		}
	}
	return result
}

func invOptimized(matrix, matrixInv *mat.Dense, vector *mat.VecDense, index int) *mat.Dense {
	n := len(vector.RawVector().Data)
	// step 0
	if debug {
		matPrint(matrix)
		fmt.Printf("with\n")
		matPrint(vector)
		fmt.Printf("on %v col is\n", index)
	}
	matrix.SetCol(index, vector.RawVector().Data)
	if debug {
		matPrint(matrix)
	}

	// step 1
	AInv := mat.NewDense(n, n, make([]float64, n*n)) // clonning matrixInv
	AInv.CloneFrom(matrixInv)
	l := mulVec(AInv, vector)
	if l.At(index, 0) == 0 {
		panic("Matrix is uninversable!!!")
	} else if debug {
		fmt.Printf("l[%v] of\n", index)
		matPrint(l)
		fmt.Printf("isn't 0, it means matrix can be inverted\n")
	}

	// step 2
	storedNumber := l.At(index, 0)
	l.SetVec(index, -1)

	// step 3
	l.ScaleVec((-1 / storedNumber), l)
	if debug {
		fmt.Printf("vector l^ is\n")
		matPrint(l)
	}

	// step 4
	Q := eye(n)
	Q.SetCol(index, l.RawVector().Data)

	// step 5
	result := mulOptimized(Q, AInv, index)
	return result
}

func main() {
	matrix, matrixInv, vector, index := readMatrixMatrixInvVectorIndex()
	result := invOptimized(matrix, matrixInv, vector, index)
	if debug {
		fmt.Printf("golangs mat.Inverse() is\n")
		matrix.Inverse(matrix)
		matPrint(matrix)
		fmt.Printf("result is\n")
	}
	matPrint(result)
}
