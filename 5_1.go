package main

import (
	"io/ioutil"
	"strings"

	"gonum.org/v1/gonum/mat"
)

func checkSum(a, b *mat.VecDense) bool {
	sumA, sumB := 0., 0.
	for _, v := range RawVector(a) {
		sumA += v
	}
	for _, v := range RawVector(b) {
		sumB += v
	}
	return sumA == sumB
}

func NewAdjacencyMatrix(pos []Pos) *mat.Dense {
	adjacencyMatrix := mat.NewDense(len(pos), len(pos), nil)
	for i := 0; i < len(pos); i++ {
		for j := 0; j < len(pos); j++ {
			if i != j && (pos[i].i == pos[j].i || pos[i].j == pos[j].j) {
				adjacencyMatrix.Set(i, j, 1)
			}
		}
	}
	return adjacencyMatrix
}

func deleteRow(m *mat.Dense, posMatrix [][]Pos, row int) (*mat.Dense, [][]Pos) {
	r, c := m.Dims()
	newSlice := mat.NewDense(r-1, c, nil)
	for i := 0; i < r; i++ {
		if i < row {
			newSlice.SetRow(i, m.RawRowView(i))
		} else if i > row {
			newSlice.SetRow(i-1, m.RawRowView(i))
		}
	}
	newPosMatrix := make([][]Pos, r-1)
	for i := 0; i < r; i++ {
		if i < row {
			newPosMatrix[i] = posMatrix[i]
		} else if i > row {
			newPosMatrix[i-1] = posMatrix[i]
		}
	}
	return newSlice, newPosMatrix
}

func deleteCol(m *mat.Dense, posMatrix [][]Pos, col int) (*mat.Dense, [][]Pos) {
	r, c := m.Dims()
	newSlice := mat.NewDense(r, c-1, nil)
	for i := 0; i < c; i++ {
		if i < col {
			newSlice.SetCol(i, RawVector(m.ColView(i)))
		} else if i > col {
			newSlice.SetCol(i-1, RawVector(m.ColView(i)))
		}
	}
	newPosMatrix := make([][]Pos, r)
	for i := 0; i < r; i++ {
		newPosMatrix[i] = append(posMatrix[i][:col], posMatrix[i][col+1:]...)
	}
	return newSlice, newPosMatrix
}

func readTransportProblem(input string, lenA, lenB int) (*mat.VecDense, *mat.VecDense, *mat.Dense) {
	str, err := ioutil.ReadFile(input)
	if err != nil {
		panic(err)
	}
	lines := strings.Split(string(str), "\n")
	lines, a := readVector(lines, lenA)
	lines, b := readVector(lines, lenA)
	c, tmp := mat.NewDense(lenA, lenB, nil), mat.NewVecDense(lenB, nil)
	for i := 0; i < lenA; i++ {
		lines, tmp = readVector(lines, lenB)
		c.SetRow(i, RawVector(tmp))
	}
	return a, b, c
}

func getNonBaselinePos(baslinePos []Pos, lenA, lenB int) []Pos {
	nonBaselinePos := make([]Pos, 0)
	for i := 0; i < lenA; i++ {
		for j := 0; j < lenB; j++ {
			curPos := Pos{i, j}
			if CountPos(baslinePos, curPos) == 0 {
				nonBaselinePos = append(nonBaselinePos, curPos)
			}
		}
	}
	return nonBaselinePos
}
