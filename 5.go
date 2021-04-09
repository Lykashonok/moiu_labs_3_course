package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

type Pos struct {
	i, j int
}

func NorthWestMethod(a, b *mat.VecDense) (*mat.Dense, []Pos) {
	lenA, lenB := a.Len(), b.Len()
	x, pos := mat.NewDense(lenA, lenB, nil), make([]Pos, 0)
	i, j := 0, 0
	for {
		val := b.AtVec(j) - a.AtVec(i)
		if val >= 0 {
			x.Set(i, j, a.AtVec(i))
			b.SetVec(j, val)
			a.SetVec(i, 0)
			pos = append(pos, Pos{i, j})
			if i < lenA {
				i++
			}
		} else {
			val = a.AtVec(i) - b.AtVec(j)
			x.Set(i, j, b.AtVec(j))
			a.SetVec(i, val)
			b.SetVec(j, 0)
			pos = append(pos, Pos{i, j})
			if j < lenB {
				j++
			}
		}
		if a.AtVec(lenA-1) == 0 && b.AtVec(lenB-1) == 0 {
			break
		} else if i == lenA && j < lenB {
			i--
		} else if i < lenA && j == lenB {
			j--
		}
	}
	return x, pos
}

func bfs(adj *mat.Dense, pos []Pos, newPos Pos) []int {
	n, _ := adj.Dims()
	signs, queue, currentSign := make([]int, n), make([]Pos, 0), 1
	_, i := FindPos(pos, newPos)
	queue = append(queue, newPos)
	signs[i] = currentSign
	currentSign *= -1
	for len(queue) > 0 {
		currentPos := queue[0]
		queue = queue[1:]
		_, currentPosIndex := FindPos(pos, currentPos)
		adjRow := adj.RawRowView(currentPosIndex)
		for j, adjRowValue := range adjRow {
			if int(adjRowValue) == 1 {
				found, posIndex := FindPos(pos, pos[j])
				if found && signs[posIndex] == 0 {
					signs[posIndex] = currentSign
					queue = append(queue, pos[posIndex])
				}
			}
		}
		currentSign *= -1
	}
	return signs
}

func getUVBfs(c *mat.Dense, b []Pos) (*mat.VecDense, *mat.VecDense) {
	lenA, lenB := c.Dims()
	u, v := mat.NewVecDense(lenA, nil), mat.NewVecDense(lenB, nil)

	// 0 - unvisited, 1 - u visited, 2 - v visited, 3 - u and v visited
	visited := make([]int, len(b))

	// Artificial u1 = 0
	u.SetVec(0, 0)
	visited[0] = 1
	queue := make([]Pos, 1)
	queue[0] = b[0]

	// Solving linear system equation indirectly
	for len(queue) > 0 {
		currentPos := queue[0]
		queue = queue[1:]
		_, currentPosIndex := FindPos(b, currentPos)
		visit := visited[currentPosIndex]
		if visit != 0 && visit != 3 {
			// componentWasSet := ' '
			if visit == 1 {
				v.SetVec(currentPos.j, c.At(currentPos.i, currentPos.j)-u.AtVec(currentPos.i))
				// componentWasSet = 'i'
				visited[currentPosIndex] = 3
			} else if visit == 2 {
				u.SetVec(currentPos.i, c.At(currentPos.i, currentPos.j)-v.AtVec(currentPos.j))
				// componentWasSet = 'j'
				visited[currentPosIndex] = 3
			}
			for k := 0; k < len(visited); k++ {
				//currentPos != b[k]
				if visited[k] != 3 {
					if currentPos.i == b[k].i {
						visited[k] = 1
						queue = append(queue, b[k])
					} else if currentPos.j == b[k].j {
						visited[k] = 2
						queue = append(queue, b[k])
					}
				}
			}
		}
	}
	return u, v
}

func PotentialsMethod(a, b *mat.VecDense, c *mat.Dense) *mat.Dense {
	// If consumers and producers have different sums of values
	if !checkSum(a, b) {
		panic("Sums of needed and produced values are not the same!")
	}

	x, baselinePos := NorthWestMethod(a, b)
	return PotentialsMethodMainPhase(a, b, c, x, baselinePos)
}

// Potentials method solves transport problem and take 2 vectors and matrix
func PotentialsMethodMainPhase(a, b *mat.VecDense, c *mat.Dense, x *mat.Dense, baselinePos []Pos) *mat.Dense {
	fmt.Println("---Iteration start---")
	lenA, lenB := a.Len(), b.Len()

	fmt.Println("baselinePos at start", baselinePos, "\nfirst plan")
	matPrint(x)

	uVector, vVector := getUVBfs(c, baselinePos)
	fmt.Println("uVector", uVector.RawVector().Data, "\nvVector", vVector.RawVector().Data)

	nonBaselinePos, isOptimal, newBaselinePos := getNonBaselinePos(baselinePos, lenA, lenB), true, Pos{}

	for _, pos := range nonBaselinePos {
		if uVector.AtVec(pos.i)+vVector.AtVec(pos.j) > c.At(pos.i, pos.j) {
			isOptimal = false
			newBaselinePos = pos
			break
		}
	}
	if isOptimal {
		fmt.Println("There's no pos with u[i]+v[j]>c[i][j], current x is optimal")
		return x
	} else {
		fmt.Printf("There's nonbasline position with u[%v]+v[%v]>c[%v][%v] => %v + %v > %v\n", newBaselinePos.i+1, newBaselinePos.j+1, newBaselinePos.i+1, newBaselinePos.j+1, uVector.AtVec(newBaselinePos.i), vVector.AtVec(newBaselinePos.j), c.At(newBaselinePos.i, newBaselinePos.j))
	}

	// Copying x and clearing it
	x_copy, is_cleared := mat.DenseCopyOf(x), false

	baselinePosMatrix, nullPos := make([][]Pos, lenA), Pos{-1, -1}
	for i := 0; i < lenA; i++ {
		baselinePosMatrix[i] = make([]Pos, lenB)
		for j := 0; j < lenB; j++ {
			if found, _ := FindPos(baselinePos, Pos{i, j}); found {
				baselinePosMatrix[i][j] = Pos{i, j}
			} else {
				baselinePosMatrix[i][j] = nullPos
			}
		}
	}
	baselinePosMatrix[newBaselinePos.i][newBaselinePos.j] = newBaselinePos

	// set is_cleared = true, if row was deleted, set is_cleared = false. Exit from cycle when
	// no row or column deleted
	for is_cleared == false {
		is_cleared = true
		for i := 0; i < lenA; i++ {
			rowBaselinePosNumber := 0
			for j := 0; j < lenB; j++ {
				// Count baselinepos. If 1 or 0, delete this row
				if baselinePosMatrix[i][j] != nullPos {
					rowBaselinePosNumber++
				}
			}
			if rowBaselinePosNumber <= 1 {
				x_copy, baselinePosMatrix = deleteRow(x_copy, baselinePosMatrix, i)
				is_cleared = false
				break
			}
		}
		lenA, lenB = x_copy.Dims()
		for j := 0; j < lenB; j++ {
			colBaselinePosNumber := 0
			for i := 0; i < lenA; i++ {
				// Count baselinepos. If 1 or 0, delete this col
				if baselinePosMatrix[i][j] != nullPos {
					colBaselinePosNumber++
				}
			}
			if colBaselinePosNumber <= 1 {
				x_copy, baselinePosMatrix = deleteCol(x_copy, baselinePosMatrix, j)
				is_cleared = false
				break
			}
		}
		lenA, lenB = x_copy.Dims()
	}
	// New BaselinePos
	baselinePos_copy := make([]Pos, 0)
	for i := 0; i < lenA; i++ {
		for j := 0; j < lenB; j++ {
			if baselinePosMatrix[i][j] != nullPos {
				baselinePos_copy = append(baselinePos_copy, baselinePosMatrix[i][j])
			}
		}
	}
	fmt.Println("Cleared plan is")
	matPrint(x_copy)

	// finding min theta then process operation with x on baseline pos
	// first element of baselinePos is newbaseline pos
	adjacencyMatrix := NewAdjacencyMatrix(baselinePos_copy)
	signs := bfs(adjacencyMatrix, baselinePos_copy, newBaselinePos)

	minTheta, minThetaPos := math.Inf(1), Pos{}
	for k := 0; k < len(baselinePos_copy); k++ {
		if signs[k] == -1 {
			for i := 0; i < lenA; i++ {
				for j := 0; j < lenB; j++ {
					if baselinePos_copy[k] == baselinePosMatrix[i][j] && x_copy.At(i, j) < minTheta {
						minTheta, minThetaPos = x_copy.At(i, j), baselinePos_copy[k]
					}
				}
			}
		}
	}

	fmt.Printf("minTheta - %v %v\nnew x -\n", minTheta, minThetaPos)
	for i := 0; i < len(baselinePos_copy); i++ {
		x.Set(baselinePos_copy[i].i, baselinePos_copy[i].j, x.At(baselinePos_copy[i].i, baselinePos_copy[i].j)+minTheta*float64(signs[i]))
	}
	matPrint(x)

	// New x was created. Update in pos where theta was min to new pos
	for i := 0; i < len(baselinePos); i++ {
		if baselinePos[i] == minThetaPos {
			baselinePos[i] = newBaselinePos
		}
	}

	fmt.Printf("NewBaselinePos at end - %v\n---Iteration end---\n", baselinePos)
	return PotentialsMethodMainPhase(a, b, c, x, baselinePos)
}

func main() {
	a, b, c := readTransportProblem("input.txt", 3, 3)
	x := PotentialsMethod(a, b, c)
	matPrint(x)
}
