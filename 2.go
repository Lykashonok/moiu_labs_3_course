package main

import (
	"fmt"
	"io/ioutil"
	"math"
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

// Must be in canonical form
func readOptimizationProblem(input string, varNumber, conditionsNumber int, preparationPhase bool) (*mat.VecDense, *mat.Dense, *mat.VecDense, *mat.VecDense, *mat.VecDense) {
	var (
		scalesVector      *mat.VecDense
		conditionsMatrix  []float64
		freeMembersVector *mat.VecDense
		baselineVector    *mat.VecDense
		baselineIndexes   *mat.VecDense
	)

	//opening file
	str, err := ioutil.ReadFile(input)
	if err != nil {
		panic(err)
	}
	lines := strings.Split(string(str), "\n")

	//scalesVector
	lines, scalesVector = readVector(lines, varNumber)

	//conditionsMatrix
	for i := 0; i < conditionsNumber; i++ {
		conditionsMatrixNumbers := strings.Split(lines[i], " ")
		for j := 0; j < varNumber; j++ {
			number, _ := strconv.ParseFloat(conditionsMatrixNumbers[j], 64)
			conditionsMatrix = append(conditionsMatrix, number)
		}
	}
	lines = lines[conditionsNumber:]

	//freeMembersVector
	freeMembersVector = mat.NewVecDense(conditionsNumber, nil)
	lines, freeMembersVector = readVector(lines, conditionsNumber)

	//baselineVector
	baselineVector = mat.NewVecDense(varNumber, nil)
	if preparationPhase == false {
		lines, baselineVector = readVector(lines, varNumber)
	}

	//determining baseline indexes
	baselineIndexes = mat.NewVecDense(conditionsNumber, nil)
	j := 0
	for i := 0; i < varNumber; i++ {
		if baselineVector.AtVec(i) != 0 {
			baselineIndexes.SetVec(j, float64(i))
			j++
		}
	}

	return scalesVector, mat.NewDense(conditionsNumber, varNumber, conditionsMatrix), freeMembersVector, baselineVector, baselineIndexes
}

// SimplexMainPhase - solves optimization problem in canonical form
func SimplexMainPhase(scalesVector *mat.VecDense, conditionsMatrix, inversedBaselineMatrix *mat.Dense, baselineVector, baselineIndexes *mat.VecDense, lowestIndex int, iteration int) (*mat.VecDense, *mat.VecDense) {
	conditionsNumber, varNumber := conditionsMatrix.Dims()

	// Building baselineMatrix from baselineIndexes of conditionsMatrix
	baselineMatrix := mat.NewDense(conditionsNumber, conditionsNumber, nil)
	for i := 0; i < conditionsNumber; i++ {
		baselineMatrix.SetCol(i, RawVector(conditionsMatrix.ColView(int(baselineIndexes.AtVec(i)))))
	}
	fmt.Printf("New baseline vector\n")
	matPrint(baselineVector)
	matPrint(baselineMatrix)

	inversedBaselineMatrixTmp := mat.NewDense(conditionsNumber, conditionsNumber, nil)
	if iteration == 0 {
		// First iteration. Inversing via gonums mat.Inverse()
		inversedBaselineMatrix.Inverse(baselineMatrix)
		inversedBaselineMatrixTmp.Copy(inversedBaselineMatrix)
	} else {
		// Other operations. Inversing via invOptimized() from 1.go file from 1 lab
		inversedBaselineMatrixTmp.Copy(invOptimized(baselineMatrix, inversedBaselineMatrix, mat.VecDenseCopyOf(conditionsMatrix.ColView(int(baselineIndexes.AtVec(lowestIndex)))), lowestIndex))
	}

	// finding components of scalesVector
	components := mat.NewVecDense(conditionsNumber, nil)
	for i := 0; i < conditionsNumber; i++ {
		components.SetVec(i, scalesVector.AtVec(int(baselineIndexes.AtVec(i))))
	}

	// potentials vector = components * inversed baselineMatrix
	potentials := vecMulMat(components, inversedBaselineMatrixTmp)
	scoreVector := vecMulMat(potentials, conditionsMatrix)
	scoreVector.AddScaledVec(scoreVector, -1, scalesVector)
	// find nonBaseline indexes
	nonBaseLineIndexes, j := mat.NewVecDense(varNumber-conditionsNumber, nil), 0
	for i := 0; i < conditionsNumber; i++ {
		if !Find(RawVector(baselineIndexes), float64(i)) {
			nonBaseLineIndexes.SetVec(j, float64(i))
			j++
		}
	}

	// First exit condition, if current case is optimal
	isOptimalCase := true
	lowestIndex = 0
	for i := 0; i < scoreVector.Len(); i++ {
		if scoreVector.AtVec(i) < 0 {
			isOptimalCase = false
			lowestIndex = i
			break
		}
	}
	if isOptimalCase {
		// THIS IS OPTIMAL CASE
		fmt.Printf("every deltas element of \n")
		matPrint(scoreVector)
		fmt.Printf("> 0, baseline vector is optimal case \n")
		return baselineVector, baselineIndexes
	}
	fmt.Printf("scoreVector[%v] of delta\n", lowestIndex+1)
	matPrint(scoreVector)
	fmt.Printf("%v < 0\n", scoreVector.AtVec(lowestIndex))

	// The lowest nonBaseline index (Blends rule) for vector z
	var conditionsColumn = conditionsMatrix.ColView(lowestIndex)
	var zVector = mat.VecDenseCopyOf(conditionsColumn)
	zVector.MulVec(inversedBaselineMatrixTmp, conditionsColumn)

	// Theta
	minTheta, minThetaIndex, thetaValue := math.Inf(+1), 0, 0.0
	for j := 0; j < conditionsNumber; j++ {
		z := zVector.AtVec(j)
		if z > 0 {
			thetaValue = baselineVector.AtVec(int(baselineIndexes.AtVec(j))) / z
		} else {
			thetaValue = math.Inf(+1)
		}
		if thetaValue < minTheta {
			minTheta = thetaValue
			minThetaIndex = j
		}
	}
	if math.IsInf(minTheta, 1) {
		panic("Problem is unsolvable")
	}

	// changing baseline indexes
	baselineIndexes.SetVec(minThetaIndex, float64(lowestIndex))
	newBaselineIndexes := mat.VecDenseCopyOf(baselineIndexes)

	// creating new baselineVector (new baseline case)
	newBaselineVector := mat.NewVecDense(varNumber, nil)
	// minThetaIndex value equals to theta, others following the formula
	newBaselineVector.SetVec(int(baselineIndexes.AtVec(minThetaIndex)), minTheta)
	for i := 0; i < conditionsNumber; i++ {
		newValue := 0.0
		if i != minThetaIndex {
			newValue = baselineVector.AtVec(int(baselineIndexes.AtVec(i))) - minTheta*zVector.AtVec(i)
		} else {
			newValue = minTheta
		}
		newBaselineVector.SetVec(int(baselineIndexes.AtVec(i)), newValue)
	}

	fmt.Printf("New baseline vector on %v iteration is\n", iteration+1)
	matPrint(newBaselineVector)

	// Run again with new baseline vector and new baseline indexes ()
	return SimplexMainPhase(scalesVector, conditionsMatrix, inversedBaselineMatrixTmp, newBaselineVector, newBaselineIndexes, minThetaIndex, iteration+1)
}

// func main() {
// 	c, r := 6, 4
// 	scalesVector, conditionsMatrix, _, baselineVector, baselineIndexes := readOptimizationProblem("input.txt", c, r, false)
// 	fmt.Printf("scales vector is\n")
// 	matPrint(scalesVector)
// 	fmt.Printf("matrix of conditions is\n")
// 	matPrint(conditionsMatrix)
// 	fmt.Printf("first baseline vector is\n")
// 	matPrint(baselineVector)
// 	fmt.Printf("and it's baseline indexes\n")
// 	matPrint(baselineIndexes)
// 	result, indexes := SimplexMainPhase(scalesVector, conditionsMatrix, mat.NewDense(r, r, nil), baselineVector, baselineIndexes, 0, 0)
// 	fmt.Printf("result is\n")
// 	matPrint(result)
// 	matPrint(indexes)
// }
