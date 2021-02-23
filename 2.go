package main

import (
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
func readOptimizationProblem(input string, varNumber, conditionsNumber int) (*mat.VecDense, *mat.Dense, *mat.VecDense, *mat.VecDense, *mat.VecDense) {
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
	lines, freeMembersVector = readVector(lines, conditionsNumber)

	//baselineVector
	lines, baselineVector = readVector(lines, varNumber)

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
func SimplexMainPhase(scalesVector *mat.VecDense, conditionsMatrix, inversedBaselineMatrix *mat.Dense, baselineVector, baselineIndexes *mat.VecDense, lowestIndex int, optimizedInv bool) *mat.VecDense {
	conditionsNumber, varNumber := conditionsMatrix.Dims()

	baselineMatrix := mat.NewDense(conditionsNumber, conditionsNumber, nil)
	for i := 0; i < conditionsNumber; i++ {
		baselineMatrix.SetCol(i, RawVector(conditionsMatrix.ColView(int(baselineIndexes.AtVec(i)))))
	}

	inversedBaselineMatrixTmp := mat.NewDense(conditionsNumber, conditionsNumber, nil)
	if optimizedInv {
		inversedBaselineMatrixTmp.Copy(invOptimized(baselineMatrix, inversedBaselineMatrix, mat.VecDenseCopyOf(conditionsMatrix.ColView(int(baselineIndexes.AtVec(lowestIndex)))), lowestIndex))
	} else {
		inversedBaselineMatrix.Inverse(baselineMatrix)
		inversedBaselineMatrixTmp.Copy(inversedBaselineMatrix)
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
		return baselineVector
	}

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

	// Run again with new baseline vector and new baseline indexes ()
	return SimplexMainPhase(scalesVector, conditionsMatrix, inversedBaselineMatrixTmp, newBaselineVector, newBaselineIndexes, minThetaIndex, true)
}

func main() {
	// a, b := 5, 3
	a, b := 6, 4

	scalesVector, conditionsMatrix, _, baselineVector, baselineIndexes := readOptimizationProblem("input.txt", a, b)
	result := SimplexMainPhase(scalesVector, conditionsMatrix, mat.NewDense(b, b, nil), baselineVector, baselineIndexes, 0, false)
	matPrint(result)
}
