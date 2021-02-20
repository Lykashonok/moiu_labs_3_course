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
func SimplexMainPhase(scalesVector *mat.VecDense, conditionsMatrix *mat.Dense, freeMembersVector, baselineVector, baselineIndexes *mat.VecDense, optimizedMultiplication bool) *mat.VecDense {
	conditionsNumber, varNumber := conditionsMatrix.Dims()

	// creating baselineMatrix and inverse it (first iteration)
	baselineMatrix := mat.NewDense(conditionsNumber, conditionsNumber, nil)
	for i := 0; i < conditionsNumber; i++ {
		baselineMatrix.SetCol(i, RawVector(conditionsMatrix.ColView(int(baselineIndexes.AtVec(i)))))
	}

	inversedBaselineMatrix := mat.DenseCopyOf(baselineMatrix)
	if optimizedMultiplication {
		inversedBaselineMatrix.Inverse(baselineMatrix)
	} else {
		inversedBaselineMatrix.Inverse(baselineMatrix)
	}

	// finding components of scalesVector
	components := mat.NewVecDense(conditionsNumber, nil)
	for i := 0; i < conditionsNumber; i++ {
		components.SetVec(i, scalesVector.AtVec(int(baselineIndexes.AtVec(i))))
	}

	// potentials vector = components * inversed baselineMatrix
	potentials := vecMulMat(components, inversedBaselineMatrix)

	// score vector = potentials vector * baselineMatrix - scalesVector
	scoreVector := vecMulMat(potentials, conditionsMatrix)
	// scoreVector := mat.NewVecDense(varNumber, nil)
	// for i := 0; i < conditionsNumber; i++ {
	// 	scoreVector.SetVec(int(baselineIndexes.AtVec(i)), scoreVectorShifted.AtVec(i)) // !! or scoreVector.SetVec(i, scoreVectorShifted.AtVec(i)) !!
	// 	// scoreVector.SetVec(i, scoreVectorShifted.AtVec(i))
	// }
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
	for i := 0; i < nonBaseLineIndexes.Len(); i++ {
		if scoreVector.AtVec(int(nonBaseLineIndexes.AtVec(i))) < 0 {
			isOptimalCase = false
			break
		}
	}
	if isOptimalCase {
		// THIS IS OPTIMAL CASE
		return baselineVector
	}

	// The lowest nonBaseline index (Blends rule) for vector z
	lowestIndex := int(nonBaseLineIndexes.AtVec(0))
	conditionsColumn := mat.NewVecDense(conditionsNumber, RawVector(conditionsMatrix.ColView(lowestIndex)))
	zVector := matMulVec(inversedBaselineMatrix, conditionsColumn)

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
	// First value equals to theta, others following the formula
	newBaselineVector.SetVec(int(baselineIndexes.AtVec(0)), minTheta)
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
	return SimplexMainPhase(scalesVector, conditionsMatrix, freeMembersVector, newBaselineVector, newBaselineIndexes, true)
}

func main() {
	a, b := 5, 3
	// a, b := 6, 4
	// matPrint(vecMulMat(mat.NewVecDense(2, []float64{1, 2}), mat.NewDense(2, 3, []float64{3, 4, 5, 6, 7, 8})))
	scalesVector, conditionsMatrix, freeMembersVector, baselineVector, baselineIndexes := readOptimizationProblem("input.txt", a, b)
	result := SimplexMainPhase(scalesVector, conditionsMatrix, freeMembersVector, baselineVector, baselineIndexes, false)
	matPrint(result)
}
