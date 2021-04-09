package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

func doubleSimplexMethod(scalesVector *mat.VecDense, conditionsMatrix *mat.Dense, freeVector, baselineIndexes, yVector *mat.VecDense) (*mat.VecDense, *mat.VecDense) {
	fmt.Println("New iteration")
	// conditionsNumber - rows, varNumber - columns
	conditionsNumber, varNumber := conditionsMatrix.Dims()
	nonBaseLineIndexes, j := mat.NewVecDense(varNumber-conditionsNumber, nil), 0
	for i := 0; i < varNumber; i++ {
		if !Find(RawVector(baselineIndexes), float64(i)) {
			nonBaseLineIndexes.SetVec(j, float64(i))
			j++
		}
	}

	// BaselineVector and baselinematrix
	baselineMatrix := mat.NewDense(conditionsNumber, conditionsNumber, nil)
	baselineVector := mat.NewVecDense(conditionsNumber, nil)
	for i := 0; i < conditionsNumber; i++ {
		baselineMatrix.SetCol(i, RawVector(conditionsMatrix.ColView(int(baselineIndexes.AtVec(i)))))
		baselineVector.SetVec(i, scalesVector.AtVec(int(baselineIndexes.AtVec(i))))
	}
	baselineMatrixInv := mat.NewDense(conditionsNumber, conditionsNumber, nil)
	baselineMatrixInv.Inverse(baselineMatrix)

	// Vector Kappa
	baselineKappa := mat.NewVecDense(conditionsNumber, nil)
	baselineKappa.MulVec(baselineMatrixInv, freeVector)

	kappa := mat.NewVecDense(varNumber, nil)
	for i := 0; i < conditionsNumber; i++ {
		kappa.SetVec(int(baselineIndexes.AtVec(i)), baselineKappa.AtVec(i))
	}

	// Checking if kappa is optimal case
	isOptimalCase, negativeBaselineIndex := true, -1
	for i := 0; i < conditionsNumber; i++ {
		if baselineKappa.AtVec(i) < 0 {
			isOptimalCase = false
			negativeBaselineIndex = i
			break // if break is commented, last negative value will be observed, otherwise first
		}
	}
	if isOptimalCase {
		fmt.Println("current kappa is positive everywhere, end.")
		matPrint(kappa)
		return kappa, baselineIndexes
	} else {
		fmt.Println("current kappa is not positive everywhere")
		matPrint(kappa)
	}

	if yVector == nil {
		yVector = vecMulMat(baselineVector, baselineMatrixInv)
	}

	// y Deltavector is row with index of negative kappa value
	yDeltaVector := mat.NewVecDense(conditionsNumber, baselineMatrixInv.RawRowView(negativeBaselineIndex))
	fmt.Println("yDeltaVector")
	matPrint(yDeltaVector)

	muList := make([]float64, varNumber-conditionsNumber)
	// for nonbaseline indexes
	for i := 0; i < varNumber-conditionsNumber; i++ {
		mu := mat.VecDenseCopyOf(yDeltaVector)
		muList[i] = mat.Dot(mu, conditionsMatrix.ColView(int(nonBaseLineIndexes.AtVec(i))))
	}

	// If there's nothing to change, problem is not consistent
	isConsistent := false
	for i := 0; i < varNumber-conditionsNumber; i++ {
		if muList[i] < 0 {
			isConsistent = true
			break
		}
	}
	if !isConsistent {
		panic("Problem is not consistent")
	}

	// Finding min sigma and its index
	minSigma, minSigmaIndex := math.Inf(1), -1
	for i := 0; i < varNumber-conditionsNumber; i++ {
		if muList[i] < 0 {
			currentNonBaselineIndex := int(nonBaseLineIndexes.AtVec(i))
			Cj := scalesVector.AtVec(currentNonBaselineIndex)
			Aj := mat.Dot(conditionsMatrix.ColView(currentNonBaselineIndex), yVector)
			muj := muList[i]
			currentSigma := (Cj - Aj) / muj
			if currentSigma < minSigma {
				minSigma, minSigmaIndex = currentSigma, i
			}
		} else {
			muList[i] = math.Inf(1)
		}
	}
	// Changing dual plan (baseline indexes)
	newBaselineIndexes := mat.VecDenseCopyOf(baselineIndexes)
	newBaselineIndexes.SetVec(negativeBaselineIndex, float64(minSigmaIndex))
	fmt.Println("newBaselineIndexes")
	matPrint(newBaselineIndexes)
	yDeltaVector.ScaleVec(minSigma, yDeltaVector)

	// Updating y vector by adding y vector and scaled y delta vector
	yVector.AddVec(yVector, yDeltaVector)

	// Next iteration
	return doubleSimplexMethod(scalesVector, conditionsMatrix, freeVector, newBaselineIndexes, yVector)
}

// func main() {
// 	// Reading problem
// 	scalesVector, conditionsMatrix, freeVector, baselineIndexes := readDoubleOptimizationProblem("input.txt", 4, 2)

// 	// shifting all values by one cause of test cases
// 	for i := 0; i < baselineIndexes.Len(); i++ {
// 		baselineIndexes.SetVec(i, baselineIndexes.AtVec(i)-1)
// 	}

// 	// Solving problem
// 	optimalPlan, baselineIndexes := doubleSimplexMethod(scalesVector, conditionsMatrix, freeVector, baselineIndexes, nil)
// 	fmt.Printf("Result:\n")
// 	matPrint(optimalPlan)
// 	matPrint(baselineIndexes)
// }
