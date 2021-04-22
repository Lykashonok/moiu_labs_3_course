package main

import (
	"fmt"
	"math"
	"reflect"

	"gonum.org/v1/gonum/mat"
)

func solveSquareProblem(objectiveVector *mat.VecDense, semiDefiniteMatrix, conditionsMatrix *mat.Dense, feasiblePlan, supConstraints, supConstraintsEx *mat.VecDense) (*mat.VecDense, *mat.VecDense, *mat.VecDense) {
	condNumber, varNumber := conditionsMatrix.Dims()
	supNumber, supExNumber := supConstraints.Len(), supConstraintsEx.Len()

	// 1 rank

	// 2 non-degenerate matrix
	baselineMatrix, baselineMatrixInv := mat.NewDense(condNumber, condNumber, nil), mat.NewDense(condNumber, condNumber, nil)
	for i := 0; i < condNumber; i++ {
		baselineMatrix.SetCol(i, RawVector(conditionsMatrix.ColView(int(supConstraints.AtVec(i)))))
	}
	baselineMatrixInv.Inverse(baselineMatrix)

	if mat.Det(baselineMatrix) == 0 {
		panic("determinant of Baseline matrix is 0!")
	}

	// 3.1
	rawSupEx, contains := RawVector(supConstraintsEx), true
	for i := 0; i < supNumber; i++ {
		if found, _ := Find(rawSupEx, supConstraints.AtVec(i)); !found {
			contains = false
			break
		}
	}
	if !contains {
		matPrint(supConstraints)
		matPrint(supConstraintsEx)
		panic("supConstraintsEx doesnt contain supConstraints")
	}

	// 3.2
	Dx := mat.NewVecDense(varNumber, nil)
	Dx.MulVec(semiDefiniteMatrix, feasiblePlan)
	cVector := mat.NewVecDense(varNumber, nil)
	cVector.AddVec(objectiveVector, Dx)
	cVectorBaseline := mat.NewVecDense(condNumber, nil)
	for i := 0; i < condNumber; i++ {
		cVectorBaseline.SetVec(i, cVector.AtVec(int(supConstraints.AtVec(i))))
	}
	cVectorBaseline.ScaleVec(-1, cVectorBaseline)
	uVector := vecMulMat(cVectorBaseline, baselineMatrixInv)
	uA := vecMulMat(uVector, conditionsMatrix)
	deltaVector := mat.NewVecDense(varNumber, nil)
	deltaVector.AddVec(uA, cVector)

	// H matrix creating
	conditionsMatrixEx := mat.NewDense(condNumber, supExNumber, nil)
	dMatrixEx := mat.NewDense(supExNumber, supExNumber, nil)
	for i := 0; i < supExNumber; i++ {
		conditionsMatrixEx.SetCol(i, RawVector(conditionsMatrix.ColView(int(supConstraintsEx.AtVec(i)))))
		for j := 0; j < supExNumber; j++ {
			dMatrixEx.Set(i, j, semiDefiniteMatrix.At(int(supConstraintsEx.AtVec(i)), int(supConstraintsEx.AtVec(j))))
		}
	}

	conditionsMatrixExT := conditionsMatrixEx.T()

	hMatrix := mat.NewDense(supExNumber+condNumber, supExNumber+condNumber, nil)
	for i := 0; i < supExNumber+condNumber; i++ {
		for j := 0; j < supExNumber+condNumber; j++ {
			if i < supExNumber && j < supExNumber {
				hMatrix.Set(i, j, dMatrixEx.At(i, j))
			} else if i < supExNumber && j >= supExNumber {
				hMatrix.Set(i, j, conditionsMatrixExT.At(i, j-supExNumber))
			} else if i >= supExNumber && j < supExNumber {
				hMatrix.Set(i, j, conditionsMatrixEx.At(i-supExNumber, j))
			}
		}
	}

	hMatrixInv := mat.NewDense(supExNumber+condNumber, supExNumber+condNumber, nil)
	hMatrixInv.Inverse(hMatrix)

	// optimal criteria
	isOptimal, _, negativeIndex := true, 0.0, 0 //negativeValue
	for i := 0; i < varNumber; i++ {
		if deltaVector.AtVec(i) < -0.00001 {
			// negativeValue = deltaVector.AtVec(i)
			negativeIndex = i
			isOptimal = false
			break
		}
	}
	if isOptimal {
		return feasiblePlan, supConstraints, supConstraintsEx
	}

	// l vector creating
	lVector := mat.NewVecDense(varNumber, nil)
	lVector.SetVec(negativeIndex, 1.0)
	bVector := mat.NewVecDense(supExNumber+condNumber, nil)
	for i := 0; i < supExNumber; i++ {
		bVector.SetVec(i, semiDefiniteMatrix.At(int(supConstraintsEx.AtVec(i)), negativeIndex))
	}
	for i := 0; i < condNumber; i++ {
		bVector.SetVec(i+supExNumber, conditionsMatrix.At(i, negativeIndex))
	}
	xVector := mat.NewVecDense(supExNumber+condNumber, nil)
	hMatrixInvNegative := mat.DenseCopyOf(hMatrixInv)
	hMatrixInvNegative.Scale(-1, hMatrixInvNegative)

	xVector.MulVec(hMatrixInvNegative, bVector)
	for i := 0; i < supExNumber; i++ {
		lVector.SetVec(int(supConstraintsEx.AtVec(i)), xVector.AtVec(i))
	}

	// minTheta
	δ := mat.Dot(vecMulMat(lVector, semiDefiniteMatrix), lVector)
	minTheta, minThetaIndex := 0.0, negativeIndex
	if δ == 0 {
		minTheta = math.Inf(1)
	} else if δ > 0 {
		minTheta = math.Abs(deltaVector.AtVec(negativeIndex)) / δ
	}

	for i := 0; i < supExNumber; i++ {
		if i != negativeIndex {
			currentTheta := math.Inf(1)
			if lVector.AtVec(i) < 0 {
				currentTheta = -(feasiblePlan.AtVec(i) / lVector.AtVec(i))
				if currentTheta < minTheta {
					minTheta = currentTheta
					minThetaIndex = i
				}
			}
		}
	}
	if minTheta == math.Inf(1) {
		panic("mintheta is infinity. Problem is inconsistent")
	}

	// feasiblePlan updating
	feasiblePlan.AddScaledVec(feasiblePlan, minTheta, lVector)
	fmt.Println("new plan is")
	matPrint(feasiblePlan)
	fmt.Println("new constraint vector is")
	matPrint(supConstraints)
	fmt.Println("new constraint vector extended is")
	matPrint(supConstraintsEx)

	// supConstraints updating
	rawSup := RawVector(supConstraints)
	rawSupEx = RawVector(supConstraintsEx)
	supConstraintsSubstraction := SubstractSets(rawSupEx, rawSup)
	// negativeIndex - j0
	// minThetaIndex - j*
	if int(minThetaIndex) == negativeIndex {
		rawSupEx = append(rawSupEx, float64(negativeIndex))
	} else if found, _ := Find(supConstraintsSubstraction, float64(minThetaIndex)); found {
		rawSupEx, _ = RemoveByValue(rawSupEx, float64(minThetaIndex))
	} else if found, s := Find(rawSup, float64(minThetaIndex)); found {
		// 1 - 3 condition, 2 - 4 condition, 0 - neither condition
		for _, jplus := range supConstraintsSubstraction {
			tempVector := mat.VecDenseCopyOf(conditionsMatrix.ColView(int(jplus)))
			tempVector.MulVec(baselineMatrixInv, tempVector)
			if tempVector.AtVec(s) != 0 {
				rawSup[s] = jplus
				rawSupEx, _ = RemoveByValue(rawSupEx, float64(minThetaIndex))
				break
			} else if tempVector.AtVec(s) == 0 || reflect.DeepEqual(rawSup, rawSupEx) {
				rawSup[s] = float64(negativeIndex)
				rawSupEx[s] = float64(negativeIndex)
				break
			}
		}
	}
	supConstraints = mat.NewVecDense(len(rawSup), rawSup)
	supConstraintsEx = mat.NewVecDense(len(rawSupEx), rawSupEx)

	return solveSquareProblem(objectiveVector, semiDefiniteMatrix, conditionsMatrix, feasiblePlan, supConstraints, supConstraintsEx)
}

func main() {
	varNumber, conditionsNumber := 3, 2
	objectiveVector, semiDefiniteMatrix, conditionsMatrix := readSquareProblem("input.txt", varNumber, conditionsNumber)
	// feasiblePlan, supConstraints, supConstraintsEx := mat.NewVecDense(varNumber, []float64{0, 0.5, 1}), mat.NewVecDense(2, []float64{1, 2}), mat.NewVecDense(2, []float64{1, 2})
	// feasiblePlan, supConstraints, supConstraintsEx := mat.NewVecDense(varNumber, []float64{2, 3, 0, 0}), mat.NewVecDense(2, []float64{0, 1}), mat.NewVecDense(2, []float64{0, 1})
	feasiblePlan, supConstraints, supConstraintsEx := mat.NewVecDense(varNumber, []float64{0, 10, 4}), mat.NewVecDense(2, []float64{1, 2}), mat.NewVecDense(2, []float64{1, 2})
	plan, constraints, constraintsEx := solveSquareProblem(objectiveVector, semiDefiniteMatrix, conditionsMatrix, feasiblePlan, supConstraints, supConstraintsEx)
	fmt.Println("Result is")
	matPrint(plan)
	matPrint(constraints)
	matPrint(constraintsEx)
}
