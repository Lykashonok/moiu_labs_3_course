package main

import (
	"fmt"
	"reflect"

	"gonum.org/v1/gonum/mat"
)

// SimplexPreparationPhase - returns baseline indexes with baseline vector
func SimplexPreparationPhase(scalesVector *mat.VecDense, conditionsMatrix *mat.Dense, freeVector *mat.VecDense) (*mat.VecDense, *mat.VecDense) {
	conditionsNumber, varNumber := conditionsMatrix.Dims()

	for i := 0; i < freeVector.Len(); i++ {
		if freeVector.AtVec(i) < 0 {
			panic("inconsistent")
		}
	}

	// row[i]*=-1 of conditional matrix where b[i] < 0
	for i := 0; i < conditionsNumber; i++ {
		if freeVector.AtVec(i) < 0 {
			freeVector.SetVec(i, freeVector.AtVec(i)*-1)
			condition := mat.NewVecDense(conditionsNumber, nil)
			condition.ScaleVec(-1, conditionsMatrix.ColView(i))
			conditionsMatrix.SetCol(i, RawVector(condition))
		}
	}

	// creating artificial (искусственных) scalesVector, conditions matrix,
	// baselineVector, baselineIndexes for main simplex phase
	artificialLength := varNumber + conditionsNumber
	artificialScalesVector := mat.NewVecDense(artificialLength, nil)
	artificialBaselineIndexes := mat.NewVecDense(conditionsNumber, nil)
	artificialBaselineVector := mat.NewVecDense(artificialLength, nil) // all elements are zeros
	for i := 0; i < artificialLength; i++ {
		if i >= varNumber {
			artificialScalesVector.SetVec(i, -1) // set 0 to
			artificialBaselineIndexes.SetVec(i-varNumber, float64(i))
			artificialBaselineVector.SetVec(i, freeVector.AtVec(i-varNumber)) // set b[i] for artificial values
		}
	}
	artificialConditionsMatrix := mat.NewDense(conditionsNumber, artificialLength, nil)
	for i := 0; i < conditionsNumber; i++ {
		for j := 0; j < artificialLength; j++ {
			if j < varNumber {
				artificialConditionsMatrix.Set(i, j, conditionsMatrix.At(i, j))
			} else {
				if j-varNumber == i {
					artificialConditionsMatrix.Set(i, j, 1)
				} else {
					artificialConditionsMatrix.Set(i, j, 0)
				}
			}
		}
	}

	fmt.Printf("Artificial problem, scales vector:\n")
	matPrint(artificialScalesVector)
	fmt.Printf("Artificial conditions matrix:\n")
	matPrint(artificialConditionsMatrix)
	artificialBaselineVector, artificialBaselineIndexes = SimplexMainPhase(artificialScalesVector, artificialConditionsMatrix, mat.NewDense(conditionsNumber, conditionsNumber, nil), artificialBaselineVector, artificialBaselineIndexes, 0, 0)
	fmt.Printf("Solved artificial problem\n")
	matPrint(artificialBaselineVector)
	matPrint(artificialBaselineIndexes)

	var (
		indexes []float64
		found   bool
	)

	for i := 0; i < varNumber; i++ {
		found = true
		for j := 0; j < artificialBaselineIndexes.Len(); j++ {
			if int(artificialBaselineIndexes.AtVec(j)) == i {
				found = false
				break
			}
		}
		if found {
			indexes = append(indexes, float64(i))
		}
	}
	nonBaselineOwnIndexes := mat.NewVecDense(len(indexes), indexes)

	eliminationIndex := -1 // in our notes it's named k
	for i := 0; i < artificialBaselineIndexes.Len(); i++ {
		if int(artificialBaselineIndexes.AtVec(i)) >= varNumber {
			eliminationIndex = i // find first artificial index position in nonBaselineOwnIndexes
			break                // break loop to not perform extra actions
		}
	}

	// There's no rows for elimination
	if eliminationIndex == -1 {
		fmt.Printf("There's no index to eliminate. Slicing baselineVector and shifting indexes by one\n")
		if reflect.DeepEqual(artificialBaselineVector.RawVector(), make([]float64, artificialLength)) {
			panic("inconsistent")
		}
		newBaselineVector := mat.VecDenseCopyOf(artificialBaselineVector.SliceVec(0, varNumber))
		// here we're just incrementing baslineIndexes by 1 cause numeration starts from 0
		for i := 0; i < artificialBaselineIndexes.Len(); i++ {
			artificialBaselineIndexes.SetVec(i, artificialBaselineIndexes.AtVec(i)+1)
		}
		return newBaselineVector, artificialBaselineIndexes
	}

	artificialBaselineMatrix := mat.NewDense(conditionsNumber, conditionsNumber, nil)
	for i := 0; i < conditionsNumber; i++ {
		artificialBaselineMatrix.SetCol(i, RawVector(artificialConditionsMatrix.ColView(int(artificialBaselineIndexes.AtVec(i)))))
	}
	artificialBaselineMatrixInv := mat.DenseCopyOf(artificialBaselineMatrix)
	artificialBaselineMatrixInv.Inverse(artificialBaselineMatrix)

	inconsistent := false
	// findings l[i] where i - nonbaseline own index
	l := mat.NewVecDense(nonBaselineOwnIndexes.Len(), nil)
	// matPrint(artificialConditionsMatrix)
	for i := 0; i < nonBaselineOwnIndexes.Len(); i++ {
		l.MulVec(artificialBaselineMatrixInv, artificialConditionsMatrix.ColView(int(nonBaselineOwnIndexes.AtVec(i))))
		if l.AtVec(eliminationIndex) != 0 {
			inconsistent = true
			break
		}
	}

	if inconsistent {
		matPrint(l)
		panic("inconsistent, there's l with value !=0 at k index")
	}

	// elimination
	newFreeVector := mat.NewVecDense(conditionsNumber-1, nil)
	newConditionsMatrix := mat.NewDense(conditionsNumber-1, varNumber, nil)
	for i := 0; i < conditionsNumber-1; i++ {
		if i != eliminationIndex {
			newConditionsMatrix.SetRow(i, conditionsMatrix.RawRowView(i))
			newFreeVector.SetVec(i, freeVector.AtVec(i))
		}
	}

	return SimplexPreparationPhase(scalesVector, newConditionsMatrix, newFreeVector)
}

func main() {
	c, r := 4, 2
	scalesVector, conditionsMatrix, freeVector, _, _ := readOptimizationProblem("input.txt", c, r, true)
	fmt.Printf("Optimization problem, scales vector:\n")
	matPrint(scalesVector)
	fmt.Printf("Conditions matrix:\n")
	matPrint(conditionsMatrix)
	fmt.Printf("Free vector:\n")
	matPrint(freeVector)
	baselineVector, baselineIndexes := SimplexPreparationPhase(scalesVector, conditionsMatrix, freeVector)
	fmt.Printf("Answer:\n")
	matPrint(baselineVector)
	matPrint(baselineIndexes)
}
