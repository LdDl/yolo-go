package yologo

import (
	"fmt"

	"github.com/chewxy/math32"
)

// YoloTrainer Wrapper around yoloOP
// It has method for setting desired bboxes as output of network
type YoloTrainer interface {
	ActivateTrainingMode()
	DisableTrainingMode()
	SetTarget([]float32)
}

type yoloTraining struct {
	inputs  []float32
	bboxes  []float32
	scales  []float32
	targets []float32
}

// ActivateTrainingMode Activates training mode for yoloOP
func (op *yoloOp) ActivateTrainingMode() {
	op.trainMode = true
}

// DisableTrainingMode Disables training mode for yoloOP
func (op *yoloOp) DisableTrainingMode() {
	op.trainMode = false
}

// SetTarget sets []float32 as desired target for yoloOP
func (op *yoloOp) SetTarget(target []float32) {
	preparedNumOfElements := op.gridSize * op.gridSize * len(op.masks) * (5 + op.numClasses)
	if op.training == nil {
		fmt.Println("Training parameters were not set. Initializing empty slices....")
		op.training = &yoloTraining{}
	}
	op.training.scales = make([]float32, preparedNumOfElements)
	op.training.targets = make([]float32, preparedNumOfElements)
	for i := range op.training.scales {
		op.training.scales[i] = 1
	}

	gridSizeF32 := float32(op.gridSize)
	op.bestAnchors = getBestAnchorsF32(target, op.anchors, op.masks, op.dimensions, gridSizeF32)
	for i := 0; i < len(op.bestAnchors); i++ {
		scale := (2 - target[i*5+3]*target[i*5+4])
		giInt := op.bestAnchors[i][1]
		gjInt := op.bestAnchors[i][2]
		gx := invsigmF32(target[i*5+1]*gridSizeF32 - float32(giInt))
		gy := invsigmF32(target[i*5+2]*gridSizeF32 - float32(gjInt))
		bestAnchor := op.masks[op.bestAnchors[i][0]] * 2
		gw := math32.Log(target[i*5+3]/op.anchors[bestAnchor] + 1e-16)
		gh := math32.Log(target[i*5+4]/op.anchors[bestAnchor+1] + 1e-16)
		bboxIdx := gjInt*op.gridSize*(5+op.numClasses)*len(op.masks) + giInt*(5+op.numClasses)*len(op.masks) + op.bestAnchors[i][0]*(5+op.numClasses)
		op.training.scales[bboxIdx] = scale
		op.training.targets[bboxIdx] = gx
		op.training.scales[bboxIdx+1] = scale
		op.training.targets[bboxIdx+1] = gy
		op.training.scales[bboxIdx+2] = scale
		op.training.targets[bboxIdx+2] = gw
		op.training.scales[bboxIdx+3] = scale
		op.training.targets[bboxIdx+3] = gh
		op.training.targets[bboxIdx+4] = 1
		for j := 0; j < op.numClasses; j++ {
			if j == int(target[i*5]) {
				op.training.targets[bboxIdx+5+j] = 1
			}
		}
	}
}

func getBestAnchorsF32(target []float32, anchors []float32, masks []int, dims int, gridSize float32) [][]int {
	bestAnchors := make([][]int, len(target)/5)
	imgsize := float32(dims)
	for j := 0; j < len(target); j = j + 5 {
		targetRect := rectifyBoxF32(0, 0, target[j+3]*imgsize, target[j+4]*imgsize, dims) //not absolutely confident in rectangle sizes
		bestIOU := float32(0.0)
		bestAnchors[j/5] = make([]int, 3)
		for i := 0; i < len(anchors); i = i + 2 {
			anchorRect := rectifyBoxF32(0, 0, anchors[i], anchors[i+1], dims)
			currentIOU := IOUFloat32(anchorRect, targetRect)
			if currentIOU >= bestIOU {
				bestAnchors[j/5][0] = i
				bestIOU = currentIOU
			}
		}
		bestAnchors[j/5][0] = findIntElement(masks, bestAnchors[j/5][0]/2)
		if bestAnchors[j/5][0] != -1 {
			bestAnchors[j/5][1] = int(target[j+1] * gridSize)
			bestAnchors[j/5][2] = int(target[j+2] * gridSize)
		}
	}
	return bestAnchors
}

func prepareTrainingOutputF32(input, yoloBoxes, target, scales []float32, bestAnchors [][]int, masks []int, numClasses, dims, gridSize int, ignoreTresh float32) []float32 {
	yoloBBoxes := make([]float32, len(yoloBoxes))
	bestIous := getBestIOUF32(yoloBoxes, target, numClasses, dims)
	for i := 0; i < len(yoloBoxes); i = i + (5 + numClasses) {
		if bestIous[i/(5+numClasses)][0] <= ignoreTresh {
			yoloBBoxes[i+4] = bceLossF32(0, yoloBoxes[i+4])
		}
	}
	for i := 0; i < len(bestAnchors); i++ {
		if bestAnchors[i][0] != -1 {
			giInt := bestAnchors[i][1]
			gjInt := bestAnchors[i][2]
			boxi := gjInt*gridSize*(5+numClasses)*len(masks) + giInt*(5+numClasses)*len(masks) + bestAnchors[i][0]*(5+numClasses)
			yoloBBoxes[boxi] = mseLossF32(target[boxi], input[boxi], scales[boxi])
			yoloBBoxes[boxi+1] = mseLossF32(target[boxi+1], input[boxi+1], scales[boxi+1])
			yoloBBoxes[boxi+2] = mseLossF32(target[boxi+2], input[boxi+2], scales[boxi+2])
			yoloBBoxes[boxi+3] = mseLossF32(target[boxi+3], input[boxi+3], scales[boxi+3])
			for j := 0; j < numClasses+1; j++ {
				yoloBBoxes[boxi+4+j] = bceLossF32(target[boxi+4+j], yoloBoxes[boxi+4+j])
			}
		}
	}
	return yoloBBoxes
}

func invsigmF32(target float32) float32 {
	return -math32.Log(1-target+1e-16) + math32.Log(target+1e-16)
}

func bceLossF32(target, pred float32) float32 {
	if target == 1.0 {
		return -(math32.Log(pred + 1e-16))
	}
	return -(math32.Log((1.0 - pred) + 1e-16))
}

func mseLossF32(target, pred, scale float32) float32 {
	return math32.Pow(scale*(target-pred), 2) / 2.0
}
