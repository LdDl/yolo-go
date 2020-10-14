package yologo

import (
	"fmt"
	"image"

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
	op.bestAnchors = getBestAnchors_f32(target, op.anchors, op.masks, op.dimensions, gridSizeF32)
	for i := 0; i < len(op.bestAnchors); i++ {
		scale := (2 - target[i*5+3]*target[i*5+4])
		giInt := op.bestAnchors[i][1]
		gjInt := op.bestAnchors[i][2]
		gx := invsigm_f32(target[i*5+1]*gridSizeF32 - float32(giInt))
		gy := invsigm_f32(target[i*5+2]*gridSizeF32 - float32(gjInt))
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

func invsigm_f32(target float32) float32 {
	return -math32.Log(1-target+1e-16) + math32.Log(target+1e-16)
}

func getBestAnchors_f32(target []float32, anchors []float32, masks []int, dims int, gridSize float32) [][]int {
	bestAnchors := make([][]int, len(target)/5)
	imgsize := float32(dims)
	for j := 0; j < len(target); j = j + 5 {
		targetRect := rectifyBox_f32(0, 0, target[j+3]*imgsize, target[j+4]*imgsize, dims) //not absolutely confident in rectangle sizes
		bestIOU := float32(0.0)
		bestAnchors[j/5] = make([]int, 3)
		for i := 0; i < len(anchors); i = i + 2 {
			anchorRect := rectifyBox_f32(0, 0, anchors[i], anchors[i+1], dims)
			currentIOU := iou_f32(anchorRect, targetRect)
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

func rectifyBox_f32(x, y, h, w float32, imgSize int) image.Rectangle {
	return image.Rect(MaxInt(int(x-w/2), 0), MaxInt(int(y-h/2), 0), MinInt(int(x+w/2+1), imgSize), MinInt(int(y+h/2+1), imgSize))
}

func iou_f32(r1, r2 image.Rectangle) float32 {
	intersection := r1.Intersect(r2)
	interArea := intersection.Dx() * intersection.Dy()
	r1Area := r1.Dx() * r1.Dy()
	r2Area := r2.Dx() * r2.Dy()
	return float32(interArea) / float32(r1Area+r2Area-interArea)
}

func findIntElement(arr []int, ele int) int {
	for i := range arr {
		if arr[i] == ele {
			return i
		}
	}
	return -1
}
