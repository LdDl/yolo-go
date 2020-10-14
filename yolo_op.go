package yologo

import (
	"fmt"
	"hash"
	"hash/fnv"

	"github.com/chewxy/hm"
	"github.com/chewxy/math32"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type yoloOp struct {
	anchors     []float32
	masks       []int
	ignoreTresh float32
	dimensions  int
	numClasses  int

	trainMode   bool
	gridSize    int
	bestAnchors [][]int
	training    *yoloTraining
}

func newYoloOp(anchors []float32, masks []int, netSize, gridSize, numClasses int, ignoreTresh float32) *yoloOp {
	yoloOp := &yoloOp{
		anchors:     anchors,
		dimensions:  netSize,
		numClasses:  numClasses,
		ignoreTresh: ignoreTresh,
		masks:       masks,
		trainMode:   false,
		gridSize:    gridSize,
		training:    &yoloTraining{},
	}
	return yoloOp
}

// YOLOv3Node Constructor for YOLO-based node operation
/*
	input - Input node
	anchors - Slice of anchors
	masks - Slice of masks
	netSize - Height/Width of input
	numClasses - Amount of classes
	ignoreTresh - Treshold
	targets - Desired targets.
*/
func YOLOv3Node(input *gorgonia.Node, anchors []float32, masks []int, netSize, numClasses int, ignoreTresh float32, targets ...*gorgonia.Node) (*gorgonia.Node, error) {
	// @todo: need to check input.Shape()[2] accessibility
	op := newYoloOp(anchors, masks, netSize, numClasses, input.Shape()[2], ignoreTresh)
	return gorgonia.ApplyOp(op, input)
}

/* Methods to match gorgonia.Op interface */

func (op *yoloOp) Arity() int {
	return 1
}
func (op *yoloOp) ReturnsPtr() bool  { return false }
func (op *yoloOp) CallsExtern() bool { return false }
func (op *yoloOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "YOLO{}(anchors: (%v))", op.anchors)
}
func (op *yoloOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}
func (op *yoloOp) String() string {
	return fmt.Sprintf("YOLO{}(anchors: (%v))", op.anchors)
}
func (op *yoloOp) InferShape(inputs ...gorgonia.DimSizer) (tensor.Shape, error) {
	shp := inputs[0].(tensor.Shape)
	if len(shp) < 4 {
		return nil, fmt.Errorf("InferShape() for YOLO must contain 4 dimensions, but recieved %d)", len(shp))
	}
	s := shp.Clone()
	if op.trainMode {
		return []int{s[0], s[2] * s[3] * len(op.masks), (s[1] - 1) / len(op.masks)}, nil
	}
	return []int{s[0], s[2] * s[3] * len(op.masks), s[1] / len(op.masks)}, nil
}
func (op *yoloOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := gorgonia.TensorType{Dims: 4, Of: a}
	o := gorgonia.TensorType{Dims: 3, Of: a}
	return hm.NewFnType(t, o)
}
func (op *yoloOp) OverwritesInput() int {
	return -1
}

func (op *yoloOp) Do(inputs ...gorgonia.Value) (retVal gorgonia.Value, err error) {
	inputTensor, err := op.checkInput(inputs...)
	if err != nil {
		return nil, errors.Wrap(err, "Can't check YOLO input")
	}
	batchSize := inputTensor.Shape()[0]
	stride := op.dimensions / inputTensor.Shape()[2]
	gridSize := inputTensor.Shape()[2]
	bboxAttributes := 5 + op.numClasses
	numAnchors := len(op.anchors) / 2
	currentAnchors := []float32{}
	for i := range op.masks {
		if op.masks[i] >= numAnchors {
			return nil, fmt.Errorf("Num of anchors is %[1]d, but mask values is %[2]d > %[1]d", numAnchors, op.masks[i])
		}
		currentAnchors = append(currentAnchors, op.anchors[i*2], op.anchors[i*2+1])
	}

	// Prepare reshaped input (it's common for both training and detection mode)
	err = prepareReshapedInput(inputTensor, batchSize, gridSize, bboxAttributes, len(op.masks))
	if err != nil {
		return nil, errors.Wrap(err, "Can't prepare reshaped input")
	}

	// Just inference without backpropagation in case of detection mode
	if !op.trainMode {
		return op.evaluateYOLOF32(inputTensor, batchSize, stride, gridSize, bboxAttributes, len(op.masks), currentAnchors)
	}

	// @todo Training mode
	return nil, nil
}

/* Unexported methods */

// checkInput Check if everything is OK with inputs and returns tensor.Tensor (if OK)
func (op *yoloOp) checkInput(inputs ...gorgonia.Value) (tensor.Tensor, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, errors.Wrap(err, "Can't check arity")
	}
	input := inputs[0]
	inputTensor := input.(tensor.Tensor)
	shp := inputTensor.Shape().Dims()
	if shp != 4 {
		return nil, fmt.Errorf("YOLO must contain 4 dimensions, but recieved %d)", shp)
	}
	inputNumericType := input.Dtype()
	switch inputNumericType {
	case gorgonia.Float32:
		return inputTensor, nil
	default:
		return nil, fmt.Errorf("Only Float32 supported for inputs, but got %v", inputNumericType)
	}
}

func prepareReshapedInput(input tensor.Tensor, batchSize, grid, bboxAttrs, numAnchors int) error {
	err := input.Reshape(batchSize, bboxAttrs*numAnchors, grid*grid)
	if err != nil {
		return errors.Wrap(err, "Can't make reshape grid^2 for YOLO")
	}
	err = input.T(0, 2, 1)
	if err != nil {
		return errors.Wrap(err, "Can't safely transponse input for YOLO")
	}
	err = input.Transpose()
	if err != nil {
		return errors.Wrap(err, "Can't transponse input for YOLO")
	}
	err = input.Reshape(batchSize, grid*grid*numAnchors, bboxAttrs)
	if err != nil {
		return errors.Wrap(err, "Can't reshape bbox for YOLO")
	}
	return nil
}

func (op *yoloOp) evaluateYOLOF32(input tensor.Tensor, batchSize, stride, grid, bboxAttrs, numAnchors int, currentAnchors []float32) (retVal tensor.Tensor, err error) {

	// Activation of x, y via sigmoid function
	slXY, err := input.Slice(nil, nil, Slice(0, 2))
	_, err = slXY.Apply(_sigmoidf32, tensor.UseUnsafe())
	if err != nil {
		return nil, errors.Wrap(err, "Can't activate XY")
	}

	// Activation of classes (objects) via sigmoid function
	slClasses, err := input.Slice(nil, nil, Slice(4, 5+op.numClasses))
	_, err = slClasses.Apply(_sigmoidf32, tensor.UseUnsafe())
	if err != nil {
		return nil, errors.Wrap(err, "Can't activate classes")
	}

	step := grid * numAnchors
	for i := 0; i < grid; i++ {

		vy, err := input.Slice(nil, Slice(i*step, i*step+step), Slice(1))
		if err != nil {
			return nil, errors.Wrap(err, "Can't slice while doing steps for grid")
		}

		_, err = tensor.Add(vy, float32(i), tensor.UseUnsafe())
		if err != nil {
			return nil, errors.Wrap(err, "Can't do tensor.Add(...) for float32; (1)")
		}

		for n := 0; n < numAnchors; n++ {
			anchorsSlice, err := input.Slice(nil, Slice(i*numAnchors+n, input.Shape()[1], step), Slice(0))
			if err != nil {
				return nil, errors.Wrap(err, "Can't slice anchors while doing steps for grid")
			}
			_, err = tensor.Add(anchorsSlice, float32(i), tensor.UseUnsafe())
			if err != nil {
				return nil, errors.Wrap(err, "Can't do tensor.Add(...) for float32; (1)")
			}
		}

	}

	anchors := []float32{}
	for i := 0; i < grid*grid; i++ {
		anchors = append(anchors, currentAnchors...)
	}

	anchorsTensor := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(1, grid*grid*numAnchors, 2))
	for i := range anchors {
		anchorsTensor.Set(i, anchors[i])
	}

	_, err = tensor.Div(anchorsTensor, float32(stride), tensor.UseUnsafe())
	if err != nil {
		return nil, errors.Wrap(err, "Can't do tensor.Div(...) for float32")
	}

	vhw, err := input.Slice(nil, nil, Slice(2, 4))
	if err != nil {
		return nil, errors.Wrap(err, "Can't do slice on input S(2,4)")
	}

	_, err = vhw.Apply(math32.Exp, tensor.UseUnsafe())
	if err != nil {
		return nil, errors.Wrap(err, "Can't apply exp32 to YOLO operation")
	}

	_, err = tensor.Mul(vhw, anchorsTensor, tensor.UseUnsafe())
	if err != nil {
		return nil, errors.Wrap(err, "Can't do tensor.Mul(...) for anchors")
	}

	vv, err := input.Slice(nil, nil, Slice(0, 4))
	if err != nil {
		return nil, errors.Wrap(err, "Can't do slice on input S(0,4)")
	}

	_, err = tensor.Mul(vv, float32(stride), tensor.UseUnsafe())
	if err != nil {
		return nil, errors.Wrap(err, "Can't do tensor.Mul(...) for float32")
	}

	return input, nil
}
