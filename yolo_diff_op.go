package yologo

import (
	"fmt"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Wrap yoloOp
type yoloDiffOp struct {
	yoloOp
}

/* Methods to match gorgonia.Op interface */
func (op *yoloDiffOp) Arity() int { return 2 }
func (op *yoloDiffOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := gorgonia.TensorType{Dims: 4, Of: a}
	o := gorgonia.TensorType{Dims: 3, Of: a}
	return hm.NewFnType(t, o, t)
}

func (op *yoloDiffOp) ReturnsPtr() bool     { return true }
func (op *yoloDiffOp) CallsExtern() bool    { return false }
func (op *yoloDiffOp) OverwritesInput() int { return -1 }
func (op *yoloDiffOp) InferShape(inputs ...gorgonia.DimSizer) (tensor.Shape, error) {
	s := inputs[0].(tensor.Shape).Clone()
	return s, nil
}
func (op *yoloDiffOp) Do(inputs ...gorgonia.Value) (gorgonia.Value, error) {
	if op.training == nil {
		return nil, fmt.Errorf("Training parameters for yoloOp were not set")
	}
	if op.training.inputs == nil {
		return nil, fmt.Errorf("Training parameter 'inputs' for yoloOp were not set")
	}
	if op.training.scales == nil {
		return nil, fmt.Errorf("Training parameter 'scales' for yoloOp were not set")
	}
	if op.training.targets == nil {
		return nil, fmt.Errorf("Training parameter 'targets' for yoloOp were not set")
	}
	if op.training.bboxes == nil {
		return nil, fmt.Errorf("Training parameter 'bboxes' for yoloOp were not set")
	}

	in := inputs[0]
	output := inputs[1]
	inGrad := tensor.New(tensor.Of(in.Dtype()), tensor.WithShape(output.Shape().Clone()...), tensor.WithEngine(in.(tensor.Tensor).Engine()))
	switch in.Dtype() {
	case tensor.Float32:
		inGradData := inGrad.Data().([]float32)
		outGradData := output.Data().([]float32)
		op.f32(inGradData, outGradData, op.training.scales, op.training.inputs, op.training.targets, op.training.bboxes)
		break
	case tensor.Float64:
		return nil, fmt.Errorf("yoloDiffOp for Float64 is not implemented yet")
	default:
		return nil, fmt.Errorf("yoloDiffOp supports only Float32/Float64 types")
	}

	err := inGrad.Reshape(1, op.gridSize*op.gridSize, (op.numClasses+5)*len(op.masks))
	if err != nil {
		return nil, errors.Wrap(err, "Can't reshape in yoloDiffOp (1)")
	}
	err = inGrad.T(0, 2, 1)
	if err != nil {
		return nil, errors.Wrap(err, "Can't safely transponse in yoloDiffOp (1)")
	}
	err = inGrad.Transpose()
	if err != nil {
		return nil, errors.Wrap(err, "Can't transponse in yoloDiffOp (1)")
	}
	err = inGrad.Reshape(1, len(op.masks)*(5+op.numClasses), op.gridSize, op.gridSize)
	if err != nil {
		return nil, errors.Wrap(err, "Can't reshape in yoloDiffOp (2)")
	}
	return inGrad, nil
}

func (op *yoloOp) DoDiff(ctx gorgonia.ExecutionContext, inputs gorgonia.Nodes, output *gorgonia.Node) (err error) {
	return fmt.Errorf("DoDiff for yoloOp is not implemented")
}
func (op *yoloOp) DiffWRT(inputs int) []bool { return []bool{true} }
func (op *yoloOp) SymDiff(inputs gorgonia.Nodes, output, grad *gorgonia.Node) (retVal gorgonia.Nodes, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}
	in := inputs[0]
	var op2 yoloOp
	op2 = *op
	diff := &yoloDiffOp{op2}

	var ret *gorgonia.Node
	if ret, err = gorgonia.ApplyOp(diff, in, grad); err != nil {
		return nil, err
	}
	return gorgonia.Nodes{ret}, nil
}

/* Unexported methods */

func (op *yoloDiffOp) f32(inGradData, outGradData, scales, inputs, targets, bboxes []float32) {
	for i := range inGradData {
		inGradData[i] = 0
	}
	for i := 0; i < len(outGradData); i = i + 5 + op.numClasses {
		for j := 0; j < 4; j++ {
			inGradData[i+j] = outGradData[i+j] * (scales[i+j] * scales[i+j] * (inputs[i+j] - targets[i+j]))
		}
		for j := 4; j < 5+op.numClasses; j++ {
			if outGradData[i+j] != 0 {
				if targets[i+j] == 0 {
					inGradData[i+j] = outGradData[i+j] * (bboxes[i+j])
				} else {
					inGradData[i+j] = outGradData[i+j] * (1 - bboxes[i+j])
				}
			}
		}
	}
}
