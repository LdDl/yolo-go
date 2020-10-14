package yologo

import (
	"fmt"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestYolo(t *testing.T) {

	inputSize := 416
	numClasses := 80
	testAnchors := [][]float32{
		[]float32{10, 13, 16, 30, 33, 23},
		[]float32{30, 51, 62, 45, 59, 119},
		[]float32{116, 90, 156, 198, 373, 326},
	}

	numpyInputs := []string{
		"./test_yolo_op_data/1input.[(10, 13), (16, 30), (33, 23)].npy",
		"./test_yolo_op_data/1input.[(30, 61), (62, 45), (59, 119)].npy",
		"./test_yolo_op_data/1input.[(116, 90), (156, 198), (373, 326)].npy",
	}

	numpyExpectedOutputs := []string{
		"./test_yolo_op_data/1output.[(10, 13), (16, 30), (33, 23)].npy",
		"./test_yolo_op_data/1output.[(30, 61), (62, 45), (59, 119)].npy",
		"./test_yolo_op_data/1output.[(116, 90), (156, 198), (373, 326)].npy",
	}

	for i := range testAnchors {
		// Read input values from numpy format
		input := tensor.New(tensor.Of(tensor.Float32))
		r, err := os.Open(numpyInputs[i])
		if err != nil {
			t.Error(err)
			return
		}
		err = input.ReadNpy(r)
		if err != nil {
			t.Error(err)
			return
		}

		// Read expected values from numpy format
		expected := tensor.New(tensor.Of(tensor.Float32))
		r, err = os.Open(numpyExpectedOutputs[i])
		if err != nil {
			t.Error(err)
			return
		}
		err = expected.ReadNpy(r)
		if err != nil {
			t.Error(err)
			return
		}

		// Load graph
		g := gorgonia.NewGraph()
		inputTensor := gorgonia.NewTensor(g, tensor.Float32, 4, gorgonia.WithShape(input.Shape()...), gorgonia.WithName("yolo"))
		// Prepare YOLOv3 node
		outNode, _, err := YOLOv3Node(inputTensor, testAnchors[i], []int{0, 1, 2}, inputSize, numClasses, 0.7)
		if err != nil {
			t.Error(err)
			return
		}
		// Run operation
		vm := gorgonia.NewTapeMachine(g)
		if err := gorgonia.Let(inputTensor, input); err != nil {
			t.Error(err)
			return
		}
		vm.RunAll()
		vm.Close()

		// Check if everything is fine
		if !assert.Equal(t, outNode.Value().Data(), expected.Data(), "Output is not equal to expected value") {
			t.Error(fmt.Sprintf("Got: %v\nExpected: %v", outNode.Value(), expected))
		}
	}
}
