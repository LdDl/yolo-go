package yologo

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
)

type shortcutLayer struct {
	layerFromIdx int
	layerToIdx   int
}

func (l *shortcutLayer) String() string {
	return fmt.Sprintf("Shortcut layer: index from->%[1]d | index to->%[2]d", l.layerFromIdx, l.layerToIdx)
}

func (l *shortcutLayer) Type() string {
	return "shortcut"
}

func (l *shortcutLayer) ToNode(g *gorgonia.ExprGraph, inputs ...*gorgonia.Node) (*gorgonia.Node, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("Shortcut layer can accept only two nodes, but got %d", len(inputs))
	}

	addNode, err := gorgonia.Add(inputs[0], inputs[1])
	if err != nil {
		return &gorgonia.Node{}, errors.Wrap(err, "Can't prepare shortcut operation")
	}

	return addNode, nil
}
