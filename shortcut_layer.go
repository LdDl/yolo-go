package yologo

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
)

type shortcutLayer struct {
	layerIDX int
}

func (l *shortcutLayer) String() string {
	return fmt.Sprintf("Shortcut layer: index->%[1]d", l.layerIDX)
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
