package yologo

import (
	"gorgonia.org/gorgonia"
)

type layerN interface {
	String() string
	Type() string
	ToNode(g *gorgonia.ExprGraph, inputs ...*gorgonia.Node) (*gorgonia.Node, error)
}
