package yologo

import (
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"math"
	"os"

	"github.com/chewxy/math32"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Slice Just alias
var Slice = gorgonia.S

func rectifyBoxF32(x, y, h, w float32, imgSize int) image.Rectangle {
	return image.Rect(MaxInt(int(x-w/2), 0), MaxInt(int(y-h/2), 0), MinInt(int(x+w/2+1), imgSize), MinInt(int(y+h/2+1), imgSize))
}

func findIntElement(arr []int, ele int) int {
	for i := range arr {
		if arr[i] == ele {
			return i
		}
	}
	return -1
}

func tensorToF32(in tensor.Tensor) (input32 []float32, err error) {
	in.Reshape(in.Shape().TotalSize())
	input32 = make([]float32, 0)
	input32 = make([]float32, in.Shape().TotalSize())
	for i := 0; i < in.Shape()[0]; i++ {
		var buf interface{}
		buf, err = in.At(i)
		switch in.Dtype() {
		case tensor.Float32:
			input32[i] = buf.(float32)
			break
		case tensor.Float64:
			input32[i] = float32(buf.(float64))
			break
		default:
			return nil, fmt.Errorf("convertTensorToFloat32() supports only Float32/Float64 types of tensor")
		}
	}
	return input32, nil
}

// IOUFloat32 Intersection Over Union for float32
func IOUFloat32(r1, r2 image.Rectangle) float32 {
	intersection := r1.Intersect(r2)
	interArea := intersection.Dx() * intersection.Dy()
	r1Area := r1.Dx() * r1.Dy()
	r2Area := r2.Dx() * r2.Dy()
	return float32(interArea) / float32(r1Area+r2Area-interArea)
}

func getBestIOUF32(input, target []float32, numClasses, dims int) [][]float32 {
	ious := make([][]float32, 0)
	imgsize := float32(dims)
	for i := 0; i < len(input); i = i + numClasses + 5 {
		ious = append(ious, []float32{0, -1})
		r1 := rectifyBoxF32(input[i], input[i+1], input[i+2], input[i+3], dims)
		for j := 0; j < len(target); j = j + 5 {
			r2 := rectifyBoxF32(target[j+1]*imgsize, target[j+2]*imgsize, target[j+3]*imgsize, target[j+4]*imgsize, dims)
			curiou := IOUFloat32(r1, r2)
			if curiou > ious[i/(5+numClasses)][0] {
				ious[i/(5+numClasses)][0] = curiou
				ious[i/(5+numClasses)][1] = float32(j / 5)
			}
		}
	}
	return ious
}

// Rectify Creates rectangle
func Rectify(x, y, h, w, maxwidth, maxheight int) image.Rectangle {
	return image.Rect(MaxInt(x-w/2, 0), MaxInt(y-h/2, 0), MinInt(x+w/2+1, maxwidth), MinInt(y+h/2+1, maxheight))
}

// MaxInt Maximum between two integers
func MaxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// MinInt Minimum between two integers
func MinInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// arityer Duck typing for arity check
type arityer interface {
	Arity() int
}

func checkArity(op arityer, inputs int) error {
	if inputs != op.Arity() && op.Arity() >= 0 {
		return fmt.Errorf("%v has an arity of %d. Got %d instead", op, op.Arity(), inputs)
	}
	return nil
}

func _sigmoidf32(x float32) float32 {
	if x < -88 {
		return 0
	}
	if x > 15 {
		return 1
	}
	return 1.0 / (1.0 + math32.Exp(-x))
}

// Float32frombytes Converts []byte to float32
func Float32frombytes(bytes []byte) float32 {
	bits := binary.LittleEndian.Uint32(bytes)
	float := math.Float32frombits(bits)
	return float
}

// GetFloat32Image Returns []float32 representation of image file
func GetFloat32Image(fname string, resizeWidth, resizeHeight int) ([]float32, error) {
	file, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	img, err := jpeg.Decode(file)
	if err != nil {
		return nil, err
	}
	imgResized := resizeImage(img, resizeWidth, resizeHeight)
	return Image2Float32(imgResized)
}

// Image2Float32 Returns []float32 representation of image.Image
func Image2Float32(img image.Image) ([]float32, error) {
	channelsNum := 3 // Static for RGB
	width := img.Bounds().Dx()
	height := img.Bounds().Dy()
	imgwh := width * height
	imgSize := imgwh * channelsNum
	ans := make([]float32, imgSize)
	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			r, g, b, _ := img.At(y, x).RGBA()
			rpix, gpix, bpix := float32(r>>8)/float32(255.0), float32(g>>8)/float32(255.0), float32(b>>8)/float32(255.0)
			ans[y+x*height] = rpix
			ans[y+x*height+imgwh] = gpix
			ans[y+x*height+imgwh+imgwh] = bpix
		}
	}
	return ans, nil
}

// Naive image resizing. See ref. https://stackoverflow.com/a/56411381
func resizeImage(img image.Image, width int, height int) image.Image {
	minX := img.Bounds().Min.X
	minY := img.Bounds().Min.Y
	maxX := img.Bounds().Max.X
	maxY := img.Bounds().Max.Y
	for (maxX-minX)%height != 0 {
		maxX--
	}
	for (maxY-minY)%width != 0 {
		maxY--
	}
	scaleX := (maxX - minX) / height
	scaleY := (maxY - minY) / width
	imgRect := image.Rect(0, 0, height, width)
	resImg := image.NewRGBA(imgRect)
	draw.Draw(resImg, resImg.Bounds(), &image.Uniform{C: color.White}, image.ZP, draw.Src)
	for y := 0; y < width; y++ {
		for x := 0; x < height; x++ {
			averageColor := getAverageColor(img, minX+x*scaleX, minX+(x+1)*scaleX, minY+y*scaleY, minY+(y+1)*scaleY)
			resImg.Set(x, y, averageColor)
		}
	}
	return resImg
}

func getAverageColor(img image.Image, minX int, maxX int, minY int, maxY int) color.Color {
	var averageRed float64
	var averageGreen float64
	var averageBlue float64
	var averageAlpha float64
	scale := 1.0 / float64((maxX-minX)*(maxY-minY))

	for i := minX; i < maxX; i++ {
		for k := minY; k < maxY; k++ {
			r, g, b, a := img.At(i, k).RGBA()
			averageRed += float64(r) * scale
			averageGreen += float64(g) * scale
			averageBlue += float64(b) * scale
			averageAlpha += float64(a) * scale
		}
	}
	averageRed = math.Sqrt(averageRed)
	averageGreen = math.Sqrt(averageGreen)
	averageBlue = math.Sqrt(averageBlue)
	averageAlpha = math.Sqrt(averageAlpha)
	averageColor := color.RGBA{R: uint8(averageRed), G: uint8(averageGreen), B: uint8(averageBlue), A: uint8(averageAlpha)}
	return averageColor
}
