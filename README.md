# WIP. PRs are welcome
Port of [Darknet YOLO](https://github.com/pjreddie/darknet#darknet), but via [Gorgonia](https://github.com/gorgonia/gorgonia)
Both YOLOv3 and tiny-YOLOv3 are implemented.

# Usage

Navigate to [examples/yolo-v3](examples/yolo-v3) folder and run [main.go](examples/yolo-v3/main.go).

Available flags 
```go
go run main.go -h
```
```shell
  -cfg string
        Path to net configuration file (default "../../test_network_data/yolov3-tiny.cfg")
  -image string
        Path to image file for 'detector' mode (default "../../test_network_data/dog_416x416.jpg")
  -mode string
        Choose the mode: detector/training (default "detector")
  -train string
        Path to folder with labeled data (default "../../test_yolo_op_data")
  -weights string
        Path to weights file (default "../../test_network_data/yolov3-tiny.weights")
```

For testing tiny-yolov3:
```shell
go run main.go --mode detector --cfg ../../test_network_data/yolov3-tiny.cfg --weights ../../test_network_data/yolov3-tiny.weights --image ../../test_network_data/dog_416x416.jpg
```

For testing yolov3:
```shell
go run main.go --mode detector --cfg ../../test_network_data/yolov3.cfg --weights ../../test_network_data/yolov3.weights --image ../../test_network_data/dog_416x416.jpg
```

For training **WIP. PRs are welcome**:
```shell
go run main.go --mode training --cfg ../../test_network_data/yolov3-tiny.cfg --weights ../../test_network_data/yolov3-tiny.weights --image ../../test_network_data/dog_416x416.jpg --train ../../test_yolo_op_data
```

# Weights and configuration
Weights can be downloaded via curl-scripts [download_weights_yolo_v3.sh](test_network_data/download_weights_yolo_v3.sh) and [download_weights_yolo_tiny_v3.sh](test_network_data/download_weights_yolo_tiny_v3.sh).
Configuration files: [yolov3-tiny.cfg](test_network_data/yolov3-tiny.cfg) and [yolov3.cfg](test_network_data/yolov3.cfg)

# Network Architecture
## Tiny-YOLOv3 Architecture is:
```
Convolution layer: Filters->16 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Maxpooling layer: Size->2 Stride->2
Convolution layer: Filters->32 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Maxpooling layer: Size->2 Stride->2
Convolution layer: Filters->64 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Maxpooling layer: Size->2 Stride->2
Convolution layer: Filters->128 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Maxpooling layer: Size->2 Stride->2
Convolution layer: Filters->256 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Maxpooling layer: Size->2 Stride->2
Convolution layer: Filters->512 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Maxpooling layer: Size->2 Stride->1
Convolution layer: Filters->1024 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->256 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->512 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->255 Padding->0 Kernel->1x1 Stride->1 Activation->linear Batch->0 Bias->true
YOLO layer: Mask->3 Anchors->[81, 82]   |       Mask->4 Anchors->[135, 169]     |       Mask->5 Anchors->[344, 319]
Route layer: Start->13
Convolution layer: Filters->128 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Upsample layer: Scale->2
Route layer: Start->19 End->8
Convolution layer: Filters->256 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->255 Padding->0 Kernel->1x1 Stride->1 Activation->linear Batch->0 Bias->true
YOLO layer: Mask->0 Anchors->[10, 14]   |       Mask->1 Anchors->[23, 27]       |       Mask->2 Anchors->[37, 58] 
```

## YOLOv3 Architecture is:
```
Convolution layer: Filters->32 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->64 Padding->1 Kernel->3x3 Stride->2 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->32 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->64 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Shortcut layer: index->-3
Convolution layer: Filters->128 Padding->1 Kernel->3x3 Stride->2 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->64 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->128 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Shortcut layer: index->-3
Convolution layer: Filters->64 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->128 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Shortcut layer: index->-3
Convolution layer: Filters->256 Padding->1 Kernel->3x3 Stride->2 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->128 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->256 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Shortcut layer: index->-3
Convolution layer: Filters->128 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->256 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Shortcut layer: index->-3
Convolution layer: Filters->128 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->256 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Shortcut layer: index->-3
Convolution layer: Filters->128 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->256 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Shortcut layer: index->-3
Convolution layer: Filters->128 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->256 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Shortcut layer: index->-3
Convolution layer: Filters->128 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->256 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Shortcut layer: index->-3
Convolution layer: Filters->128 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->256 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Shortcut layer: index->-3
Convolution layer: Filters->128 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->256 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Shortcut layer: index->-3
Convolution layer: Filters->512 Padding->1 Kernel->3x3 Stride->2 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->256 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->512 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Shortcut layer: index->-3
Convolution layer: Filters->256 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->512 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Shortcut layer: index->-3
Convolution layer: Filters->256 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->512 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Shortcut layer: index->-3
Convolution layer: Filters->256 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->512 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Shortcut layer: index->-3
Convolution layer: Filters->256 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->512 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Shortcut layer: index->-3
Convolution layer: Filters->256 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->512 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Shortcut layer: index->-3
Convolution layer: Filters->256 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->512 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Shortcut layer: index->-3
Convolution layer: Filters->256 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->512 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Shortcut layer: index->-3
Convolution layer: Filters->1024 Padding->1 Kernel->3x3 Stride->2 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->512 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->1024 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Shortcut layer: index->-3
Convolution layer: Filters->512 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->1024 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Shortcut layer: index->-3
Convolution layer: Filters->512 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->1024 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Shortcut layer: index->-3
Convolution layer: Filters->512 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->1024 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Shortcut layer: index->-3
Convolution layer: Filters->512 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->1024 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->512 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->1024 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->512 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->1024 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->255 Padding->0 Kernel->1x1 Stride->1 Activation->linear Batch->0 Bias->true
YOLO layer: Mask->6 Anchors->[116, 90]  |       Mask->7 Anchors->[156, 198]     |       Mask->8 Anchors->[373, 326]
Route layer: Start->79
Convolution layer: Filters->256 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Upsample layer: Scale->2
Route layer: Start->85 End->61
Convolution layer: Filters->256 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->512 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->256 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->512 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->256 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->512 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->255 Padding->0 Kernel->1x1 Stride->1 Activation->linear Batch->0 Bias->true
YOLO layer: Mask->3 Anchors->[30, 61]   |       Mask->4 Anchors->[62, 45]       |       Mask->5 Anchors->[59, 119]
Route layer: Start->91
Convolution layer: Filters->128 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Upsample layer: Scale->2
Route layer: Start->97 End->36
Convolution layer: Filters->128 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->256 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->128 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->256 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->128 Padding->0 Kernel->1x1 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->256 Padding->1 Kernel->3x3 Stride->1 Activation->leaky Batch->1 Bias->false
Convolution layer: Filters->255 Padding->0 Kernel->1x1 Stride->1 Activation->linear Batch->0 Bias->true
YOLO layer: Mask->0 Anchors->[10, 13]   |       Mask->1 Anchors->[16, 30]       |       Mask->2 Anchors->[33, 23]
```