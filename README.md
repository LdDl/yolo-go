# WIP
Port of [Darknet YOLO](https://github.com/pjreddie/darknet#darknet), but via [Gorgonia](https://github.com/gorgonia/gorgonia)
**Note: do not try to use common YOLOv3, because shortcut layer is not implemented yet**

# Usage

Navigate to [examples/tiny-v3](examples/tiny-v3) folder and run [main.go](examples/tiny-v3/main.go).

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

For testing:
```shell
go run main.go --mode detector --cfg ../../test_network_data/yolov3-tiny.cfg --weights ../../test_network_data/yolov3-tiny.weights --image ../../test_network_data/dog_416x416.jpg
```

For training:
```shell
go run main.go --mode training --cfg ../../test_network_data/yolov3-tiny.cfg --weights ../../test_network_data/yolov3-tiny.weights --image ../../test_network_data/dog_416x416.jpg --train ../../test_yolo_op_data
```

# Tiny YOLOv3 Architecture
Architecture is:
```
0 Convolutional 16 3 × 3/1 416 × 416 × 3 416 × 416 × 16
1 Maxpool    2 × 2/2 416 × 416 × 16 208 × 208 × 16
2 Convolutional 32 3 × 3/1 208 × 208 × 16 208 × 208 × 32
3 Maxpool    2 × 2/2 208 × 208 × 32 104 × 104 × 32
4 Convolutional 64 3 × 3/1 104 × 104 × 32 104 × 104 × 64
5 Maxpool    2 × 2/2 104 × 104 × 64 52 × 52 × 64
6 Convolutional 128 3 × 3/1 52 × 52 × 64 52 × 52 × 128
7 Maxpool    2 × 2/2 52 × 52 × 128 26 × 26 × 128
8 Convolutional 256 3 × 3/1 26 × 26 × 128 26 × 26 × 256
9 Maxpool    2 × 2/2 26 × 26 × 256 13 × 13 × 256
10 Convolutional 512 3 × 3/1 13 × 13 × 256 13 × 13 × 512
11 Maxpool    2 × 2/1 13 × 13 × 512 13 × 13 × 512
12 Convolutional 1024 3 × 3/1 13 × 13 × 512 13 × 13 × 1024
13 Convolutional 256 1 × 1/1 13 × 13 × 1024 13 × 13 × 256
14 Convolutional 512 3 × 3/1 13 × 13 × 256 13 × 13 × 512
15 Convolutional 255 1 × 1/1 13 × 13 × 512 13 × 13 × 255
16 YOLO        
17 Route 13       
18 Convolutional 128 1 × 1/1 13 × 13 × 256 13 × 13 × 128
19 Up‐sampling    2 × 2/1 13 × 13 × 128 26 × 26 × 128
20 Route 19 8       
21 Convolutional 256 3 × 3/1 13 × 13 × 384 13 × 13 × 256
22 Convolutional 255 1 × 1/1 13 × 13 × 256 13 × 13 × 256
23 YOLO 
```