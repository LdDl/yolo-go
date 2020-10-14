# WIP
Port of [Darknet YOLO](https://github.com/pjreddie/darknet#darknet), but via [Gorgonia](https://github.com/gorgonia/gorgonia)



# Usage
Navigate to [examples/tiny-v3](examples/tiny-v3) folder and run [main.go](examples/tiny-v3/main.go).

For testing:
```shell
go run main.go --mode detector --cfg ../../test_network_data/yolov3-tiny.cfg --weights ../../test_network_data/yolov3-tiny.weights --image ../../test_network_data/dog_416x416.jpg
```