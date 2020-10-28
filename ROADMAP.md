# ROADMAP

New ideas, thought about needed features will be stored in this file.

- [x] Darknet YOLO v3 layers
    - [x] Convolution layer;
    - [x] Maxpool layer;
    - [x] Upsample layer;
    - [x] Route layer;
    - [x] Shortcut layer;
    - [x] YOLO layer;
- [x] Utilities
    - [x] Darknet-based weights parser
    - [x] Darknet-based neural network configurtion file parser
    - [x] NMS for detected objects and all corresponding functions such as IOU, bbox rectifying
    - [x] Convert image to slice of float32
        - [x] Resize image and average color picker
        - [x] Pixel extracting and adapting it for []float32
- [x] New Gorgonia operations (or modified existing ones):
    - [x] Upsample and its derivative
    - [x] YOLO and its derivative
- [ ] Test coverage (Its always good)
- [x] Full inference example
- [x] Training **WIP**
    - [ ] Loss function **WIP, PRs are welcome**
    - [ ] Proper backpropagation **WIP, PRs are welcome**
- [ ] Optimizations (replace 'raw loop' in code with Gorgonia core functions)
- [ ] GPU (Here we have much work to do)
    - [ ] Instructions
    - [ ] Code itself (with Gorgonia CUDA-based library)
    - [ ] Benchmarks
- [ ] Documentation (fill [README.md](README.md) with usefull stuff) **WIP**

Updated at: 2020-10-28