# Advanced Lane Finding

Overview
---

This project follows the first (Self-Driving-Car-ND-Project-1) & improves upon it with more advanced lane detection techniques. Still based on the assumption of a fixed/mounted & centrally located front-facing camera, this project addresses the major weakness of the previous implementation for lane detection (Self-Driving-Car-ND-Project-1) which did not perform well around curves. Several image manipulation subroutines are implemented to extract useful information from individual images or frames of videos to detect lane lines, radius of curvatur & distance from the camera to the center lines of the road the vehicle is moving on.

https://user-images.githubusercontent.com/76077647/126907377-696d4f9d-d790-47e9-8b93-fce55160a876.mp4 

In oder to accompish more accurate lane detection including radius of curvature, the project implements the following subroutines:
Goals & Steps
---
* **Camera Calibration** - Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* **Image/Video Distortion Correctio** - Apply a distortion correction to raw images.
* **Color/Gradient Transformations** - Use color transforms, gradients, etc., to create a thresholded binary image.
* **Perspective Transform** - Apply a perspective transform to rectify binary image ("birds-eye view").
* **Lane Pixel Detection** - Detect lane pixels and fit to find the lane boundary.
* **Lane Curvature** - Determine the curvature of the lane and vehicle position with respect to center.
* **Warp Images** - Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


Future Improvements
---




References & Credits
---
* https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7
* https://github.com/dkarunakaran
* https://github.com/ricardosllm
