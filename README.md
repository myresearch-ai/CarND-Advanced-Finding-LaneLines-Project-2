# Advanced Lane Finding


Overview
---
This project follows the first (Self-Driving-Car-ND-Project-1) & improves upon it with more advanced lane detection techniques. Still based on the assumption of a fixed/mounted & centrally located front-facing camera, this project addresses the major weakness of the previous implementation for lane detection (Self-Driving-Car-ND-Project-1) which did not perform well around curves. Several image manipulation subroutines are implemented to extract useful information from individual images or frames of videos to detect lane lines, radius of curvatur & distance from the camera to the center lines of the road the vehicle is moving on.



Goals & Steps
---
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


Future Improvements
---




References & Credits
---
* https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7
* https://github.com/dkarunakaran
* https://github.com/ricardosllm
