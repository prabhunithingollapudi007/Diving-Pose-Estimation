15 Aug 2024

1. MediaPipe
Live Performance: Yes, MediaPipe is designed specifically for real-time performance. It works efficiently on CPUs and can run at high frame rates, making it ideal for live applications.
Use Case: It is well-suited for live streaming, real-time monitoring, and applications where low latency is critical.
2. OpenPose with GPU Acceleration
Live Performance: Yes, OpenPose can work in real-time if it’s run on a capable GPU. On a powerful GPU (e.g., NVIDIA GTX/RTX series), it can process video frames quickly enough for live applications.
Use Case: Suitable for scenarios where high accuracy is needed and a GPU is available, like live sports analysis or motion capture in studios.
3. TensorFlow.js (PoseNet/MoveNet)
Live Performance: Yes, TensorFlow.js, when used with models like PoseNet or MoveNet, can perform real-time pose estimation directly in the browser. It’s designed to work on live webcam feeds with low latency.
Use Case: Ideal for web-based applications where the user interacts with a camera in real-time, such as browser-based fitness apps or live video filters.
4. MoveNet
Live Performance: Yes, MoveNet is specifically designed for real-time applications and can run on CPUs, GPUs, and even mobile devices. The Lightning version is optimized for speed, making it suitable for live use.
Use Case: Excellent for mobile apps, embedded systems, or any scenario where very low latency is required, like real-time fitness tracking or live gesture control.
Key Points on Live Performance:
MediaPipe and MoveNet are the best choices if you need reliable real-time performance on a range of devices, from desktops to mobile.
OpenPose can also run live but usually requires more powerful hardware (like a good GPU) to maintain high frame rates.
TensorFlow.js is great if you're developing web-based applications that require real-time pose estimation directly in the browser.


21 Aug 2024

Creating frames from video format so that it can be used in the model.
Frames are rotated and saved.

Questions:
1. Is the camera angle going to be fixed ?
2. Background to be removed or not ?
3. How many frames per second are required ?
4. Video format to be used (input and output)?



29 aug 2024

1. Work on ideal video
2. Change brightness, contrast, saturation to highlight the person in the video
3. Remove background
4. RTMO and RTM check time complexities


04 Sep 2024

1. Drawing joints on ideal yoga video