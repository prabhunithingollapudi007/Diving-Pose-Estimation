

### Run the project

To process a video, run the following command:

```bash
python .\main.py --video ../data/raw/Lou_5337D_1.5Salti_vorwaerts_3.5_Schrauben.avi --output ../data/interim/Lou.mp4
```

use the `--help` flag to see all the available options:

```bash
python .\main.py --help
```

### Project Structure

The project is structured as follows:

```plaintext

├── data
│   ├── interim
│   │   └── Lou.mp4
│   └── raw
│       └── Lou_5337D_1.5Salti_vorwaerts_3.5_Schrauben.avi
├── main.py
├── README.md
└── requirements.txt

```

### Data

The `data` directory contains two subdirectories:

- `raw`: Contains the raw video file `Lou_5337D_1.5Salti_vorwaerts_3.5_Schrauben.avi`.
- `interim`: Will contain the processed video file `Lou.mp4`.

### `main.py`

The `main.py` script is the main entry point for the project. It processes the input video file and saves the output to the specified location.


### `requirements.txt`

The `requirements.txt` file lists all the Python dependencies required to run the project. You can install them using `pip`:

```bash

pip install -r requirements.txt

```

### Questions

1. Is the camera angle going to be fixed?
2. Background to be removed or not?
3. How many frames per second are required?
4. Video format to be used (input and output)?
5. Multiple people in the video, how to handle that?

### Conclusion

In this project, we have processed a video file to create a new video with rotated frames. This can be useful for scenarios where the original video was recorded at an incorrect angle or orientation. By rotating the frames, we can correct the orientation of the video and make it more visually appealing. This project demonstrates how to read a video file, process its frames, and save the output to a new video file. The code can be easily modified to handle different video formats, frame rotations, and other processing tasks. By understanding the basics of video processing, you can build more advanced applications that involve video analysis, object detection, and other computer vision tasks.