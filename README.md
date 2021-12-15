
# YearbookRevampLibrary
This repository contains the development code for the YearbookRevampLibrary package

## Package installation:
1. Open the command prompt
2. Run `pip install git+https://github.com/devclub-iitd/YearbookRevampLibrary` in the cmd prompt

## Various Functions:
* **To Remove Background:**

  Copy the path where the photos are present (Only photos must be present in the respective folder) - `input_path` ;`cv2_list` can also pass list of cv2 objects;  specify where you want the images with background removed to get stored - `output_path`[Also returns list of cv2 objects] ; the background color to set in form of color_channels eg. - `(255,255,255)`, `model` type 0 or 1 - 0 is general,  1 is landscape (faster) ; `threshold` (between 0 and 1) to define the degree of accurate background cuts (more the threshold more will be the time taken) and then execute the following code -
  ```python
  from YearbookRevampLibrary.BackgroundModule import remove_background
  remove_background(r'cv2_list',r'input_path', r'output_path', color_channels, model, threshold)
  ##default :- color_channels - (255,255,255),  model - 0, threshold - 0.1
  ```

* **To Crop Images:**
  You can choose from 3 different versions:
  1. Pass the path of a single image file - `img_file_src` ;specify the `size` of the image as tuple of (length,breadth); specify the `output_path` [Also will return image as cv2 file]; specify `make_circle`, True od False to make individual images circle ;  `remove_background` specify to remove background or not [Default is False];can pass cv2 object as`img_file`[Default is None] and then execute the following code-
  ```python
  from YearbookRevampLibrary.Crop import CropBody
  CropBody(r'input_path', r'size', r'output_path', r'makeCircle',r'removeBackground',r'img_file')
  ```

  2. Pass the path of a single image file - `img_file_src` ;specify the `size` of the image as  tuple of (length,breadth); specify the `output_path` [Also will return image as cv2 file]; specify `make_circle`, True or False to make individual images circle;can pass cv2 object as`img_file`[Default is None] and then execute the following code-
  ```python
  from YearbookRevampLibrary.Crop import CropFace
  CropFace(r'input_path', r'size', r'output_path', r'makeCircle',r'img_file')
  ```

  3.  Copy the path where the photos are present (Only photos must be present in the respective folder) - `input_path`;`cv2_list` can also pass list of cv2 objects; specify the `output_path`[Also returns list of cv2 objects] ; specify the `type` of the collage as 0,1 for body crop or face crop [Default is 0]; specify `make_circle`, True od False to make individual images circle [Default is True] ;  `remove_background` specify to remove background or not [Default is False] and then execute the following code -
  ```python
  from YearbookRevampLibrary.Crop import CropAll
  CropAll(r'cv2_list', r'input_path', r'output_path', r'type', r'makeCircle',r'removeBackground')
  ```


* **To Auto-Align Images:**

  Copy the path where the photos are present (Only photos must be present in the respective folder) - `input_path` ; `cv2_list` can also pass list of cv2 objects;specify where you want the images with background removed to get stored - `output_path` [Also returns list of cv2 objects]; `min_face_detection_confidence` ; `min_pose_detection_confidence` and then execute the following code -
  ```python
  from YearbookRevampLibrary.AutoAlignerModule import auto_align
  auto_align(r'cv2_list',r'input_path', r'output_path', min_face_detection_confidence, min_pose_detection_confidence)
  ##default :- min_face_detection_confidence - 0.5, min_pose_detection_confidence - 0,5
  ```
* **To Create Collage:**
  You can choose from 3 different collages

  1. Copy the path where the photos are present (Only photos must be present in the respective folder) - `input_path` ; specify the final collage `file_name` with the required extension ; `width` ; `height` and then execute the following code -
  ```python
  from YearbookRevampLibrary.CollageModule import make_collage
  make_collage(r'input_path', 'file_name', width, height)
  ```

  2. Copy the path where the photos are present (Only photos must be present in the respective folder) - `input_path` ;`cv2_list` can also pass list of cv2 objects; specify the final collage `file_name` with the required extension ; specify the `output_path`[Also returns cv2 object]; specify a template image `template_file` which is a black an white image where white color are parts where circles should be made(Eg: A black background with IIT in white written willl make a collage that resembles this text) and then execute the following code - -
  ```python
  from YearbookRevampLibrary.CircleCollage import MakeCircleCollage
  MakeCircleCollage(r'template_file',r'cv2_list',r'input_path', r'output_path',r'file_name'):
  ```

  3. Copy the path where the photos are present (Only photos must be present in the respective folder) - `input_path`; `cv2_list` can also pass list of cv2 objects; specify the `output_path` [Also returns cv2 object]; specify the `size` of the collage as int (length of collage square edge); specify `make_circle`, True od False to make individual images circle ; specify `array` as numpy array with 0,1 where to display image or not [Default takes and 8x8 array of ones];  `remove_background` specify to remove background or not [Default is False];   specify the final collage `file_name` with the required extension and then execute the following code -
  ```python
  from YearbookRevampLibrary.SimpleCollage import SimpleCollage
  SimpleCollage(r'cv2_list',r'input_path',r'output_path', r'size', r'array', r'removeBackground',r'filename')
  ```

* **To Apply Cartoon Filter:**
  You can choose from 2 different filters
  Copy the path where the photos are present (Only photos must be present in the respective folder) - `input_path` ; `cv2_list` can also pass list of cv2 objects; specify the `output_path`[Also returns list of cv2 objects] and then execute the following code -
  ```python
  from YearbookRevampLibrary.Cartoon import CartoonFilter
  CartoonFilter(r'cv2_list',r'input_path', r'output_path')
  ```
  or 
  ```python
  from YearbookRevampLibrary.Cartoon import BlurredCartoonFilter
  BlurredCartoonFilter(r'input_path', r'output_path')
  ```


* **Mosaic Creator**
  This function named  ```CreateMosaic```  is used to create mosaic of a given target image using multiple images.

  It takes in arguments in the following order:</br>
    ```target_image```: the image whose mosaic is to be generated (Path of the image has to be entered in this parameter).
    
    ```input_images```: the list/folder of the images that we want to use to generate the mosaic.
    
    ```grid_size```: tuple containing the number of images we want along the height and breadth of mosaic respectively.
    
    ```output_filename```: string containing the filename of the generated mosaic
    
    ```output_path```: string containing the required path where the generated mosaic is to be saved (optional argument)
    
    ```is_folder_input```: a boolean specify whether the input images are passed to the function as a folder or list (True means a folder is passed, default value is set to False)
    
    ```reuse_images```: a boolean to specify whether we want input images to be reused to create the mosaic(optional argument , by default set to True )

  The grid size is a tuple and contains number of rows and columns respectively in the given mosaic.</br>
  For example : grid size of (25,40) means 25 images along the height and 40 images along the breadth of the mosaic. So, the resulting mosaic will comprise of 25*40 images.
   
  This function returns the generated mosaic image and it can also be saved by entering the output_path and ouput_filename in the parameters.

  In order to implemenet this function just use the following code:
  ```python
  import YearbookRevampLibrary
  YearbookRevampLibrary.CreateMosaic(target_image,input_images,grid_size, output_filename, output_path, is_folder_input,reuse_images)
  ```
  *Note that the output image is generated in jpeg format.*

