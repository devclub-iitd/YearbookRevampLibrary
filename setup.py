from distutils.core import setup
setup(
  name = 'YearbookCV',         
  packages = ['YearbookCV'],   
  version = '0.1',     
  description = 'Image processing library using opencv and mediapipe for YearBook of IITD',  
  author = 'Dev Club IITD',                   
  author_email = 'devclub.iitd@gmail.com',      
  url = 'https://github.com/devclub-iitd/YearbookRevampLibrary',   
  download_url = 'https://github.com/devclub-iitd/YearbookRevampLibrary/archive/refs/tags/0.2.tar.gz',    
  keywords = ['PoseDetector', 'AutoAligner', 'BackgroundRemover', 'MosaicMaker', 'CollageMaker','CropImages'],  
  install_requires=[            
            'absl-py==0.13.0',
            'attrs==21.2.0',
            'cvzone==1.3.7',
            'cycler==0.10.0',
            'kiwisolver==1.3.1',
            'matplotlib==3.4.2',
            'mediapipe==0.8.6',
            'numpy==1.21.0',
            'opencv-contrib-python==4.5.3.56',
            'opencv-python==4.5.3.56',
            'Pillow==8.3.1',
            'protobuf==3.17.3',
            'pyparsing==2.4.7',
            'python-dateutil==2.8.1',
            'six==1.16.0', 
      ],
  classifiers=[
    'Intended Audience :: Education',      
    'Topic :: Image Processing :: Utilities',
    'License :: OSI Approved :: MIT License',   
    'Programming Language :: Python :: 3.6',      
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)