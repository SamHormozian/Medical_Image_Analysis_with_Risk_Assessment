# Medical_Image_Analysis_with_Risk_Assessment

## About + Thought process

### **Goal:**
- Make a system that detects skin cancer in patients(detects melanoma vs benign lesions, classify between multiple legions such as melanoma, nevus, keratosi, etc.), and provide uncertainty estimates for predictions
- Trained bayesian neural network
- C++ applicaton for real time inference with uncertainty visualizations
- Can be used in Medical Industry

### **Steps:**

**1: Gather and Understand Data**
- Using the [**ISIC dataset**](https://api.isic-archive.com/collections/249/)
  - BCN200 Dataset consists of dermoscopic images of skin lesions between 2010 - 2016 at Hospital Clinic in Barcelona. As well as the metadata associated with each image
- Understand data characteristics:

**2: Setup Virtual Environment**
- optional

**3. Build prototype in python**
- data preprocessing
- Model development
- integrate tensorflow(possibly)
- training and testing
- Export Model

**4. Prepare C++ deployment**
- setup tensorflow or ONNX runtime
- Load and Infer Model
- Preprocess images in c++
- Run inference

**5. Visualize Results**


## Challenges
- issues with pushing to repo due to size of data
- Solutions:
  - **Lower image resolution (no smaller than 128 by 128)**
  - **Compress metadata**

 # USE ``` gunzip metadata.csv.gz ``` to uncompress metadata and ``` gzip metadata.csv ``` to compress





