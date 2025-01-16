# Medical_Image_Analysis_with_Risk_Assessment

## About + Thought process

### **Goal:**
- Make a system that detects skin cancer in patients(detects melanoma vs benign lesions, classify between multiple legions such as melanoma, nevus, keratosi, etc.), and provide uncertainty estimates for predictions
- Trained bayesian neural network
- C++ applicaiton for real time inference with uncertainty visualizations
- Can be used in Medical Industry

### **Steps:**

**1: Gather and Understand Data**
- Using the [**ISIC dataset**](https://api.isic-archive.com/collections/249/)
  - BCN200 Dataset consists of dermoscopic images of skin lesions between 2010 - 2016 at Hospital Clinic in Barcelona.
- Understand data characteristics

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




