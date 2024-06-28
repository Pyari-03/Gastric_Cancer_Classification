# Gastric_Cancer_Classification
Gastric Cancer Classification Through Computer Vision Techniques

Link of Dataset :- https://drive.google.com/drive/folders/1JF6zExBSyHELVnsmRDYGxGDnBMWqHCVV?usp=sharing

Abstract-

Gastric cancer is a diverse disease making it tough to diagnose and categorize accurately. This project seeks to create a computer vision system that can effectively classify cancer. Current diagnostic methods for cancer demand specialized knowledge and are time intensive. The proposed system aims to tackle this issue by using computer vision techniques to identify the features of tissues in gastric images. It will employ image processing-based feature learning methods, such as feature extraction and classification to detect the presence of cancerous tissues in gastric images. Furthermore, the proposed system will provide details about the tissue, including its size, location and severity. By enhancing the efficiency and accuracy of cancer diagnosis this system will empower professionals to offer better treatment options for their patients.  To enhance accuracy and efficiency the proposed system will integrate machine learning algorithms, especially deep learning-based CNN architectures into its framework. These algorithms will allow the system to learn the required feature space from the medical image datasets, thereby improving its ability to classify cancerous tissues with precision. Moreover, the proposed system is designed to handle different types of data corresponding to the image datasets, such as histopathology images. This project is expected to bring about advantages in terms of precision, efficiency and cost efficiency rendering it an asset for healthcare practitioners as well as for the cancer patients to get timely treatment. The system aims to enhance the precision of diagnosing conditions, shorten diagnosis durations and decrease the necessity for procedures like biopsies. Additionally, it is expected to offer a cost alternative to existing methods for detecting gastric cancer.

Keywords-  Computer Vision, Residual Network, Gastric Cancer Classification

Introduction-

Gastric cancer is the fifth most common malignancy and the third most common cause of cancer related death worldwide. Early detection and diagnosis are key to help improve outcomes for patients, as early-stage gastric cancer is very curable. But such conventional diagnostic procedures as endoscopy with biopsy are subjective and may lack the degree of human error. Of course, this requires the development of new, precise and reliable means of diagnosis.
Over the past few years, we have seen the emergence of several fascinating opportunities of using computer vision algorithms in medical image analysis together with other AI advances. The elucidations of the methods driven by artificial intelligence (AI) have given way to several promising uses of computer vision methods in the realm of imaging for medical interventions. Automation capabilities have the potential to save time, resources, and increase the precision of the outcomes, while providing insights of interests to medical professionals. 
• This necessitates the exploration of novel, non-invasive
techniques to enhance the efficiency and accuracy of
gastric cancer diagnosis.
• This project seeks to create a computer vision system that
can effectively classify cancer.
• Current diagnostic methods for cancer demand special-
ized knowledge and are time intensive.
• The proposed system aims to tackle this issue by using
computer vision techniques to identify the features of
tissues in gastric images.
• It will employ image processing-based feature learning
methods, such as feature extraction and classification to
detect the presence of cancerous tissues in gastric images.
• Furthermore, the proposed system will provide details
about the tissue, including its size, location and severity.

Methodologies-

ResNet (Residual Network): A strong CNN architecture to address the difficulties that deepen the gradient vanish in deep learning. It brings the "shortcut connections” of information within the residual blocks, making it possible to directly pass information from earlier layers to the later ones in deep networks. This helps in the training and learning of the features where there is complexity. 
 
 ResNet-18:-
 Efficiency: Simplified and computationally less demanding compared to deeper ResNets hence it takes less time to train and being suitable for use on consumable devices. 
 Advantages: 
 Faster training.
 For analysis and debugging, the methods used in the architecture should be uncomplicated. 
 Reducing the computational load in training and testing phase.

ResNet-34:-
As the name suggests, it consists of 34 layers and is considered as a deeper version of the ResNet architecture, providing better performance on complex tasks. The architecture explained as follows:
1. Convolutional Layers and Activation Functions: Similar to ResNet-18, the base of ResNet-34 are the convolutional layers which integrate features from the input images. These layers include filters that analyze the image, searching for patterns and filtering information pertinent to the task. Activation functions such as ReLU are applied after each convolutional layer to model non-linearity.
2. Residual Blocks: The key to the ResNet-34 architecture is its residual blocks. These blocks help overcome the vanishing gradient problem that occurs in deep neural networks. Within these blocks, shortcut connections are implemented, which enable the original input data to bypass some of the layers, allowing the network to learn residual functions with reference to the layer inputs, rather than learning unreferenced functions.
3. Downsampling: ResNet-34 uses max pooling and strided convolutions for downsampling the feature maps, reducing the spatial dimensions while increasing the number of channels. This allows the network to capture higher-level features as the depth increases.
4. Fully Connected Layers: After the convolutional and residual blocks, the network ends with global average pooling and a fully connected layer for classification.
The deeper architecture of ResNet-34, with 34
Depth: Residual designed, 34 layers deep, not too shallow and not too deep that excels at image classification with low error rate. 
 Key Features: 
 Bottleneck blocks for a higher efficiency realized at higher layers 
 Proposed the stacked residual blocks to perform the feature extraction. 
 C2: Orientation for Ricer Representation and Classification of Channels 
 Advantages: 
 The training of very deep networks 
 Better model generalization for a higher classification rate 
 Improvement in feature representation for capturing all the intricacies 
 Suitable for different image databases

ResNet-50:- 
 Depth: A far deeper network (50 layers) that enables it understand hierarchical representations of data more so than in ResNets. 
 Architecture: 
 The following is a procedure to implement bottleneck blocks for the purpose of the reduced computational cost: 
 To enhance the gradient flow, skip connections as a remedy 
 Max pooling on top of the neural network before the softmax layer as a technique of dimensionality reduction and to prevent overfitting 
 Advantages: 
 Proposing a deep architecture to increase the richness of descriptions done on features. 
 Shortcuts to overcome the bad gradient good for gradient flow and decreasing of a vanishing gradient problem 
 Enhances satisfactory rates of image recognition and classification exactly as the current trend across the states. 
 There is also less overfitting min pooling connection and skip connection 
 More scale to create greater or lesser models with high accuracy.

System Requirements:-

A collection of tools and technologies which are used together for the development of software. 
Just like any other toolkit one is accustomed to such as Jupyter Notebook or Jupyter Lab to perform data analysis and visualization. 
Key libraries for numerical computing, systems computing, image processing, and machine learning: Python 3.7+ (or higher) NumPy, SciPy Pillow OpenCV OpenSlide Scikit-learn PyTorch 1.3.0+ with CUDA 10.1 or higher is used for deep learning tasks supported by torchvision library.
We have also utilized Google Colab pro, as it is easy to use, collaborative, has pre-installed libraries and also provides free GPU and TPU access.

Proposed Algorithm(s):-

These following metrics would provide insights into the performance and effectiveness of the proposed computer vision approach for gastric cancer classification.

Accuracy: This is a popular parameter that tells us what fraction of images are correctly classified in general. It is obtained as (TP + TN) / (TP + TN + FP + FN), where TP stands for True Positives (cancerous images that were classified accurately), TN means True Negatives (healthy images that were identified properly), FP implies False Positives (misclassified cancerous images - normal ones identified as having cancer), and FN means False Negatives (misclassified normal images – sickly identified).
In the code we have implemented the given formula-
train accuracy = train acc / len(dataloader)
test accuracy = test acc / len(dataloader)

Loss function: Evaluates how well an algorithm is modeling a dataset. 
In the code we have implemented the given formula-
train loss = train loss / len(dataloader)
test loss = test loss / len(dataloader)

Cross-entropy loss: In this project we have used Cross-entropy loss. It is a type of loss function that is frequently used in machine learning. It calculates the dissimilarity between the output of the provided model and the expected output, and its main objective is to ensure that the energy between the estimated probability density function and the actual probability density function is at a minimum level. This is achieved through the use of the squared error, which is summed for all output units and weighted by the true output probabilities. A cross-entropy loss of less than 0. 3 is considered a good model for the framework.  

Dataset:-https://figshare.com/articles/dataset/GasHisSDB/15066147

Conclusions and Future Scope:-

This provides an insight into the potential magnitude of how computer vision techniques could be used in gastric cancer classification. Owing to the importance of the implications involved, our study intended to investigate this potential, using the GasHisSDB dataset and a Residual Network based model.

The performance was measured in either accuracy and cross entropy loss. Instead it provides very informative insights into what the model can and cannot do. Although research and refinement should be done further, it implied that computer vision is a breakthrough in supporting the diagnosis of gastric cancer.

This technology has the potential for numerous advantages: automated analysis to improve upon human errors, quicker diagnosis from fast data analysis, and thus possibly earlier detection. Moreover, broader implementation can solve the problem of the limited number of specialist endoscopists and pathologists and contribute to the more just provision of health care. Based on the provided metrics, the ResNet-50 model appears to be the most optimized result among the three models presented with accuracy at 83.87% and the highest validation accuracy at 79.33%, indicating better overall performance.

In the future, researchers should work to expand the availability of data, address the issue of data bias, and improve model interpretability for the technology to be better integrated into clinical practice. Therefore, advanced deep learning architecture development, coupled with its integration into current clinical workflows, becomes an imperative step to unlock computer vision technologies in the fight against gastric cancer.

References:-

[1] S. -A. Lee, H. C. Cho and H. -C. Cho, “A Novel Approach for Increased Convolutional Neural Network Performance in Gastric-Cancer Classification Using Endoscopic Images,” in IEEE Access, vol. 9, pp. 51847-51854, (2021), doi: 10.1109/ACCESS.2021.3069747.

[2] Sano, T., Aiko, T. “New Japanese classifications and treatment guidelines for gastric cancer: revision concepts and major revised points,” Gastric Cancer 14, 97–100 (2011), doi:10.1007/s10120-011-0040-6.

[3] Ikenoyama Y, Hirasawa T, Ishioka M, Namikawa K, Yoshimizu S, Horiuchi Y, et al., “Detecting early gastric cancer: Comparison between the diagnostic ability of convolutional neural networks and endoscopists,” doi: 10.1111/den.13688. 

[4] Atsushi Teramoto,Tomoyuki Shibata,Hyuga Yamada,Yoshiki Hirooka,KuniakiSaito and Hiroshi Fujita , “Automated Detection of Gastric Cancer by Retrospective Endoscopic Image Dataset Using U-Net R-CNN,” doi:10.3390/app112311275.

[5] Xiaodong Wang,Ying Chen, Yunshu Gao, Huiqing Zhang, Zehui Guan, Zhou Dong, Yuxuan Zheng, Jiarui Jiang, Haoqing Yang, Liming Wang, Xianming Huang, “Predicting gastric cancer outcome from resected lymph node histopathology images using deep learning,” ,doi: 10.1038/s41467-021-21674-7.

[6] Ming Ping Yong ,Yan Chai Hum,Khin Wee Lai,Ying Loong Lee ,Choon-Hian Goh ,Wun-She Yap  andYee Kai Tee, “Histopathological Gastric Cancer Detection on GasHisSDB Dataset Using Deep Ensemble Learning,”doi:10.3390/diagnostics13101793.

[7] Tomoyuki Shibata , Atsushi Teramoto, Hyuga Yamada , Naoki Ohmiya , Kuniaki Saito and Hiroshi Fujita , “Automated Detection and Segmentation of Early Gastric Cancer from Endoscopic Images Using Mask R-CNN,” Appl. Sci. 2020, 10, 3842, doi:10.3390/app10113842 .

[8] Yuanpeng Li, Liangyu Deng, Xinhao Yang, Zhao Liu, Xiaoping Zhao, Furong Huang, Siqi Zhu, Xingdan Chen, Zhenqiang Chen, and Weimin Zhang. , “Early diagnosis of gastric cancer based on deep learning combined with the spectral-spatial classification method,”doi:10.1364/BOE.10.004999.

[9] Zhang, K., Wang, H., Cheng, Y. et al. “Early gastric cancer detection and lesion segmentation based on deep learning and gastroscopic images,” doi:10.1038/s41598-024-58361-8

[10] Shuang-Li Zhu, Jie Dong, Chenjing Zhang, Yao-Bo Huang, and Wensheng Pan. , “Application of Machine Learning in the Diagnosis of Gastric Cancer Based on Noninvasive Characteristics,”doi: 10.1371/journal.pone.0244869.

[11]  Chen, Ying; Wang, Xiaodong; Liu, Xiyang; Yu, Guanzhen, “Gastric cancer lymph node data set,” doi: 10.6084/m9.figshare.13065986.v34.

[12] Chen Li, Weiming Hu, Changhao Sun, “Histopathological Gastric Cancer Detection onGasHisSDB,” doi:10.6084/m9.figshare.15066147.v1.   
 
[13] Sun,Changhao; Li, Chen; Li, Yixin (2020), “Data for HCRF,” MendeleyData,V2,doi: 10.17632/thgf23xgy7.2.










 
