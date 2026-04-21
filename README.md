# STATS507-Coursework
## Course Description
This course surveys data science tools and frameworks that are currently used in academia and industry. The first part of this course is an accelerated introduction to the Python programming language, with an emphasis on using object-oriented and functional programming techniques for data analysis. The second part covers methods for handling structured data (regular expressions, HTML/JSON, SQL), data visualization, numerical computing, version control, and the UNIX/Linux command line. The third part of this course is an introduction to deep learning and/or large-scale distributed computing.
## Purpose of this Repository
1. Organize all the course-related content, such as in-calss practices, homeworks, and final project. <br /> 
2. Develop and organize final project, open access to other related people.
## Project Guideline
This project investigates the preoperative prediction of lymphovascular space invasion (LVSI) in endometrial cancer using multi-modal MRI-based radiomics features (T1, T2, and ADC).

Both supervised learning (baseline models and adaptive fusion networks) and unsupervised learning (clustering and stability analysis) are employed to evaluate predictive performance and uncover intrinsic data structure.

### Dataset
The original data for the final project are located in the multi-omics folder, which contains four primary datasets corresponding to T1-, T2-, ADC-derived features, as well as the post-hoc dataset. Separate train/test split files are not provided, as these datasets can be generated automatically by running the code in the Explanatory_Data_Analysis.ipynb notebook.

### Code Notebook
The project’s step-by-step code development is provided in the code folder, which contains four Jupyter notebooks covering:

1. Exploratory data analysis (EDA)
2. Supervised baseline models
3. Supervised adaptive gated fusion networks (progressive development)
4. Unsupervised clustering analysis

Each notebook includes detailed implementation of the corresponding methods and summaries of model performance. Most functions and models are run with fixed random seeds to ensure reproducibility. Nevertheless, if certain code blocks are executed independently, slight variations in randomized results may still occur.
