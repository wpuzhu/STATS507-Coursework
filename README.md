# STATS507-Coursework
## Course Description
This course surveys data science tools and frameworks that are currently used in academia and industry. The first part of this course is an accelerated introduction to the Python programming language, with an emphasis on using object-oriented and functional programming techniques for data analysis. The second part covers methods for handling structured data (regular expressions, HTML/JSON, SQL), data visualization, numerical computing, version control, and the UNIX/Linux command line. The third part of this course is an introduction to deep learning and/or large-scale distributed computing.
## Purpose of this Repository
1. Organize all the course-related content, such as in-calss practices, homeworks, and final project. <br /> 
2. Develop and organize final project, open access to other related people.
## Project Guideline
├── multi-omics/
│   ├── T1 dataset
│   ├── T2 dataset
│   ├── ADC dataset
│   └── post-hoc dataset
│
├── code/
│   ├── EDA.ipynb
│   ├── supervised_baselines.ipynb
│   ├── adaptive_gated_fusion.ipynb
│   └── unsupervised_clustering.ipynb
│
└── README.md
1. The original data for the final project are in the multi-omics folder with 4 primary dataset, each corresponding to T1, T2, ADC-driven features and post-hoc dataset. The train and test after split dataset are not uploaded as if you run the code in EDA, it will automatically generated for you. 
2. The step by step code development of project can be found in the code folder with 4 jupyter notebooks, each corresponbding to EDA, supervised baseline models, supervised adaptive gated fusion network, and unsupervised cluster learning that contain detailed information of the models and model performance. All the functions and models in the notebook are carefully set seed to be determinictics but you can still run some code chunk isolated to get different randomizaed results.
