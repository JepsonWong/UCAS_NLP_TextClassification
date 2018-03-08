# UCAS-NLP-TextClassification

## Members

- 蔡恒毅<201618013229001>
- 王忠朴<201618013229001>
- 乔舜杰<201618013229001>

## Prerequisites
- Linux or OSX.
- Python 2.

## Getting Started

### Installation

在终端下执行如下命令以安装所需依赖

```bash
$ cd PATH_TO_PROJECT
$ pip install -r requirements.txt
```

### Train

在项目根目录下(以下命令除特殊说明外均是在项目根路径下执行)，
输入 `python train.py -h` 以查看可用的训练选项及其说明:

```bash
$ python train.py -h
usage: train.py [-h] [--name NAME] [--checkpoints_dir CHECKPOINTS_DIR]
                [--encoding ENCODING] [--suffix_accepted SUFFIX_ACCEPTED]
                [--corpus_root CORPUS_ROOT] [--tfidf_top_k TFIDF_TOP_K]
                --path_to_save_model PATH_TO_SAVE_MODEL
                [--which_filter WHICH_FILTER]
                [--which_classifier WHICH_CLASSIFIER]
                [--mi_threshold MI_THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           name for this execution
  --checkpoints_dir CHECKPOINTS_DIR
                        path to save options for this execution
  --encoding ENCODING   file encoding of documents
  --suffix_accepted SUFFIX_ACCEPTED
                        file with suffix_accepted will be read
  --corpus_root CORPUS_ROOT
                        path to documents(should have subfolders
                        C1-Class1Name, C2-Class2Name,...,Cn-ClassnName)
  --tfidf_top_k TFIDF_TOP_K
                        features with tfidf value within top_k will be
                        selected
  --path_to_save_model PATH_TO_SAVE_MODEL
                        path to save the model
  --which_filter WHICH_FILTER
                        mi or gbdt to filter the features
  --which_classifier WHICH_CLASSIFIER
                        svm or gbdt classifier
  --mi_threshold MI_THRESHOLD
                        threshold value used by mutual information feature
                        selection
```

其中，参数`path_to_save_model`为必填选项，用于指定模型训练好之后的存储路径，其它选项均有默认值。

在终端中输入如下命令，用以训练模型:

```bash
$ python train.py \
--name mi_0.06_svm \
--encoding gb18030 \
--corpus_root ./corpus_train/ \
--tfidf_top_k 5000 \
--which_filter mi \
--mi_threshold 0.06 \
--which_classifier svm \
--path_to_save_model ./model_saved/mi_0.06_svm.m
```

其中，训练语料位于路径`./corpus_train/`下，文档的编码是`gb18030`, 选取 **TFIDF Top 5000** 的特征作为初始特征, 
特征选择使用`mi`方式(互信息), 阈值设为`0.06`, 分类器采用`SVM`, 模型训练好后，保存在
`./model_saved/mi_0.06_svm.m` 下。

训练时间随训练语料的大小各异，在一万篇文档的情况下，训练时间为4h。

类似，也可以采用其他选项训练模型。

### Apply a pre-trained model

模型训练好后，可以使用测试样本对模型进行测试，可用的测试选项有:

```bash
$ python test.py -h
usage: test.py [-h] [--name NAME] [--checkpoints_dir CHECKPOINTS_DIR]
               [--encoding ENCODING] [--suffix_accepted SUFFIX_ACCEPTED]
               --model_path MODEL_PATH --test_dir TEST_DIR

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           name for this execution
  --checkpoints_dir CHECKPOINTS_DIR
                        path to save options for this execution
  --encoding ENCODING   file encoding of documents
  --suffix_accepted SUFFIX_ACCEPTED
                        file with suffix_accepted will be read
  --model_path MODEL_PATH
                        path to pretrained model
  --test_dir TEST_DIR   path to test dir(should have some documents under it)
```

其中, 参数 `model_path` 和 `test_dir` 均为必填项，
用于指定模型所在的本地路径和测试样本所在的本地路径。

在终端中输入如下命令，用以测试模型:

```bash
$ python test.py \
-name test_for_mi_0.06_svm \
--encoding gb18030 \
--suffix_accepted txt,csv,html \
--model_path ./model_saved/mi_0.06_svm.m \
--test_dir ./corpus_test/
```

其中，测试语料位于路径 `./corpus_test/` 下, 测试文档的编码为 `gb18030`, 
允许读取的文件后缀为`txt,csv,html`, 使用的模型为 `./model_saved/mi_0.06_svm.m`。