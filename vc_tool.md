# Semi-supervised predictive framework for genetic variants using epigenome features

## Introduction

The purpose of this software is to predict genetic variants of interest using epigenome features, in cases where only a small amount of labeled training data is available. This software was developed in the [Gaulton lab](http://www.gaultonlab.org/) at UC San Diego as part of a research project in which we aimed to use computational approaches to predict genetic variants that may have a role in Type 2 Diabetes (T2D) risk. We modeled the epigenome landscape of human pancreatic islets for known T2D risk variants in accessible chromatin regions, and then made predictions on all genetic variants in islet accessible chromatin genome-wide. A semi supervised algorithm, label propagation (LP), was used in this research because the number of labeled training data, in this case known T2D risk variants in islet accessible chromatin, was limited. Label propagation uses labeled and unlabeled data in order make predictions. For example, known T2D risk variants were considered true positive, variants with the lowest GWAS associations at each corresponding T2D locus were considered true negative, and all other variants at each corresponding T2D locus were considered unlabeled. The number of unlabeled training data should be much larger than labeled training data. This framework is extensible to other diseases or traits in which a limited number of "true positive" genetic variants are known.

## Installation

* Clone this repository to the machine where you will be running this software.
* All dependencies should be included in [Anaconda](https://www.anaconda.com/download/) for Python3.
* Python3 is required to run this software.

## Usage

### Input file format

```
#chrom  start   end   feat1   feat2   ...   featN   label
chr1    1   2   x   x   ...   x   0
chr1    11   12   x   x   ...   x   1
chr1    20   21   x   x   ...   x   -1
```
The file format for training file is seen as above. Header line is required. The file should be tab delimited. The first three columns describe genomic coordinates, the last column is the label (1=positive, 0=negative, -1=unlabeled), and all remaining columns are features used to describe each of these coordinates.

The file format for testing and pca files are almost the same, but do not have a column for labels.

### Options

```
$ python3 vc-tool.py -h
usage: vc-tool.py [-h] -t TRAIN -o OUTDIR -n NAME -p PCADATA -x OPTIMIZE
                  [-a PCAVALUES [PCAVALUES ...]] [-g GAMMAS [GAMMAS ...]]
                  [-s TEST] [-m MEAN] [-d NUMSEEDS] [-c THREADS]

optional arguments:
  -h, --help            show this help message and exit
  -a PCAVALUES [PCAVALUES ...], --pcavalues PCAVALUES [PCAVALUES ...]
                        minimum and maximum number of PCs to test(step size =
                        1). Example: -a 10 15. Use one value if model is
                        already trained.
  -g GAMMAS [GAMMAS ...], --gammas GAMMAS [GAMMAS ...]
                        minimum, maximum, and step size for RBF kernel gamma
                        parameter. Example: -g 0.1 5 0.1. Use one value if
                        model is already trained.
  -s TEST, --test TEST  path to testing feature file
  -m MEAN, --mean MEAN  input an integer n. predictions are comprised of the
                        mean of n models with n random seeds. Must also
                        include training file when using this flag.
  -d NUMSEEDS, --numseeds NUMSEEDS
                        integer number of random seeds to evaluate during
                        training. default = 25
  -c THREADS, --threads THREADS
                        number of threads for multithreading, default = 8

required arguments:
  -t TRAIN, --train TRAIN
                        path to training feature file, or trained model
                        (.model extension)
  -o OUTDIR, --outdir OUTDIR
                        path to directory where output will be stored
  -n NAME, --name NAME  prefix of output file names
  -p PCADATA, --pcadata PCADATA
                        path to pca data feature matrix
  -x OPTIMIZE, --optimize OPTIMIZE
                        method of hyperparameter optimization, grid or
                        bayesian
```

### Example 1: Training and testing in one line

```
$ python3 vc-tool.py --train training_file --test testing_file --gammas 0.1 1.0 0.1 --pcavalues 10 20 --pcadata testing_file --name test --outdir out/ -x grid 
```

Let's pick apart what each of these options are doing.

```
--train training_file
```
This is simply the path to your file containing your labeled training data.

```
--test testing_file
```
Similarly, this is the path to your file containing unlabeled testing data.

```
--gammas 0.1 1.0 0.1
```
This tells the software to evaluate model performance using different values of gamma between 0.1 up to but not including 1.0 with step size 0.1. Gamma is a parameter internal to the RBF kernel of the LP algorithm, which changes how much influence one labeled data point has on inferring the label of other data. If the bayesian option is used instead of grid, step size does not matter but it is still required as of now.

```
--pcavalues 10 20
```
This tells the software to evaluate model performance when using different numbers of principal components from PCA. 

```
--pcadata testing_file
```
This is the path to the file that PCA will be fit to. You probably want a file containing a lot of data, so we are just going to use the same file as our testing file. 

```
--name test --outdir out/
```
The stem name of all of the resulting output files, and the directory in which these files will be written to.

```
-x grid
```
The method of hyperparameter optimization. Can either use grid or bayesian.

### Example 2: Training only
```
$ python3 vc-tool.py --train training_file --gammas 0.1 1.0 0.1 --pcavalues 10 20 --pcadata testing_file --name test --outdir out/ -x grid
```
This is simply the same command as above, but without the --test option. The best performing model and parameters will be saved in your output directory, and you can make predictions using this saved model/parameters later.

### Example 3: Testing using a saved model
```
$ python3 vc-tool.py --train test.model --test testing_file --outdir out/ --name test --pcadata testing_file --pcavalues 11
```
Here we are loading a .model file, which gets saved to the specified output directory after training, in order to make predictions. Only one value is needed for the --pcavalues option, which is the number of PCs used in the best performing model.
