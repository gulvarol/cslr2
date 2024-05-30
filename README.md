<div align="center">

# CSLR<sup>2</sup>
## Large-Vocabulary Continuous *Sign Language* Recognition <br> from *Spoken Language* Supervision

<a href="https://imagine.enpc.fr/~raudec/"><strong>Charles Raude</strong></a> · <a href="https://www.robots.ox.ac.uk/~prajwal/"><strong>Prajwal KR</strong></a> · <a href="https://www.robots.ox.ac.uk/~liliane/"><strong>Liliane Momeni</strong></a> ·
<a href="https://hannahbull.github.io/"><strong>Hannah Bull</strong></a> · <a href="https://samuelalbanie.com/"><strong>Samuel Albanie</strong></a> · <a href="https://www.robots.ox.ac.uk/~az/"><strong>Andrew Zisserman</strong></a> · <a href="https://imagine.enpc.fr/~varolg"><strong>G&uuml;l Varol</strong></a>

[![arXiv](https://img.shields.io/badge/arXiv-CSLR2-A10717.svg?logo=arXiv)](https://arxiv.org/abs/2405.10266)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

</div>

## Description
Official PyTorch implementation of the paper:
<div align="center">

[**A Tale of Two Languages: Large-Vocabulary Continuous *Sign Language* Recognition from *Spoken Language* Supervision**](https://arxiv.org/abs/2405.10266).

</div>

Please visit our [**webpage**](https://imagine.enpc.fr/~varolg/cslr2/) for more details.

### Bibtex
If you find this code useful in your research, please cite:

```bibtex
@article{raude2024,
    title={A Tale of Two Languages: Large-Vocabulary Continuous Sign Language Recognition from Spoken Language Supervision},
    author={Raude, Charles and Prajwal, K R and Momeni, Liliane and Bull, Hannah and Albanie, Samuel and Zisserman, Andrew and Varol, G{\"u}l},
    journal={arXiv},
    year={2024}
}
```

## Installation :construction_worker: 

<details><summary>Create environment</summary>
&emsp;

Create a conda environment associated to this project by running the following lines:
```bash
conda create -n cslr2 python=3.9.16
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install anaconda::pandas=1.5.3
conda install conda-forge::einops=0.6.0
conda install conda-forge::humanize=4.6.0
conda install conda-forge::tqdm=4.65.0
pip install hydra-core==1.3.2
pip install matplotlib==3.7.1
pip install plotly==5.14.1
pip install nltk==3.8.1
pip install seaborn==0.12.2
pip install sentence-transformers==2.2.2
pip install wandb==0.14.0
pip install lmdb
pip install tabulate
pip install opencv-python==4.7.0.72
```
You can also create the environment using the associated `.yaml` file using conda (this might not always work, depending on the machine and the version of conda installed, try to update the version of conda).

```bash
conda env create --file=environment.yaml
```

After installing these packages, you will have to install a few `ntlk` packages manually in Python.

```python
import nltk
nltk.download("wordnet")
```
</details>

<details><summary>Set up the BOBSL data</summary>

* Make sure you have the permission to use the BOBSL dataset. You can request access following the instructions at [the official BOBSL webpage](https://www.robots.ox.ac.uk/~vgg/data/bobsl/).
* With the username/password obtained, you can download the two required files via the following:
  ``` bash
  # Download the pre-extracted video features [262G]
  wget --user ${BOBSL_USERNAME} --password ${BOBSL_PASSWORD} \
      https://thor.robots.ox.ac.uk/~vgg/data/bobsl/features/lmdb/feats_vswin_t-bs256_float16/data.mdb
  # Download the raw video frames [1.5T] (you can skip this if purely training/testing with features, and not visualizing)
  wget --user ${BOBSL_USERNAME} --password ${BOBSL_PASSWORD} \
      https://thor.robots.ox.ac.uk/~vgg/data/bobsl/videos/lmdb/rgb_anon-public_1962/data.mdb
  ```
* Download [`bobsl.zip` 
  (1.9G)](https://drive.google.com/file/d/13pp83GCoy1SVScvZRmNsoFxtNm8ogI7h/view?usp=sharing) 
  for the rest of the files (including annotations and metadata). Note the folder becomes 
  15G when decompressed. Make sure they correspond to the paths defined here: 
  `config/paths/public.yaml`.
* Download [`t5_checkpoint.zip` (1.4G)](https://drive.google.com/file/d/1hxkb8KAC0sgSYKefOLue1wyO2fJmmqxT/view?usp=sharing) for the T5 pretrained model weights, also defined at `config/paths/public.yaml`.

</details>

## Training :rocket:

```python
export HYDRA_FULL_ERROR=1  # to get better error messages if job crashes
python main.py run_name=cslr2_train
```
permits to train the CSLR2 model with the best set of hyperparameters obtained in the paper.
Using 4 x V100-32Gb, training for 20 epochs should take less than 20 hours.

To change training parameters, you should be looking at changing parameters in the `config/` folder.

To manually synchronise the offline jobs on wandb, one should run: `wandb sync --sync-all` in the folder of the experiment (do not forget to do `export WANDB_MODE=offline` first).

Training should save one model per epoch as `$EXP_NAME/models/model_$EPOCH_NB.pth`. Also, the model that obtains the best T2V performance on validation set is saved as `$EXP_NAME/models/model_best.pth`.

## Test :bar_chart:

You can download a pretrained model from [here](https://drive.google.com/file/d/1qyFHSFnxmy1rRGjlKEBfsjC8yt2kdalx/view?usp=sharing).

### 1. Retrieval on 25K manually aligned test set

To test any model for the retrieval task on the 25K manually aligned test set, one should run the following command:

```python
python main.py run_name=cslr2_retrieval_25k checkpoint=$PATH_TO_CHECKPOINT test=True
```

### 2. CSLR evaluation

CSLR evaluation is done in two steps. First, extract frame-level predictions and then evaluate.

#### 2.1 Feature Extraction

```python
python extract_for_eval.py checkpoint=$PATH_TO_CHECKPOINT
```
extracts predictions (linear layer classification, nearest neighbor classification) for both heuristic aligned subtitles and manually aligned subtitles.

#### 2.2 Evaluation

```python
python frame_level_evaluation.py prediction_pickle_files=$PRED_FILES gt_csv_root=$GT_CSV_ROOT
```
Note that by default, if gt_csv_root is not provided, it will use `${paths.heuristic_aligned_csv_root}`.


## Pre-processing of gloss annotations :computer:

You do not need to run this pre-processing, but we release the scripts for how to convert raw 
gloss annotations (released from the official BOBSL webpage) into the format used for our 
evaluation. A total of 4 steps are required to fully pre-process gloss annotations that are 
stored in json files.

<details>
<summary> 1. Assign each annotation to its closest subtitle</summary>

```python
python misc/process_cslr_json/preprocess_raw_json_annotations.py --output_dir OUTPUT_DIR --input_dir INPUT_DIR --subs_dir SUBS_DIR --subset2episode SUBSET2EPISODE
```
where `INPUT_DIR` is the directory where json files are stored and `OUTPUT_DIR` is the directory where the assigned annotations are saved.
`SUBS_DIR` is the directory where manually aligned subtitles are saved. This corresponds to the `subtitles/manually-aligned` files from the public release.
`SUBSET2EPISODE` is the path to the json file containing information about splits and episodes. This corresponds to the `subset2episode.json` file from the public release.
</details>



<details>
<summary>2. Fix boundaries of subtitles.</summary>

During assignment, it could happen that certain annotations overlap with the boundaries of subtitles. It could even happen that certain annotations are not within the boundaries of its associated subtitle.
Since at evaluation time, we load all features corresponding to subtitles timestamps, we need to extend boundaries of certain subtitles.

```python
python misc/process_cslr_json/fix_boundaries.py --csv_file OUTPUT_DIR
```
</details>


<details>
<summary>3. Fix alignment of subtitles.</summary>

Subtitles have been manually aligned. However, since gloss annotations are much more treated more precisely, it could happen that certain gloss annotations better match surrounding subtitles.
In order to fix this, we propose an automatic re-alignment algorithm.

```python
python misc/process_cslr_json/fix_alignment.py --csv_file OUTPUT_DIR2
python misc/process_cslr_json/preprocess_raw_json_annotations.py --output_dir OUTPUT_DIR3 --input_dir INPUT_DIR --subs_dir OUTPUT_DIR2 --misalignment_fix
```

where `OUTPUT_DIR2 = OUTPUT_DIR[:-8] + "extended_boundaries_" + OUTPUT_DIR[-8:]` and `OUTPUT_DIR3 = OUTPUT_DIR2[:-8] + "fix_alignment_" + OUTPUT_DIR2[-8:]`.
Here we assume that `OUTPUT_DIR` ends with a date in the format DD.MM.YY
</details>

<details>
<summary>4. Only keep lexical annotations.</summary>

We only evaluate against lexical annotations: i.e., annotations that are associated with a word.

```python
python misc/process_cslr_json/remove_star_annots_from_csvs.py --csv_root OUTPUT_DIR2  # only boundary extension fix
python misc/process_cslr_json/remove_star_annots_from_csvs.py --csv_root OUTPUT_DIR3  # with total alignment fix
```
</details>

<details>
<summary>Do all the steps with one command.</summary>

**Instead, you can also use `python misc/process_cslr_json/run_pipeline.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR --subs_dir SUBS_DIR --subset2episode SUBSET2EPISODE`**
</details>

## License :books:
This code is developed by [Charles Raude](https://github.com/charles-raude), may not be 
maintained, and is distributed under an [MIT LICENSE](LICENSE).

Note that the code depends on other libraries, including PyTorch, T5, Hydra, and use the BOBSL dataset which each have their own respective licenses that must also be followed.

The license for the BOBSL-CSLR data can be found at [https://imagine.enpc.fr/~varolg/cslr2/license.txt](https://imagine.enpc.fr/~varolg/cslr2/license.txt).
