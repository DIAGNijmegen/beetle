# BEETLE: a multicentric dataset for training and benchmarking breast cancer segmentation in H&E slides

<img width="5264" height="1054" alt="banner" src="https://github.com/user-attachments/assets/64a39678-6a75-44d3-83fe-9884c43c3ba6" />

> Automated semantic segmentation of whole-slide images (WSIs) stained with hematoxylin and eosin (H&E) is essential for large-scale artificial intelligence-based biomarker analysis in breast cancer. However, existing public datasets for breast cancer segmentation lack the morphological diversity needed to support model generalizability and robust biomarker validation across heterogeneous patient cohorts. We introduce BrEast cancEr hisTopathoLogy sEgmentation (BEETLE), a dataset for multiclass semantic segmentation of H&E-stained breast cancer WSIs. It consists of 587 biopsies and resections from three collaborating clinical centers and two public datasets, digitized using seven scanners, and covers all molecular subtypes and histological grades. Using diverse annotation strategies, we collected annotations across four classes - invasive epithelium, non-invasive epithelium, necrosis, and other - with particular focus on morphologies underrepresented in existing datasets, such as ductal carcinoma in situ and dispersed lobular tumor cells. The dataset's diversity and relevance to the rapidly growing field of automated biomarker quantification in breast cancer ensure its high potential for reuse. Finally, we provide a well-curated, multicentric external evaluation set to enable standardized benchmarking of breast cancer segmentation models.

### Repository layout

Welcome to the GitHub repository for the BEETLE dataset. This repository provides:

* Code to download the dataset from the associated [Zenodo repository](https://zenodo.org/records/16812932)
* Code to run inference using the model trained on the dataset for technical validation

The repository is laid out as follows:

* The [`data/`](data/) folder starts out empty and is populated with files after running the [`download_all.sh`](download_all.sh) shell script, which downloads and extracts the dataset files. After extraction, the folder is organized as follows:

```bash
.
└── data/
    ├── annotations/       # Annotations for the development set in multiple formats        
    │   ├── jsons/         # JSON format with tissue compartments annotated as polygons
    │   ├── label_map.json # Mapping of pixel values to class labels
    │   ├── masks/         # Multiresolution TIFF images with pixel-wise class labels
    │   └── xmls/          # XML format with tissue compartments annotated as polygons
    │
    ├── images/            # Images for the development and evaluation sets
    │   ├── development/
    │   │   └── wsis/      # Whole-slide images for development
    │   └── evaluation/
    │       ├── rois/      # PNG images of ROIs for evaluation
    │       └── wsis/      # Whole-slide images for evaluation
    │
    └── model/             # Weights of the final ensemble model used for technical validation
```

* The [`code/`](code/) contains Python code for running inference on the evaluation set.

We describe additional details regarding the datasets on our [Zenodo data repository](https://zenodo.org/records/16812932).

### Quickstart guide
1. Download all data from Zenodo by running the `download_all.sh` shell script. All data is automatically organized in the directory layout as described above.
2. To run code for running inference, build the [Docker](code/docker/Dockerfile). Then use the script: [`code/docker/run_inference.sh`](code/docker/run_inference.sh)

### Citation & license
This GitHub repository is released under the [Apache-2.0 license](LICENSE). The data of the BEETLE dataset is released under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

If you use this dataset, please cite:
```
<PREPRINT FORTHCOMING>
```
