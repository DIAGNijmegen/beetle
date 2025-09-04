import numpy as np
import torch
from pathlib import Path
from PIL import Image

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.nnUNetTrainer.variants.pathology.nnUNetTrainer_custom_dataloader_test import (
    nnUNetTrainer_custom_dataloader_test,
)
from nnunetv2.utilities.file_path_utilities import load_json


def norm_01(x_batch: np.ndarray) -> np.ndarray:
    """Normalize image batch from 0-255 to 0-1 and reorder axes to (C, N, H, W)."""
    x_batch = x_batch.astype(np.float32) / 255.0
    return x_batch.transpose(3, 0, 1, 2)


def ensemble_softmax_list(trainer, predictor, patch: np.ndarray) -> list[np.ndarray]:
    """Compute softmax outputs for each fold in the ensemble."""
    patch_tensor = torch.tensor(patch, dtype=torch.float32)
    logits_list = predictor.get_logits_list_from_preprocessed_data(patch_tensor)
    return [trainer.label_manager.apply_inference_nonlin(logits).numpy() for logits in logits_list]


def process_roi_image(trainer, predictor, roi_path: Path, output_folder: Path):
    """Process a single ROI image: load, normalize, predict, and save output."""
    image = Image.open(roi_path)
    patch = np.expand_dims(np.array(image), axis=0)
    patch = norm_01(patch)

    softmax_list = ensemble_softmax_list(trainer, predictor, patch)
    softmax_mean = np.mean(softmax_list, axis=0)
    pred_output = np.squeeze(np.argmax(softmax_mean, axis=0))

    output_path = output_folder / f"{roi_path.stem}.png"
    Image.fromarray(pred_output.astype(np.uint8)).save(output_path)
    print(f"Saved prediction to: {output_path}")


def main(
    model_base_path: str,
    checkpoint_name: str,
    folds_to_use: tuple,
    roi_folder: str,
    output_folder: str,
):
    """Main function for inference."""
    model_base_path = Path(model_base_path)
    roi_folder = Path(roi_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load model metadata
    plans_dict = load_json(model_base_path / "plans.json")
    dataset_dict = load_json(model_base_path / "dataset.json")

    # Initialize trainer and predictor
    trainer = nnUNetTrainer_custom_dataloader_test(plans_dict, "2d", 0, dataset_dict)
    predictor = nnUNetPredictor()
    predictor.initialize_from_trained_model_folder(
        model_base_path,
        use_folds=folds_to_use,
        checkpoint_name=checkpoint_name,
    )

    # Process all ROI images
    for roi_path in roi_folder.glob("*.png"):
        process_roi_image(trainer, predictor, roi_path, output_folder)

    print(f"Inference completed. Predictions saved to: {output_folder}")


if __name__ == "__main__":
    # Paths and parameters
    MODEL_BASE_PATH = "../../data/model/nnUNetTrainer_WSD_wei_i0_nnunet_aug_json__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"
    CHECKPOINT_NAME = "checkpoint_best.pth"
    FOLDS_TO_USE = (0, 1, 2, 3, 4)

    ROI_FOLDER = "../../data/images/evaluation/rois"
    OUTPUT_FOLDER = "../../data/inference"

    MODEL_BASE_PATH = "/data/pa_cpgarchive/projects/breast-epithelium-segmentation/breast-multiclass-segmentation/paper/zenodo/model/nnUNetTrainer_WSD_wei_i0_nnunet_aug_json__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"
    ROI_FOLDER = "/data/pa_cpgarchive/projects/breast-epithelium-segmentation/breast-multiclass-segmentation/paper/zenodo/images/evaluation/rois"
    OUTPUT_FOLDER = "/data/temporary/carlijn"

    print("Running inference...")
    main(
        model_base_path=MODEL_BASE_PATH,
        checkpoint_name=CHECKPOINT_NAME,
        folds_to_use=FOLDS_TO_USE,
        roi_folder=ROI_FOLDER,
        output_folder=OUTPUT_FOLDER,
    )
