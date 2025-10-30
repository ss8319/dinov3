# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import sys
import time
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed
from omegaconf import MISSING
from torch import nn
from torch.utils.data import TensorDataset
from torchmetrics import MetricTracker

from dinov3.data import SamplerType, make_data_loader, make_dataset
from dinov3.data.adapters import DatasetWithEnumeratedTargets
from dinov3.data.transforms import CROP_DEFAULT_SIZE, get_target_transform, make_classification_eval_transform
from dinov3.distributed import get_rank, get_world_size
from dinov3.eval.data import (
    create_train_dataset_dict,
    extract_features_for_dataset_dict,
    get_num_classes,
    split_train_val_datasets,
)
from dinov3.eval.helpers import args_dict_to_dataclass, cli_parser, write_results
from dinov3.eval.metrics import ClassificationMetricType, build_classification_metric
from dinov3.eval.setup import ModelConfig, load_model_and_context
from dinov3.eval.utils import average_metrics, evaluate, extract_features
from dinov3.eval.utils import save_results as default_save_results_func
from dinov3.run.init import job_context
from dinov3.utils.dtype import as_torch_dtype

logger = logging.getLogger("dinov3")


RESULTS_FILENAME = "results-log-regression.csv"
MAIN_METRICS = ["top-1(_mean)?"]


try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ImportError:
    logger.warning("Can't import sklearnex. If installed, that speeds up scikit-learn 10-100x")

try:
    from sklearn.linear_model import LogisticRegression as sklearnLogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
except ImportError:
    logger.warning("Can't import scikit-learn. This is necessary for evaluating log regression")
    raise ImportError


# Reduce sweep size for faster dev runs; adjust to your needs
C_POWER_RANGE = torch.linspace(-6, 5, 13)
_CPU_DEVICE = torch.device("cpu")


@dataclass
class TrainConfig:
    dataset: str = MISSING  # train dataset path
    val_dataset: Optional[str] = None  # val dataset path. If None, choose hyperparameters on 10% of the train set.
    val_metric_type: ClassificationMetricType = ClassificationMetricType.MEAN_ACCURACY
    batch_size: int = 256  # batch size for train and val set feature extraction
    num_workers: int = 5  # number of workers for train and val set feature extraction
    tol: float = 1e-12  # tolerance in logistic regression
    train_features_device: str = "cpu"  # device to gather train features (cpu, cuda, cuda:0, etc.)
    train_dtype: str = "float64"  # data type to convert the train features to
    max_train_iters: int = 1_000  # maximum number of train iterations in logistic regression


@dataclass
class EvalConfig:
    test_dataset: str = MISSING  # test dataset path
    batch_size: int | None = None  # use train.batch_size if None
    num_workers: int = 5
    test_metric_type: Optional[ClassificationMetricType] = None


@dataclass
class TransformConfig:
    resize_size: int = CROP_DEFAULT_SIZE
    crop_size: int = CROP_DEFAULT_SIZE


@dataclass
class FewShotConfig:
    enable: bool = False  # whether to use few-shot evaluation
    k_or_percent: Optional[float] = None  # number of elements or % to take per class
    n_tries: int = 1  # number of tries for few-shot evaluation


@dataclass
class LogregEvalConfig:
    model: ModelConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    few_shot: FewShotConfig = field(default_factory=FewShotConfig)
    save_results: bool = False  # save predictions and targets in the output directory
    output_dir: str = ""


class LogRegModule(nn.Module):
    def __init__(self, C, multi_label=False, logreg_config=TrainConfig):
        super().__init__()
        self.dtype = as_torch_dtype(logreg_config.train_dtype)
        self.device = torch.device(logreg_config.train_features_device)
        assert self.device == _CPU_DEVICE, f"SKLearn can only work on CPU device, got {self.device}"
        self.estimator = sklearnLogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=C,
            max_iter=logreg_config.max_train_iters,
            n_jobs=-1,
            tol=logreg_config.tol,
        )
        if multi_label:
            self.estimator = OneVsRestClassifier(self.estimator, n_jobs=-1)

    def forward(self, samples, targets):
        samples_device = samples.device
        samples = samples.to(dtype=self.dtype, device=self.device)
        if self.device == _CPU_DEVICE:
            samples = samples.numpy()
        probas = self.estimator.predict_proba(samples)
        return {"preds": torch.from_numpy(probas).to(samples_device), "target": targets}

    def fit(self, train_features, train_labels):
        train_features = train_features.to(dtype=self.dtype, device=self.device)
        train_labels = train_labels.to(dtype=self.dtype, device=self.device)
        if self.device == _CPU_DEVICE:
            # both cuml and sklearn only work with numpy arrays on CPU
            train_features = train_features.numpy()
            train_labels = train_labels.numpy()
        self.estimator.fit(train_features, train_labels)


def evaluate_logreg_model(*, logreg_model, test_metric, test_data_loader, save_results_func=None, verbose=False):
    key = "metrics"  # We need only one key as we have only one metric
    postprocessors, metrics = {key: logreg_model}, {key: test_metric}
    _, eval_metrics, accumulated_results = evaluate(
        nn.Identity(),
        test_data_loader,
        postprocessors,
        metrics,
        torch.cuda.current_device(),
        accumulate_results=save_results_func is not None,
    )
    if save_results_func is not None:
        save_results_func(**accumulated_results[key])
    
    # Compute additional binary classification metrics
    from sklearn.metrics import (
        balanced_accuracy_score,
        matthews_corrcoef,
        roc_auc_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix
    )
    
    # NumPy is already imported at module level, but we need to ensure it's accessible
    import numpy as np
    
    # Collect all predictions and targets
    logreg_model.eval()
    all_preds = []
    all_targets = []
    all_probas = []
    
    with torch.no_grad():
        for batch in test_data_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                samples, targets = batch
            else:
                samples = batch
                targets = None
            
            # Get predictions
            output = logreg_model(samples, targets)
            probas = output["preds"]
            preds = probas.argmax(dim=-1)
            
            all_preds.append(preds.cpu())
            all_probas.append(probas.cpu())
            
            if targets is not None:
                # Handle DatasetWithEnumeratedTargets collate behavior and other target shapes
                batch_size = preds.shape[0]
                targets_tensor: Optional[torch.Tensor] = None

                if isinstance(targets, torch.Tensor):
                    # Single tensor of labels
                    if targets.numel() == 1:
                        targets_tensor = targets.reshape(1).repeat(batch_size)
                    else:
                        targets_tensor = targets.view(-1)
                elif isinstance(targets, (list, tuple)):
                    # torch default collate over (index, label) tuples produces a tuple of two tensors
                    if (
                        len(targets) == 2
                        and isinstance(targets[0], torch.Tensor)
                        and isinstance(targets[1], torch.Tensor)
                        and targets[0].numel() == targets[1].numel()
                    ):
                        # Use the second tensor which contains labels
                        targets_tensor = targets[1].view(-1)
                    elif len(targets) > 0 and isinstance(targets[0], (tuple, list)) and len(targets[0]) == 2:
                        # List of (index, label) pairs
                        labels = [t[1] for t in targets]
                        if len(labels) > 0 and isinstance(labels[0], torch.Tensor):
                            stacked = [t.reshape(-1) for t in labels]
                            targets_tensor = torch.cat(stacked).view(-1)
                        else:
                            targets_tensor = torch.tensor(labels).view(-1)
                    elif len(targets) > 0 and isinstance(targets[0], torch.Tensor):
                        # List/tuple of tensors; assume already labels
                        stacked = [t.reshape(-1) for t in targets]
                        targets_tensor = torch.cat(stacked).view(-1)
                    else:
                        # Fallback: list of python scalars
                        targets_tensor = torch.tensor(list(targets)).view(-1)
                else:
                    # Python scalar
                    targets_tensor = torch.tensor([targets]).repeat(batch_size)

                # Align length to batch size (truncate or pad last value)
                if targets_tensor is not None:
                    if targets_tensor.numel() < batch_size:
                        pad_val = targets_tensor[-1] if targets_tensor.numel() > 0 else torch.tensor(0)
                        pad_count = batch_size - targets_tensor.numel()
                        pad = pad_val.repeat(pad_count)
                        targets_tensor = torch.cat([targets_tensor, pad])
                    elif targets_tensor.numel() > batch_size:
                        targets_tensor = targets_tensor[:batch_size]
                    all_targets.append(targets_tensor.cpu())
    
    if len(all_targets) > 0:
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_probas = torch.cat(all_probas)
        
        # Convert to numpy for sklearn
        preds_np = all_preds.numpy()
        targets_np = all_targets.numpy()
        probas_full = all_probas.numpy()  # (N, C)
        
        # Calculate metrics with robust error handling
        unique_classes = np.unique(targets_np)
        
        # Confusion matrix for detailed analysis (only print if verbose)
        if verbose:
            try:
                cm = confusion_matrix(targets_np, preds_np, labels=sorted(unique_classes.tolist()))
                class_names = [f"Class_{i}" for i in sorted(unique_classes.tolist())]
                logger.info("=" * 60)
                logger.info("Confusion Matrix:")
                logger.info(f"                 Predicted")
                logger.info(f"                 {'   '.join(class_names)}")
                for i, row in enumerate(cm):
                    logger.info(f"Actual {class_names[i]}: {'   '.join([f'{val:3d}' for val in row])}")
                logger.info("=" * 60)
                logger.info(f"Actual class distribution: {dict(zip([f'Class_{i}' for i in unique_classes], np.bincount(targets_np.astype(int))))}")
                logger.info(f"Predicted class distribution: {dict(zip([f'Class_{i}' for i in unique_classes], np.bincount(preds_np.astype(int))))}")
                logger.info("=" * 60)
            except Exception as e:
                logger.warning(f"Failed to compute confusion matrix: {e}")
        
        # Balanced accuracy
        try:
            balanced_acc = balanced_accuracy_score(targets_np, preds_np)
        except Exception as e:
            logger.warning(f"Failed to compute balanced_accuracy: {e}")
            balanced_acc = float("nan")
        
        # MCC (handles all cases automatically)
        try:
            mcc = matthews_corrcoef(targets_np, preds_np)
        except Exception as e:
            logger.warning(f"Failed to compute mcc: {e}")
            mcc = float("nan")
        
        # AUC-ROC: handle binary vs multiclass and tiny subsets
        try:
            if probas_full.ndim == 2 and probas_full.shape[1] == 2 and len(unique_classes) == 2:
                # Binary: use positive class probabilities
                auc_roc = roc_auc_score(targets_np, probas_full[:, 1])
            elif probas_full.ndim == 2 and probas_full.shape[1] > 2 and len(unique_classes) > 2:
                # Multiclass: use OvR on full probability matrix
                auc_roc = roc_auc_score(targets_np, probas_full, multi_class="ovr")
            else:
                # Undefined AUC (e.g., single-class tiny split)
                auc_roc = float("nan")
        except Exception as e:
            logger.warning(f"Failed to compute auc_roc: {e}")
            auc_roc = float("nan")
        
        # Determine average method based on number of classes
        num_classes = len(unique_classes)
        avg_method = 'binary' if num_classes == 2 else 'macro'
        
        # F1, Precision, Recall
        try:
            f1 = f1_score(targets_np, preds_np, average=avg_method)
        except Exception as e:
            logger.warning(f"Failed to compute f1: {e}")
            f1 = float("nan")
        
        try:
            precision = precision_score(targets_np, preds_np, average=avg_method)
        except Exception as e:
            logger.warning(f"Failed to compute precision: {e}")
            precision = float("nan")
        
        try:
            recall = recall_score(targets_np, preds_np, average=avg_method)
        except Exception as e:
            logger.warning(f"Failed to compute recall: {e}")
            recall = float("nan")
        
        # Add to eval_metrics (only add if not NaN to avoid issues)
        eval_metrics["metrics"]["balanced_accuracy"] = torch.tensor(balanced_acc if not np.isnan(balanced_acc) else 0.0)
        eval_metrics["metrics"]["mcc"] = torch.tensor(mcc if not np.isnan(mcc) else 0.0)
        eval_metrics["metrics"]["auc_roc"] = torch.tensor(auc_roc if not np.isnan(auc_roc) else 0.0)
        eval_metrics["metrics"]["f1"] = torch.tensor(f1 if not np.isnan(f1) else 0.0)
        eval_metrics["metrics"]["precision"] = torch.tensor(precision if not np.isnan(precision) else 0.0)
        eval_metrics["metrics"]["recall"] = torch.tensor(recall if not np.isnan(recall) else 0.0)
    
    return eval_metrics


def train_for_C(*, C, train_features, train_labels, logreg_config: TrainConfig):
    logreg_model = LogRegModule(C, multi_label=len(train_labels.shape) > 1, logreg_config=logreg_config)
    logreg_model.fit(train_features, train_labels)
    return logreg_model


def sweep_C_values(
    *,
    train_features,
    train_labels,
    val_data_loader,
    val_metric,
    logreg_config: TrainConfig,
):
    metric_tracker = MetricTracker(val_metric, maximize=True)
    ALL_C = 10**C_POWER_RANGE
    logreg_models: Dict[float, Any] = {}

    train_features_device = torch.device(logreg_config.train_features_device)
    train_dtype = as_torch_dtype(logreg_config.train_dtype)
    train_features = train_features.to(dtype=train_dtype, device=train_features_device)
    train_labels = train_labels.to(device=train_features_device)

    for i in range(get_rank(), len(ALL_C), get_world_size()):
        C = ALL_C[i].item()
        logger.info(
            f"Training for C = {C:.4g}, dtype={train_dtype}, "
            f"features: {train_features.shape}, {train_features.dtype}, "
            f"labels: {train_labels.shape}, {train_labels.dtype}"
        )
        logreg_models[C] = train_for_C(
            C=C,
            train_features=train_features,
            train_labels=train_labels,
            logreg_config=logreg_config,
        )

    gather_list: List[Dict[float, Any]] = [{} for _ in range(get_world_size())]
    torch.distributed.all_gather_object(gather_list, logreg_models)

    for logreg_dict in gather_list:
        logreg_models.update(logreg_dict)
    gather_list.clear()

    for i in range(len(ALL_C)):
        metric_tracker.increment()
        C = ALL_C[i].item()
        evals = evaluate_logreg_model(
            logreg_model=logreg_models.pop(C),
            test_metric=metric_tracker,
            test_data_loader=val_data_loader,
        )
        logger.info(f"Trained for C = {C:.4g}, accuracies = {evals}")
        best_stats, which_epoch = metric_tracker.best_metric(return_step=True)
        best_stats_100 = {k: 100.0 * v for k, v in best_stats.items()}
        if which_epoch["top-1"] == i:
            best_C = C
    logger.info(f"Sweep best {best_stats_100}, best C = {best_C:.4g}")

    return best_stats, best_C


def make_logreg_data_loader(batch_size: int, num_workers: int, features: torch.Tensor, labels: torch.Tensor):
    return make_data_loader(
        dataset=DatasetWithEnumeratedTargets(
            TensorDataset(features, labels), pad_dataset=True, num_replicas=get_world_size()
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
    )


def get_best_logreg_with_features(
    *,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    val_metric,
    concatenate_train_val: bool,
    train_config: TrainConfig,
    output_dir: Optional[str] = None,
):
    val_data_loader = make_logreg_data_loader(
        train_config.batch_size, train_config.num_workers, val_features, val_labels
    )
    _, best_C_t = sweep_C_values(
        train_features=train_features,
        train_labels=train_labels,
        val_data_loader=val_data_loader,
        val_metric=val_metric,
        logreg_config=train_config,
    )
    if concatenate_train_val:
        logger.info("Best parameter found, concatenating features")
        train_features = torch.cat((train_features, val_features))
        train_labels = torch.cat((train_labels, val_labels))

    logger.info("Training final model")

    logreg_model = train_for_C(
        C=best_C_t,
        logreg_config=train_config,
        train_features=train_features,
        train_labels=train_labels,
    )
    
    # Save checkpoint if output_dir is provided
    if output_dir and concatenate_train_val and train_config.train_features_device == "cpu":
        try:
            import joblib
            import os
            os.makedirs(output_dir, exist_ok=True)
            checkpoint_path = os.path.join(output_dir, "final_logreg_model.pkl")
            joblib.dump(logreg_model.estimator, checkpoint_path)
            logger.info(f"Saved final logistic regression model to {checkpoint_path}")
        except ImportError:
            logger.warning("joblib not available, skipping model checkpoint")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    return logreg_model


def make_transform(config: TransformConfig):
    if config.resize_size / config.crop_size != 1:
        logger.warning(f"Default resize / crop ratio is 1, here we have {config.resize_size} / {config.crop_size}")
    transform = make_classification_eval_transform(resize_size=config.resize_size, crop_size=config.crop_size)
    return transform


def make_train_val_datasets(train_config: TrainConfig, few_shot_config: FewShotConfig, transform):
    train_dataset = make_dataset(
        dataset_str=train_config.dataset,
        transform=transform,
        target_transform=get_target_transform(train_config.dataset),
    )
    if train_config.val_dataset is not None:
        val_dataset = make_dataset(
            dataset_str=train_config.val_dataset,
            transform=transform,
            target_transform=get_target_transform(train_config.val_dataset),
        )
    else:
        split_percentage = 0.01 if few_shot_config.enable else 0.1
        train_dataset, val_dataset = split_train_val_datasets(train_dataset, split_percentage=split_percentage)

    train_dataset_dict = create_train_dataset_dict(
        train_dataset,
        few_shot_eval=few_shot_config.enable,
        few_shot_k_or_percent=few_shot_config.k_or_percent,
        few_shot_n_tries=few_shot_config.n_tries,
    )
    num_classes = get_num_classes(train_dataset)
    return train_dataset_dict, val_dataset, num_classes


def make_test_dataset_and_data_loader(model, config: EvalConfig, transform, gather_on_cpu: bool):
    test_dataset = make_dataset(
        dataset_str=config.test_dataset,
        transform=transform,
        target_transform=get_target_transform(config.test_dataset),
    )
    test_features, test_labels = extract_features(
        model, test_dataset, config.batch_size, config.num_workers, gather_on_cpu=gather_on_cpu
    )
    assert isinstance(config.batch_size, int)  # eval batch size has been replaced by train batch size if None
    test_data_loader = make_logreg_data_loader(config.batch_size, config.num_workers, test_features, test_labels)
    return test_dataset, test_data_loader


def eval_log_regression_with_model(*, model: torch.nn.Module, autocast_dtype, config: LogregEvalConfig):
    """
    Implements the "standard" process for log regression evaluation:
    The value of C is chosen by training on train_dataset and evaluating on
    val_dataset. Then, the final model is trained on a concatenation of
    train_dataset and val_dataset, and is evaluated on test_dataset.
    If there is no val_dataset, the value of C is the one that yields
    the best results on a random 10% subset of the train dataset
    """
    start = time.time()
    cudnn.benchmark = True

    transform = make_transform(config.transform)
    config.eval.batch_size = config.eval.batch_size or config.train.batch_size  # use train batch size for eval if None

    # Setting up train and val datasets
    train_dataset_dict, val_dataset, num_classes = make_train_val_datasets(config.train, config.few_shot, transform)

    # Extracting features
    with torch.autocast("cuda", dtype=autocast_dtype):
        gather_on_cpu = torch.device(config.train.train_features_device) == _CPU_DEVICE
        train_data_dict = extract_features_for_dataset_dict(
            model, train_dataset_dict, config.train.batch_size, config.train.num_workers, gather_on_cpu=gather_on_cpu
        )
        logger.info("Choosing hyperparameters on the val dataset")
        val_features, val_labels = extract_features(
            model, val_dataset, config.train.batch_size, config.train.num_workers, gather_on_cpu=gather_on_cpu
        )
        test_dataset, test_data_loader = make_test_dataset_and_data_loader(model, config.eval, transform, gather_on_cpu)

    # Moves the model to cpu in-place. Deleting the variable would only delete a reference and not free any space.
    model.cpu()  # all features are extracted, we won't use the backbone anymore
    torch.cuda.empty_cache()

    # Setting up metrics (ensure at least 2 classes for binary tasks on tiny subsets)
    effective_num_classes = max(2, int(num_classes))
    val_metric = build_classification_metric(
        config.train.val_metric_type, num_classes=effective_num_classes, dataset=val_dataset
    )
    test_metric_type = config.eval.test_metric_type or config.train.val_metric_type
    test_metric = build_classification_metric(
        test_metric_type, num_classes=effective_num_classes, dataset=test_dataset
    )

    # Setting up save results function
    save_results_func = None
    if config.save_results:
        save_results_func = partial(default_save_results_func, output_dir=config.output_dir)

    results_dict = {}
    for _try in train_data_dict.keys():
        # Always train sklearn on CPU regardless of feature extraction device since sklearn only works for CPU
        cpu_train_config = replace(config.train, train_features_device="cpu")
        logreg_model = get_best_logreg_with_features(
            train_features=train_data_dict[_try]["train_features"],
            train_labels=train_data_dict[_try]["train_labels"],
            val_features=val_features,
            val_labels=val_labels,
            val_metric=val_metric,
            concatenate_train_val=not config.few_shot.enable,
            train_config=cpu_train_config,
            output_dir=config.output_dir,
        )
        if len(train_data_dict) > 1 and save_results_func is not None:  # add suffix
            split_results_saver = partial(save_results_func, filename_suffix=str(_try))
        else:
            split_results_saver = save_results_func  # type: ignore

        eval_metrics = evaluate_logreg_model(
            logreg_model=logreg_model,
            test_metric=test_metric.clone(),
            test_data_loader=test_data_loader,
            save_results_func=split_results_saver,
            verbose=True,  # Print confusion matrix for final test evaluation
        )
        results_dict[_try] = {k: v.item() * 100.0 for k, v in eval_metrics["metrics"].items()}

    if len(train_data_dict) > 1:
        results_dict = average_metrics(results_dict)
    else:
        results_dict = {**results_dict[_try]}

    logger.info(f"Log regression evaluation done in {int(time.time() - start)}s")
    logger.info("Training of the supervised logistic regression on frozen features completed.")
    results_string = "\n".join([f"{k}: {results_dict[k]:.4g}" for k in sorted(results_dict.keys())])
    logger.info("Results:\n" + results_string)

    torch.distributed.barrier()
    return results_dict


def benchmark_launcher(eval_args: dict[str, object]) -> dict[str, Any]:
    """Initialization of distributed and logging are preconditions for this method"""
    dataclass_config, output_dir = args_dict_to_dataclass(eval_args=eval_args, config_dataclass=LogregEvalConfig)
    model, model_context = load_model_and_context(dataclass_config.model, output_dir=output_dir)
    results_dict = eval_log_regression_with_model(
        model=model, config=dataclass_config, autocast_dtype=model_context["autocast_dtype"]
    )
    write_results(results_dict, output_dir, RESULTS_FILENAME)
    return results_dict


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    eval_args = cli_parser(argv)
    with job_context(output_dir=eval_args["output_dir"]):
        benchmark_launcher(eval_args=eval_args)
    return 0


if __name__ == "__main__":
    main()
