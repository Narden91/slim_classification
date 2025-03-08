# MIT License
#
# Copyright (c) 2024 DALabNOVA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Classification module for SLIM_GSGP.

This module adds classification capabilities to the SLIM_GSGP framework,
providing utilities and implementations for both binary and multiclass
classification problems.
"""

from slim_gsgp.classifiers.classification_utils import (
    ClassificationTreeWrapper,
    evaluate_classification_model,
    classification_accuracy_fitness,
    f1_score_fitness,
    binary_cross_entropy_with_logits,
    create_balanced_data,
    calculate_class_weights,
    convert_to_one_vs_rest,
    classification_metrics
)

from slim_gsgp.classifiers.binary_classifiers import (
    BinaryClassifier,
    train_binary_classifier
)

from slim_gsgp.classifiers.multiclass_classifiers import (
    MulticlassClassifier,
    train_multiclass_classifier,
    train_one_vs_rest_classifier,
    train_one_vs_one_classifier
)