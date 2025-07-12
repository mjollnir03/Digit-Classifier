# MNIST Data Conversion Summary

This note explains how the original `mnist.pkl.gz` file was converted into a `.npz` format using NumPy for easier and safer access.

---

## Original File

- **File**: `mnist.pkl.gz`
- **Format**: gzip-compressed Python `pickle`
- **Contents**: 3 tuples → `(images, labels)` for:
  - Training set
  - Validation set
  - Test set

Each `images` array is shaped `(N, 784)` — where each row is a flattened 28×28 grayscale image.  
Each `labels` array is shaped `(N,)` and contains digit labels (`0–9`).

---

## Converted File

- **File**: `mnist.npz`
- **Format**: NumPy ZIP archive
- **Saved using**: `np.savez_compressed(...)`

### Saved Keys and Their Contents

| Key Name                  | Type             | Shape         | Description                        |
|---------------------------|------------------|---------------|------------------------------------|
| `training_data_images`    | `np.ndarray`     | `(50000, 784)`| Flattened training images          |
| `training_data_labels`    | `np.ndarray`     | `(50000,)`    | Labels for training images         |
| `validation_data_images`  | `np.ndarray`     | `(10000, 784)`| Flattened validation images        |
| `validation_data_labels`  | `np.ndarray`     | `(10000,)`    | Labels for validation images       |
| `test_data_images`        | `np.ndarray`     | `(10000, 784)`| Flattened test images              |
| `test_data_labels`        | `np.ndarray`     | `(10000,)`    | Labels for test images             |

---

## Notes

- All arrays are `numpy.ndarray` types.
- Image arrays use pixel values scaled between 0.0 and 1.0 (float32).
- Label arrays contain integer class labels from 0 to 9.

---

## Purpose of Conversion

- `.npz` is faster and safer to load than `pickle`.
- Native NumPy format — no decoding or class structures needed.
- Easier to share, inspect, and load in any Python environment.
