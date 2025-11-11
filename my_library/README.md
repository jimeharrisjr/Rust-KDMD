# Koopman Dynamic Mode Decomposition (KDMD) Library

A Rust library for Koopman Dynamic Mode Decomposition, providing equivalent functionality to the R rdmd package.

## Features

- Koopman Matrix Generation using SVD
- Future State Prediction 
- Flexible Component Selection
- Comprehensive Error Handling

## Usage

```rust
use nalgebra::DMatrix;
use my_library::{get_a_matrix, predict_kdmd};

// Create time series data
let data = DMatrix::from_row_slice(3, 5, &[
    1.0, 2.0, 3.0, 4.0, 5.0,
    2.0, 4.0, 6.0, 8.0, 10.0,
    3.0, 6.0, 9.0, 12.0, 15.0,
]);

// Create Koopman matrix
let kdmd = get_a_matrix(&data, 0.9, None)?;

// Predict future steps
let prediction = predict_kdmd(&kdmd, &data, 2)?;
```

## API Functions

- `get_a_matrix(data, p, comp)` - Create Koopman matrix
- `predict_kdmd(kdmd, data, l)` - Predict future states

## Running

```bash
cargo test
cargo run --example basic_usage
```
