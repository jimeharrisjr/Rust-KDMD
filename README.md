# Rust Koopman DMD

A Rust implementation of Koopman Dynamic Mode Decomposition, providing equivalent functionality to the R rdmd package.

## Project Structure

- `rdmd.r` - Original R implementation with Koopman DMD functions
- `my_library/` - Rust library implementation
  - `src/lib.rs` - Main library code with KDMD functions
  - `examples/` - Usage examples
  - `README.md` - Library-specific documentation

## Features

- **Koopman Matrix Generation**: Create Koopman matrices from time series data using SVD
- **Future State Prediction**: Predict future time steps using learned Koopman operators
- **Component Selection**: Flexible control over SVD components and explained variance
- **Error Handling**: Comprehensive error handling with descriptive messages

## Quick Start

```bash
# Build the library
cd my_library
cargo build

# Run tests
cargo test

# Run example
cargo run --example basic_usage
```

## Functions

### R Functions (rdmd.r)
- `kdmd()` - Create KDMD object wrapper
- `getAMatrix()` - Generate Koopman matrix from data
- `predict.kdmd()` - Predict future states

### Rust Functions (my_library)
- `Kdmd::new()` - Create KDMD object
- `get_a_matrix()` - Generate Koopman matrix from data  
- `predict_kdmd()` - Predict future states

## Dependencies

- nalgebra - Linear algebra operations
- num-complex - Complex number support

## License

This project is open source.
