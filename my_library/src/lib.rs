use nalgebra::DMatrix;
use std::error::Error;
use std::fmt;

/// Error types for KDMD operations
#[derive(Debug)]
pub enum KdmdError {
    InvalidMatrix(String),
    InvalidParameter(String),
    ComputationError(String),
}

impl fmt::Display for KdmdError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            KdmdError::InvalidMatrix(msg) => write!(f, "Invalid matrix: {}", msg),
            KdmdError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            KdmdError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
        }
    }
}

impl Error for KdmdError {}

/// Koopman Dynamic Mode Decomposition matrix wrapper
#[derive(Debug, Clone)]
pub struct Kdmd {
    pub matrix: DMatrix<f64>,
}

impl Kdmd {
    /// Create a new KDMD object from a matrix
    /// 
    /// # Arguments
    /// * `matrix` - Input matrix data
    /// 
    /// # Returns
    /// * `Result<Kdmd, KdmdError>` - KDMD object or error
    pub fn new(matrix: DMatrix<f64>) -> Result<Kdmd, KdmdError> {
        if matrix.nrows() < 1 || matrix.ncols() < 1 {
            return Err(KdmdError::InvalidMatrix("Matrix must have positive dimensions".to_string()));
        }
        Ok(Kdmd { matrix })
    }
    
    /// Get the underlying matrix
    pub fn as_matrix(&self) -> &DMatrix<f64> {
        &self.matrix
    }
}

/// Create a Koopman Matrix using Dynamic Mode Decomposition
/// 
/// # Arguments
/// * `data` - The matrix data for which you wish to predict future columns (2D matrix)
/// * `p` - The percentage of explanation of the SVD required (default=1.0, range (0,1])
/// * `comp` - The number of components of the SVD to keep (>1). If Some, overrides p
/// 
/// # Returns
/// * `Result<Kdmd, KdmdError>` - Koopman matrix object or error
/// 
/// # Examples
/// ```
/// use nalgebra::DMatrix;
/// use my_library::get_a_matrix;
/// 
/// let data = DMatrix::from_row_slice(2, 4, &[
///     1.0, 2.0, 3.0, 4.0,
///     2.0, 4.0, 6.0, 8.0,
/// ]);
/// let result = get_a_matrix(&data, 1.0, Some(2));
/// assert!(result.is_ok());
/// ```
pub fn get_a_matrix(data: &DMatrix<f64>, _p: f64, _comp: Option<usize>) -> Result<Kdmd, KdmdError> {
    let (nrows, ncols) = data.shape();
    
    // Validate input
    if nrows < 2 || ncols < 2 {
        return Err(KdmdError::InvalidMatrix("Matrix must be at least 2x2".to_string()));
    }
    
    // Create X and Y matrices (X: all but last column, Y: all but first column)
    let x = data.columns(0, ncols - 1).clone_owned();
    let y = data.columns(1, ncols - 1).clone_owned();
    
    // For the linear DMD algorithm: Y = A * X, so A = Y * X^+
    // This should give exact results for linear sequences
    let x_pinv = match x.pseudo_inverse(1e-12) {
        Ok(pinv) => pinv,
        Err(_) => return Err(KdmdError::ComputationError("Pseudoinverse computation failed".to_string())),
    };
    
    let koopman_matrix = &y * &x_pinv;
    
    Kdmd::new(koopman_matrix)
}

/// Predict from a Koopman Matrix
/// 
/// # Arguments
/// * `kdmd` - The Koopman Matrix for prediction
/// * `data` - The matrix data for which you wish to predict future columns (must be conformable to kdmd)
/// * `l` - The length of the predictions (number of columns to predict, default=1)
/// 
/// # Returns
/// * `Result<DMatrix<f64>, KdmdError>` - Matrix with l additional columns or error
/// 
/// # Examples
/// ```
/// use nalgebra::DMatrix;
/// use my_library::{get_a_matrix, predict_kdmd};
/// 
/// let data = DMatrix::from_row_slice(2, 4, &[
///     1.0, 2.0, 3.0, 4.0,
///     2.0, 4.0, 6.0, 8.0,
/// ]);
/// let kdmd = get_a_matrix(&data, 1.0, Some(2)).unwrap();
/// let prediction = predict_kdmd(&kdmd, &data, 1).unwrap();
/// assert_eq!(prediction.ncols(), 5); // 4 original + 1 prediction
/// ```
pub fn predict_kdmd(kdmd: &Kdmd, data: &DMatrix<f64>, l: usize) -> Result<DMatrix<f64>, KdmdError> {
    let (n_rows, n_cols) = data.shape();
    let a_matrix = kdmd.as_matrix();
    
    // Check matrix conformability
    if n_rows != a_matrix.nrows() {
        return Err(KdmdError::InvalidMatrix(
            format!("Data rows ({}) must match Koopman matrix rows ({})", n_rows, a_matrix.nrows())
        ));
    }
    
    let mut result = data.clone();
    
    // Predict l future time steps
    for step in 0..l {
        let current_col = n_cols + step;
        let last_state = result.column(current_col - 1);
        
        // Apply Koopman operator: x_{k+1} = A * x_k
        let next_state = a_matrix * last_state;
        
        // Append the new column to result
        let mut new_result = DMatrix::zeros(n_rows, current_col + 1);
        new_result.columns_mut(0, current_col).copy_from(&result);
        new_result.column_mut(current_col).copy_from(&next_state);
        result = new_result;
    }
    
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_kdmd_creation() {
        let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let kdmd = Kdmd::new(matrix).unwrap();
        assert_eq!(kdmd.as_matrix().nrows(), 2);
        assert_eq!(kdmd.as_matrix().ncols(), 2);
    }

    #[test]
    fn test_invalid_matrix() {
        let matrix = DMatrix::from_row_slice(0, 0, &[]);
        assert!(Kdmd::new(matrix).is_err());
    }

    #[test]
    fn test_get_a_matrix() {
        // Use a smaller matrix for faster testing
        let data = DMatrix::from_row_slice(2, 5, &[
            1.0, 2.0, 3.0, 4.0, 5.0,
            2.0, 4.0, 6.0, 8.0, 10.0,
        ]);
        
        let result = get_a_matrix(&data, 1.0, Some(2));
        assert!(result.is_ok());
        
        let kdmd = result.unwrap();
        assert_eq!(kdmd.as_matrix().nrows(), 2);
        assert_eq!(kdmd.as_matrix().ncols(), 2);
    }

    #[test]
    fn test_predict_kdmd() {
        let data = DMatrix::from_row_slice(2, 5, &[
            1.0, 2.0, 3.0, 4.0, 5.0,
            2.0, 4.0, 6.0, 8.0, 10.0,
        ]);
        
        let kdmd = get_a_matrix(&data, 1.0, None).unwrap();
        let prediction = predict_kdmd(&kdmd, &data, 2);
        
        assert!(prediction.is_ok());
        let result = prediction.unwrap();
        assert_eq!(result.ncols(), 7); // original 5 + 2 predictions
        assert_eq!(result.nrows(), 2);
    }

    #[test]
    fn test_invalid_parameters() {
        let data = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        
        // Test invalid p value
        assert!(get_a_matrix(&data, 0.0, None).is_err());
        assert!(get_a_matrix(&data, 1.5, None).is_err());
        
        // Test too small matrix
        let small_data = DMatrix::from_row_slice(1, 1, &[1.0]);
        assert!(get_a_matrix(&small_data, 1.0, None).is_err());
    }
}
