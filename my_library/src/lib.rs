use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
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
pub fn get_a_matrix(data: &DMatrix<f64>, p: f64, comp: Option<usize>) -> Result<Kdmd, KdmdError> {
    let (nrows, ncols) = data.shape();
    
    // Validate input - following R code validation
    if nrows < 2 || ncols < 2 {
        return Err(KdmdError::InvalidMatrix("Matrix must have two dimensions (2x2 or greater)".to_string()));
    }
    
    if p <= 0.0 || p > 1.0 {
        return Err(KdmdError::InvalidParameter(format!("p={}, p value must be within the range (0,1]", p)));
    }
    
    // Create X and Y matrices following R: x<-data[,-ncol(data)], y<-data[,-1]
    let x = data.columns(0, ncols - 1).clone_owned();
    let y = data.columns(1, ncols - 1).clone_owned();
    
    // Perform SVD on X: wsvd<-base::svd(x)
    let svd = x.svd(true, true);
    let u = svd.u.ok_or(KdmdError::ComputationError("U matrix not computed".to_string()))?;
    let v_t = svd.v_t.ok_or(KdmdError::ComputationError("V^T matrix not computed".to_string()))?;
    let d = svd.singular_values;
    
    // Determine number of components r following R logic, but handle rank deficiency
    let effective_rank = d.iter().position(|&val| val < 1e-12).unwrap_or(d.len());
    
    let r = if let Some(comp_val) = comp {
        let mut r = comp_val as usize;
        if r <= 1 {
            eprintln!("Warning: component values below 2 are not supported. Defaulting to 2");
            r = 2;
        }
        r.min(effective_rank).max(1) // Ensure we don't exceed effective rank
    } else if (p - 1.0).abs() < f64::EPSILON {
        effective_rank.max(1)
    } else {
        // Calculate explained variance: sv<-(wsvd$d^2)/sum(wsvd$d^2)
        let d_squared_sum: f64 = d.iter().take(effective_rank).map(|val| val * val).sum();
        if d_squared_sum < 1e-15 {
            return Err(KdmdError::ComputationError("Matrix is rank deficient".to_string()));
        }
        
        let mut cumsum = 0.0;
        let mut r = 1; // Start with 1 for rank-deficient case
        
        for (i, &d_val) in d.iter().take(effective_rank).enumerate() {
            cumsum += (d_val * d_val) / d_squared_sum;
            if cumsum >= p {
                r = i + 1;
                break;
            }
        }
        r.max(1).min(effective_rank)
    };
    
    // Extract components: u<-wsvd$u, v<-wsvd$v, d<-wsvd$d
    let u_r = u.columns(0, r);
    let v = v_t.transpose(); // Convert V^T back to V
    let v_r = v.columns(0, r);
    let d_r = d.rows(0, r);
    
    // Follow R algorithm exactly:
    // Atil<-crossprod(u[,1:r],y) = t(u[,1:r]) %*% y
    let mut atil = u_r.transpose() * &y;
    
    // Atil<-crossprod(t(Atil),v[,1:r]) = t(t(Atil)) %*% v[,1:r] = Atil %*% v[,1:r]
    atil = &atil * &v_r;
    
    // Atil<-crossprod(t(Atil),diag(1/d[1:r])) = t(t(Atil)) %*% diag(1/d[1:r]) = Atil %*% diag(1/d[1:r])
    let d_inv_diag = DMatrix::from_diagonal(&d_r.map(|d_val| 1.0 / d_val));
    atil = &atil * &d_inv_diag;
    
    // Eigendecomposition: eig<-eigen(Atil)
    let eigendecomp = atil.clone().complex_eigenvalues();
    let phi = eigendecomp;
    
    // Check for numerical stability
    if atil.iter().any(|&x| !x.is_finite()) {
        return Err(KdmdError::ComputationError("Non-finite values in A_tilde matrix".to_string()));
    }
    
    // Get eigenvectors - try symmetric eigendecomposition first
    let (q, phi_complex) = if atil.is_square() && atil.nrows() > 0 {
        // Try symmetric eigendecomposition for real eigenvalues/vectors
        let symmetric_result = atil.clone().symmetric_eigen();
        let vals: Vec<Complex64> = symmetric_result.eigenvalues.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        (symmetric_result.eigenvectors, vals)
    } else {
        // For non-square matrices or empty matrices, use simpler approach
        let vals = phi.iter().cloned().collect();
        let dim = atil.nrows().min(atil.ncols()).max(1);
        (DMatrix::identity(dim, dim), vals)
    };
    
    // Check dimensions before multiplication
    if y.ncols() != v_r.nrows() || v_r.ncols() != d_inv_diag.nrows() || d_inv_diag.ncols() != q.nrows() {
        return Err(KdmdError::ComputationError(format!(
            "Dimension mismatch in Psi computation: Y({}x{}) * V_r({}x{}) * D_inv({}x{}) * Q({}x{})",
            y.nrows(), y.ncols(), v_r.nrows(), v_r.ncols(), 
            d_inv_diag.nrows(), d_inv_diag.ncols(), q.nrows(), q.ncols()
        )));
    }
    
    // Psi<- y %*% v[,1:r] %*% diag(1/d[1:r]) %*% (Q)
    let psi = &y * &v_r * &d_inv_diag * &q;
    
    // Check for numerical issues
    if psi.iter().any(|&x| !x.is_finite()) {
        return Err(KdmdError::ComputationError("Non-finite values in Psi matrix".to_string()));
    }
    
    // x<-Psi %*% diag(Phi) %*% pracma::pinv(Psi)
    let phi_real: Vec<f64> = phi_complex.iter().map(|c| {
        let real_part = c.re;
        if real_part.is_finite() { real_part } else { 0.0 }
    }).collect();
    
    let phi_diag = DMatrix::from_diagonal(&DVector::from_vec(phi_real));
    
    // Use a higher tolerance for pseudoinverse to avoid numerical issues
    let psi_pinv = match psi.clone().pseudo_inverse(1e-8) {
        Ok(pinv) => pinv,
        Err(_) => {
            // Fallback to simple transpose for square matrices
            if psi.is_square() {
                match psi.clone().try_inverse() {
                    Some(inv) => inv,
                    None => return Err(KdmdError::ComputationError("Matrix inversion failed - singular matrix".to_string())),
                }
            } else {
                return Err(KdmdError::ComputationError("Pseudoinverse computation failed".to_string()));
            }
        }
    };
    
    let koopman_matrix = psi * &phi_diag * &psi_pinv;
    
    // Final check for NaN values
    if koopman_matrix.iter().any(|&x| !x.is_finite()) {
        return Err(KdmdError::ComputationError("Non-finite values in final Koopman matrix".to_string()));
    }
    
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
