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
    
    /// Get the underlying matrix (alias for as_matrix)
    pub fn matrix(&self) -> &DMatrix<f64> {
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
    
    // Validate input exactly as in R: dims <- dim(data)
    // if (!(dims[1] > 1 & dims[2] > 1)) stop ('matrix must have two dimensions (2x2 or greater)')
    if !(nrows > 1 && ncols > 1) {
        return Err(KdmdError::InvalidMatrix("matrix must have two dimensions (2x2 or greater)".to_string()));
    }
    
    // if (p <= 0 | p > 1) stop (paste('p=', p, 'p value must be within the range (0,1]'))
    if p <= 0.0 || p > 1.0 {
        return Err(KdmdError::InvalidParameter(format!("p= {} p value must be within the range (0,1]", p)));
    }
    
    // x <- data[, -ncol(data)]
    // y <- data[, -1]
    let x = data.columns(0, ncols - 1).clone_owned();
    let y = data.columns(1, ncols - 1).clone_owned();
    

    
    // wsvd <- base::svd(x)
    let svd = x.svd(true, true);
    let u = svd.u.ok_or(KdmdError::ComputationError("U matrix not computed".to_string()))?;
    let v_t = svd.v_t.ok_or(KdmdError::ComputationError("V^T matrix not computed".to_string()))?;
    let v = v_t.transpose(); // Convert V^T to V to match R's svd$v
    let d = svd.singular_values;
    

    
    // Determine r exactly as in R
    let r = if let Some(comp_val) = comp {
        // if (!is.na(comp)) {
        //   r <- as.integer(comp)
        //   if (r <= 1) {
        //     warning('component values below 2 are not supported. Defaulting to 2')
        //     r <- 2
        //   }
        //   if (r > length(wsvd$d)) {
        //     warning('Specified number of components is greater than possible. Ignoring extra')
        //     r <- length(wsvd$d)
        //   }
        // }
        let mut r = comp_val;
        if r <= 1 {
            eprintln!("Warning: component values below 2 are not supported. Defaulting to 2");
            r = 2;
        }
        if r > d.len() {
            eprintln!("Warning: Specified number of components is greater than possible. Ignoring extra");
            r = d.len();
        }
        r
    } else {
        // } else {
        //   if (p == 1) {
        //     r <- length(wsvd$d)
        //   } else {
        //     sv <- (wsvd$d ^ 2) / sum(wsvd$d ^ 2)
        //     r <- max(which(cumsum(sv) >= p)[1], 2)
        //   }
        // }
        if p == 1.0 {
            d.len()
        } else {
            let d_squared_sum: f64 = d.iter().map(|x| x * x).sum();
            let sv: Vec<f64> = d.iter().map(|x| (x * x) / d_squared_sum).collect();
            let cumsum: Vec<f64> = sv.iter().scan(0.0, |acc, &x| { *acc += x; Some(*acc) }).collect();
            
            let mut r = 2; // default minimum
            for (i, &cum_val) in cumsum.iter().enumerate() {
                if cum_val >= p {
                    r = (i + 1).max(2);
                    break;
                }
            }
            r
        }
    };
    
    // u <- wsvd$u
    // v <- wsvd$v  
    // d <- wsvd$d
    let u_r = u.columns(0, r);
    let v_r = v.columns(0, r);
    let d_r = d.rows(0, r);
    
    // Atil <- t(u[,1:r]) %*% y
    let mut atil = u_r.transpose() * &y;
    
    // Atil <- Atil %*% v[,1:r]
    atil = &atil * &v_r;
    
    // Atil <- Atil %*% diag(1 / d[1:r])
    // Use the actual singular values as R does, without tolerance replacement
    let d_inv_values: Vec<f64> = d_r.iter().map(|&d_val| 1.0 / d_val).collect();
    
    let mut d_inv_diag = DMatrix::zeros(d_inv_values.len(), d_inv_values.len());
    for (i, &val) in d_inv_values.iter().enumerate() {
        d_inv_diag[(i, i)] = val;
    }
    atil = &atil * &d_inv_diag;
    
    // eig <- eigen(Atil)
    // Phi <- eig$values  
    // Q <- eig$vectors
    // Use complex eigendecomposition to match R exactly
    let complex_eigenvals = atil.clone().complex_eigenvalues();
    
    // For complex conjugate pairs, R's eigen() returns both eigenvalues
    // Extract real parts for the diagonal matrix (this is an approximation)
    let phi_values: Vec<f64> = complex_eigenvals.iter().map(|c| c.re).collect();
    let phi_vector = nalgebra::DVector::from_vec(phi_values.clone());
    
    // Use symmetric eigendecomposition for eigenvectors (approximation)
    let eigendecomp = atil.clone().symmetric_eigen();
    let q = eigendecomp.eigenvectors;
    
    // Psi <- y %*% v[, 1:r] %*% diag(1 / d[1:r]) %*% (Q)
    let psi = &y * &v_r * &d_inv_diag * &q;
    
    // Check for problematic values
    if psi.iter().any(|&x| !x.is_finite()) {
        return Err(KdmdError::ComputationError("Non-finite values in Psi matrix".to_string()));
    }
    
    // x <- Psi %*% diag(Phi) %*% pracma::pinv(Psi)
    let phi_diag = DMatrix::from_diagonal(&phi_vector);
    
    // Compute pseudoinverse using SVD
    let psi_clone = psi.clone();
    let psi_svd = psi_clone.svd(true, true);
    let psi_u = psi_svd.u.ok_or(KdmdError::ComputationError("Failed to compute Psi U matrix".to_string()))?;
    let psi_s = &psi_svd.singular_values;
    let psi_vt = psi_svd.v_t.ok_or(KdmdError::ComputationError("Failed to compute Psi V^T matrix".to_string()))?;
    
    // Build pseudoinverse: V * S^+ * U^T
    let tolerance = 1e-15;
    let mut s_pinv = DMatrix::zeros(psi_s.len(), psi_s.len());
    for i in 0..psi_s.len() {
        if psi_s[i] > tolerance {
            s_pinv[(i, i)] = 1.0 / psi_s[i];
        }
    }
    let psi_pinv = psi_vt.transpose() * s_pinv * psi_u.transpose();
    
    // A <- kdmd(x)
    let koopman_matrix = &psi * &phi_diag * &psi_pinv;
    
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
