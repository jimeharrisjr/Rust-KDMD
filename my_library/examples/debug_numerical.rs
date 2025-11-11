use nalgebra::DMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Test case with rank deficiency
    let data = DMatrix::from_row_slice(2, 4, &[
        1.0, 2.0, 3.0, 4.0,
        2.0, 4.0, 6.0, 8.0,
    ]);
    
    println!("Data:");
    println!("{:.3}", data);
    
    let x = data.columns(0, 3).clone_owned();
    let y = data.columns(1, 3).clone_owned();
    
    println!("X:");
    println!("{:.3}", x);
    println!("Y:");
    println!("{:.3}", y);
    
    // SVD
    let svd = x.svd(true, true);
    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();
    let d = svd.singular_values;
    
    println!("Singular values: {:?}", d.as_slice());
    
    // Check for near-zero singular values
    let r = d.iter().position(|&val| val < 1e-10).unwrap_or(d.len());
    println!("Using r={} components (total {})", r, d.len());
    
    if r == 0 {
        println!("Matrix is completely rank deficient!");
        return Ok(());
    }
    
    let u_r = u.columns(0, r);
    let v = v_t.transpose();
    let v_r = v.columns(0, r);
    let d_r = d.rows(0, r);
    
    println!("d_r values: {:?}", d_r.as_slice());
    
    // Check if any singular values are too small
    if d_r.iter().any(|&val| val < 1e-12) {
        println!("Warning: Very small singular values detected");
    }
    
    // Step by step
    let step1 = u_r.transpose() * &y;
    println!("Step1 (U_r^T * Y): {}x{}", step1.nrows(), step1.ncols());
    
    let step2 = &step1 * &v_r;
    println!("Step2: {}x{}", step2.nrows(), step2.ncols());
    
    let d_inv_diag = DMatrix::from_diagonal(&d_r.map(|d_val| 1.0 / d_val));
    println!("D_inv diagonal values: {:?}", d_inv_diag.diagonal().as_slice());
    
    let atil = &step2 * &d_inv_diag;
    println!("A_tilde: {}x{}", atil.nrows(), atil.ncols());
    println!("{:.6}", atil);
    
    // Check for problematic values
    let has_nan = atil.iter().any(|x: &f64| x.is_nan());
    let has_inf = atil.iter().any(|x: &f64| x.is_infinite());
    
    if has_nan {
        println!("A_tilde contains NaN values!");
    }
    if has_inf {
        println!("A_tilde contains infinite values!");
    }
    if !has_nan && !has_inf {
        println!("A_tilde looks numerically stable");
    }
    
    Ok(())
}
