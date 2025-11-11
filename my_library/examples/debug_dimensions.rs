use nalgebra::DMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Simple test case
    let data = DMatrix::from_row_slice(2, 5, &[
        1.0, 2.0, 3.0, 4.0, 5.0,
        2.0, 4.0, 6.0, 8.0, 10.0,
    ]);
    
    println!("Data shape: {}x{}", data.nrows(), data.ncols());
    
    let x = data.columns(0, 4).clone_owned();
    let y = data.columns(1, 4).clone_owned();
    
    println!("X shape: {}x{}", x.nrows(), x.ncols());
    println!("Y shape: {}x{}", y.nrows(), y.ncols());
    
    // SVD
    let svd = x.svd(true, true);
    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();
    let d = svd.singular_values;
    
    println!("U shape: {}x{}", u.nrows(), u.ncols());
    println!("V^T shape: {}x{}", v_t.nrows(), v_t.ncols());
    println!("Singular values: {}", d.len());
    
    let r = d.len().min(2); // Use all or max 2 components
    
    let u_r = u.columns(0, r);
    let v = v_t.transpose();
    let v_r = v.columns(0, r);
    
    println!("U_r shape: {}x{}", u_r.nrows(), u_r.ncols());
    println!("V_r shape: {}x{}", v_r.nrows(), v_r.ncols());
    
    // Test each step carefully
    println!("Testing Step 1: U_r^T * Y");
    let step1 = u_r.transpose() * &y;
    println!("Step 1 result shape: {}x{}", step1.nrows(), step1.ncols());
    
    println!("Testing Step 2: step1 * V_r");
    println!("step1 shape: {}x{}, V_r shape: {}x{}", step1.nrows(), step1.ncols(), v_r.nrows(), v_r.ncols());
    
    // This should be compatible: (2x4) * (4x2) = (2x2)
    if step1.ncols() == v_r.nrows() {
        let step2 = &step1 * &v_r;
        println!("Step 2 result shape: {}x{}", step2.nrows(), step2.ncols());
    } else {
        println!("Dimension mismatch: step1.ncols()={}, v_r.nrows()={}", step1.ncols(), v_r.nrows());
    }
    
    Ok(())
}
