use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    let stub = rust_bindings::stub_info()?;
    stub.generate()?;
    Ok(())
}
