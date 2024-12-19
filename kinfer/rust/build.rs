use std::env;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("proto");
    std::fs::create_dir_all(&out_dir).unwrap();

    let mut config = prost_build::Config::new();
    config.out_dir(&out_dir);
    config.retain_enum_prefix();
    config.enable_type_names();

    config
        .compile_protos(&["../proto/kinfer.proto"], &["../proto/"])
        .unwrap();
}
