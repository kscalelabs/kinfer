fn main() {
    prost_build::compile_protos(&["protos/kinfer.proto"], &["protos/"]).unwrap();
}
