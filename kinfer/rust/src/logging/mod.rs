//! Public entry-points for everything in `logging/`

pub mod logger;          // exposes logging::logger::*

// optional: re-export the top-level API so callers can do logging::foo()
pub use logger::*;
