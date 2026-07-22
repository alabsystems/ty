// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use crate::io::{Error, ErrorKind, Result};
use core::mem::size_of;
pub const ERROR_ZST_FORBIDDEN: &str = "Collections of zero-sized types are not allowed due to deny-of-service concerns on deserialization.";

pub(crate) fn check_zst<T>() -> Result<()> {
    if size_of::<T>() == 0 {
        return Err(Error::new(ErrorKind::InvalidData, ERROR_ZST_FORBIDDEN));
    }
    Ok(())
}
