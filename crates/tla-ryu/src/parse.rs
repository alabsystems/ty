// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use core::fmt::{self, Display};

#[derive(Copy, Clone, Debug)]
pub enum Error {
    InputTooShort,
    InputTooLong,
    MalformedInput,
}

impl Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let msg = match self {
            Error::InputTooShort => "input too short",
            Error::InputTooLong => "input too long",
            Error::MalformedInput => "malformed input",
        };
        formatter.write_str(msg)
    }
}
