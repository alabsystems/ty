// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

#[cfg(feature = "Wdk_Devices")]
pub mod Devices;
#[cfg(feature = "Wdk_Foundation")]
pub mod Foundation;
#[cfg(feature = "Wdk_Graphics")]
pub mod Graphics;
#[cfg(feature = "Wdk_NetworkManagement")]
pub mod NetworkManagement;
#[cfg(feature = "Wdk_Storage")]
pub mod Storage;
#[cfg(feature = "Wdk_System")]
pub mod System;
