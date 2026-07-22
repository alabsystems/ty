// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use crate::runtime::task;

#[cfg(all(test, feature = "upstream-tests"))]
use std::cell::RefCell;

pub(crate) trait Overflow<T: 'static> {
    fn push(&self, task: task::Notified<T>);

    fn push_batch<I>(&self, iter: I)
    where
        I: Iterator<Item = task::Notified<T>>;
}

#[cfg(all(test, feature = "upstream-tests"))]
impl<T: 'static> Overflow<T> for RefCell<Vec<task::Notified<T>>> {
    fn push(&self, task: task::Notified<T>) {
        self.borrow_mut().push(task);
    }

    fn push_batch<I>(&self, iter: I)
    where
        I: Iterator<Item = task::Notified<T>>,
    {
        self.borrow_mut().extend(iter);
    }
}
