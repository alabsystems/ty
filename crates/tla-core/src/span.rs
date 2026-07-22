//! Source location tracking for error reporting

use std::fmt;

/// Unique identifier for a source file
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct FileId(pub u32);

impl fmt::Debug for FileId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FileId({})", self.0)
    }
}

/// A span in the source code
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Span {
    /// The file this span is in
    pub file: FileId,
    /// Start byte offset (inclusive)
    pub start: u32,
    /// End byte offset (exclusive)
    pub end: u32,
}

impl Span {
    /// Create a new span
    pub fn new(file: FileId, start: u32, end: u32) -> Self {
        Self { file, start, end }
    }

    /// Create a dummy span for generated code
    pub fn dummy() -> Self {
        Self::default()
    }

    /// The length of this span in bytes
    pub fn len(&self) -> u32 {
        self.end - self.start
    }

    /// Whether this span is empty
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// Merge two spans into one that covers both
    pub fn merge(self, other: Span) -> Span {
        debug_assert_eq!(
            self.file, other.file,
            "Cannot merge spans from different files"
        );
        Span {
            file: self.file,
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

/// A value with an associated source span
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Spanned<T> {
    /// The wrapped value (e.g. an `Expr` or a name).
    pub node: T,
    /// The source location the value was parsed from.
    pub span: Span,
}

impl<T> Spanned<T> {
    /// Wrap `node` with the given source `span`.
    pub fn new(node: T, span: Span) -> Self {
        Self { node, span }
    }

    /// Wrap `node` with a dummy span, for synthesized/generated nodes that have
    /// no real source location.
    pub fn dummy(node: T) -> Self {
        Self {
            node,
            span: Span::dummy(),
        }
    }

    /// Transform the wrapped value with `f`, preserving the span unchanged.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Spanned<U> {
        Spanned {
            node: f(self.node),
            span: self.span,
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Spanned<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} @ {:?}", self.node, self.span)
    }
}
