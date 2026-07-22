---- MODULE InsertNewlineRangeFunc ----
EXTENDS Integers, Naturals, Sequences

CONSTANTS MaxLines, MaxLineLen

VARIABLES lineLengths, cursorLine, cursorCol

vars == <<lineLengths, cursorLine, cursorCol>>

LineCount == Len(lineLengths)

LineLength(line) ==
    IF line >= 1 /\ line <= LineCount
    THEN lineLengths[line]
    ELSE 0

Init ==
    /\ lineLengths = <<2>>
    /\ cursorLine = 1
    /\ cursorCol \in 0..2

InsertNewline ==
    LET charsAfter == LineLength(cursorLine) - cursorCol
    IN
    /\ LineCount < MaxLines
    \* This range depends on the current sequence length. The resulting
    \* function value is intentionally assigned into the sequence-shaped
    \* lineLengths variable.
    /\ lineLengths' =
        [i \in 1..(LineCount + 1) |->
            IF i < cursorLine THEN lineLengths[i]
            ELSE IF i = cursorLine THEN cursorCol
            ELSE IF i = cursorLine + 1 THEN charsAfter
            ELSE lineLengths[i - 1]]
    /\ cursorLine' = cursorLine + 1
    /\ cursorCol' = 0

Next ==
    \/ InsertNewline
    \/ UNCHANGED vars

TypeOK ==
    /\ Len(lineLengths) \in 1..MaxLines
    /\ \A i \in 1..Len(lineLengths): lineLengths[i] \in 0..MaxLineLen
    /\ cursorLine \in 1..Len(lineLengths)
    /\ cursorCol \in 0..LineLength(cursorLine)

====
