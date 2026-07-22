---- MODULE StrongFairness ----
EXTENDS Integers

VARIABLES pc, done

Vars ==
    <<pc, done>>

Init ==
    /\ pc = 0
    /\ done = FALSE

Tick ==
    /\ pc' = 1 - pc
    /\ done' = done

Commit ==
    /\ done = FALSE
    /\ pc = 0
    /\ pc' = 1
    /\ done' = TRUE

StutterDone ==
    /\ done = TRUE
    /\ pc' = pc
    /\ done' = done

Next ==
    \/ Commit
    \/ Tick
    \/ StutterDone

Spec ==
    Init /\ [][Next]_Vars /\ WF_Vars(Next) /\ SF_Vars(Commit)

Live ==
    []<>(done = TRUE)

====
